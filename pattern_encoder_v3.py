from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f


IMAGE_SIZE = 224
EMBEDDING_DIM = 256
BACKBONE_DIM = 384
BOTTOM_EVAL_FRACTION = 0.45
BOTTOM_TRAIN_RANGE = (0.40, 0.50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def border_color(image: Image.Image) -> tuple[int, int, int]:
    array = np.asarray(image.resize((32, 32)), dtype=np.uint8)
    border = np.concatenate(
        [array[0], array[-1], array[:, 0], array[:, -1]], axis=0
    )
    return tuple(int(value) for value in np.median(border, axis=0))


def preserving_augmentation(image: Image.Image) -> Image.Image:
    fill = border_color(image)
    if random.random() < 0.70:
        width, height = image.size
        maximum_x = max(1, int(round(width * 0.06)))
        maximum_y = max(1, int(round(height * 0.04)))
        image = ImageOps.expand(
            image,
            border=(
                random.randint(0, maximum_x),
                random.randint(0, maximum_y),
                random.randint(0, maximum_x),
                random.randint(0, maximum_y),
            ),
            fill=fill,
        )
    if random.random() < 0.70:
        image = image.rotate(
            random.uniform(-3.0, 3.0),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.82, 1.18)
    )
    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(0.82, 1.18)
    )
    image = ImageEnhance.Color(image).enhance(
        random.uniform(0.65, 1.35)
    )
    if random.random() < 0.12:
        image = image.filter(
            ImageFilter.GaussianBlur(random.uniform(0.1, 0.7))
        )
    return image


def letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    fill = border_color(image)
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = vision_f.resize(
        image,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(
        resized,
        ((size - resized_width) // 2, (size - resized_height) // 2),
    )
    return canvas


def normalized_tensor(
    image: Image.Image, size: int = IMAGE_SIZE
) -> torch.Tensor:
    image = letterbox(image, size=size)
    tensor = vision_f.pil_to_tensor(image).float().div_(255.0)
    return vision_f.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def image_to_views(
    image: Image.Image,
    training: bool = False,
    bottom_eval_fraction: float = BOTTOM_EVAL_FRACTION,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    if training:
        image = preserving_augmentation(image)
        bottom_fraction = random.uniform(*BOTTOM_TRAIN_RANGE)
    else:
        bottom_fraction = bottom_eval_fraction
    width, height = image.size
    bottom_top = max(0, min(height - 1, round(height * (1 - bottom_fraction))))
    bottom = image.crop((0, bottom_top, width, height))
    try:
        return torch.stack(
            [
                normalized_tensor(image, size=image_size),
                normalized_tensor(bottom, size=image_size),
            ]
        )
    finally:
        bottom.close()


class FrozenFeatureBackbone(nn.Module):
    def __init__(
        self,
        kind: str,
        pretrained: bool = True,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.kind = kind
        if kind == "dinov2_vits14":
            self.model = timm.create_model(
                "vit_small_patch14_dinov2",
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
            )
            self.feature_dim = 384
        elif kind == "convnext_tiny_in22k":
            self.model = timm.create_model(
                "convnext_tiny.fb_in22k_ft_in1k",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.feature_dim = 768
        elif kind == "resnet18_imagenet":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {kind}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if output.ndim > 2:
            output = output.flatten(1)
        return torch_f.normalize(output, dim=1)

    def feature_variants(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.kind in {"resnet18_imagenet", "convnext_tiny_in22k"}:
            return {"default": self.forward(images)}
        layers = self.model.get_intermediate_layers(
            images,
            n=4,
            return_prefix_tokens=True,
            norm=True,
        )
        final_patches, final_prefix = layers[-1]
        final_cls = final_prefix[:, 0]
        patch_mean = final_patches.mean(dim=1)
        last4_cls_mean = torch.stack(
            [prefix[:, 0] for _, prefix in layers], dim=0
        ).mean(dim=0)
        last4_patch_mean = torch.stack(
            [patches.mean(dim=1) for patches, _ in layers], dim=0
        ).mean(dim=0)
        cls_normalized = torch_f.normalize(final_cls, dim=1)
        patch_normalized = torch_f.normalize(patch_mean, dim=1)
        return {
            "cls": cls_normalized,
            "patch_mean": patch_normalized,
            "cls_patch_concat": torch_f.normalize(
                torch.cat([cls_normalized, patch_normalized], dim=1), dim=1
            ),
            "last4_cls_mean": torch_f.normalize(last4_cls_mean, dim=1),
            "last4_patch_mean": torch_f.normalize(last4_patch_mean, dim=1),
        }


class PatternEncoderV3(nn.Module):
    def __init__(
        self,
        backbone_kind: str = "dinov2_vits14",
        pretrained: bool = True,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.backbone_kind = backbone_kind
        if backbone_kind == "dinov2_vits14":
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2",
                pretrained=pretrained,
                num_classes=0,
                img_size=image_size,
            )
            self.feature_dim = 384
        elif backbone_kind == "convnext_tiny_in22k":
            self.backbone = timm.create_model(
                "convnext_tiny.fb_in22k_ft_in1k",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported trainable backbone: {backbone_kind}")
        self.full_norm = nn.LayerNorm(self.feature_dim)
        self.bottom_norm = nn.LayerNorm(self.feature_dim)
        self.gate = nn.Linear(self.feature_dim * 2, 2)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(
                torch.tensor([math.log(0.30), math.log(0.70)])
            )
        self.projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(self.feature_dim, EMBEDDING_DIM),
        )

    def backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_kind == "convnext_tiny_in22k":
            return self.backbone(images)
        layers = self.backbone.get_intermediate_layers(
            images,
            n=4,
            return_prefix_tokens=True,
            norm=True,
        )
        return torch.stack(
            [patches.mean(dim=1) for patches, _ in layers], dim=0
        ).mean(dim=0)

    def forward(
        self, views: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, regions, channels, height, width = views.shape
        if regions != 2:
            raise ValueError(f"Expected two views, received {regions}.")
        features = self.backbone_features(
            views.reshape(batch * regions, channels, height, width)
        ).reshape(batch, regions, self.feature_dim)
        return self.fuse_features(
            features[:, 0], features[:, 1], return_gate=return_gate
        )

    def fuse_features(
        self,
        full_features: torch.Tensor,
        bottom_features: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        full = self.full_norm(full_features)
        bottom = self.bottom_norm(bottom_features)
        weights = torch.softmax(self.gate(torch.cat([full, bottom], dim=1)), dim=1)
        fused = full * weights[:, :1] + bottom * weights[:, 1:]
        embeddings = torch_f.normalize(self.projector(fused), dim=1)
        return (embeddings, weights) if return_gate else embeddings

    def set_trainable_stage(self, stage: int) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.full_norm, self.bottom_norm, self.gate, self.projector):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if stage >= 2:
            if self.backbone_kind == "dinov2_vits14":
                modules = [*self.backbone.blocks[-2:], self.backbone.norm]
            else:
                modules = [self.backbone.stages[-1], self.backbone.head.norm]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        if stage >= 3:
            modules = (
                list(self.backbone.blocks[-4:-2])
                if self.backbone_kind == "dinov2_vits14"
                else [self.backbone.stages[-2]]
            )
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True


def fixed_fusion(
    full: torch.Tensor, bottom: torch.Tensor, full_weight: float = 0.30
) -> torch.Tensor:
    full = torch_f.normalize(full, dim=1)
    bottom = torch_f.normalize(bottom, dim=1)
    return torch_f.normalize(
        full * full_weight + bottom * (1.0 - full_weight), dim=1
    )
