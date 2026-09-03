from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from . import host_ops as ops
from .geometry import mask_to_quad_ransac, order_quad

PACKAGE = Path(__file__).resolve().parents[1]
THRESHOLD = 0.9267578125
PAD_RGB = (123.675, 116.28, 103.53)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def assert_fp32(value, name):
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise RuntimeError(f"{name} is not finite FP32")


def audit_state(model):
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if not value.is_floating_point():
            continue
        if value.dtype != torch.float32:
            raise RuntimeError(f"Non-FP32 model tensor: {name}")
        if name != "decoder.anchors" and not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite model tensor: {name}")


class ChekiEdgeFitRT:
    """One frozen detector, one box-prompted segmenter, and unchanged G."""

    def __init__(self, weights_dir=None, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError("Only CPU/CUDA FP32 execution is supported by this release")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        configure_fp32()
        self.device = torch.device(device)
        manifest = json.loads((PACKAGE / "checkpoints.json").read_text(encoding="utf-8"))
        weights_dir = Path(weights_dir) if weights_dir else PACKAGE / "checkpoints"
        paths = {}
        for item in manifest["assets"]:
            path = weights_dir / item["name"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise RuntimeError(f"Missing or changed checkpoint: {path}. Run download_checkpoints.py.")
            paths[item["role"]] = path
        upstream = PACKAGE / "third_party"
        rt_root = upstream / "rtdetrv2_pytorch"
        sys.path.insert(0, str(rt_root))
        import src
        from src.core import YAMLConfig
        if not Path(src.__file__).resolve().is_relative_to(rt_root.resolve()):
            raise RuntimeError("Conflicting 'src' module; use a fresh process for this runtime")
        config = YAMLConfig(str(rt_root / "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"),
                            num_classes=1, eval_spatial_size=[768, 768], PResNet={"pretrained": False})
        detector = config.model.eval()
        expected_anchors = detector.decoder.anchors.detach().clone()
        expected_valid = detector.decoder.valid_mask.detach().clone()
        # The original checkpoints contain optimizer/RNG state: load only after SHA verification.
        state = torch.load(paths["detector"], map_location="cpu", weights_only=False)
        detector.load_state_dict(state["model"], strict=True)
        if not torch.equal(expected_anchors, detector.decoder.anchors) or not torch.equal(expected_valid, detector.decoder.valid_mask):
            raise RuntimeError("Official anchor values or invalid-slot sentinel pattern changed")
        del state, config
        audit_state(detector)
        self.detector = detector.to(self.device).eval()

        sys.path.insert(0, str(upstream / "EdgeSAM"))
        ops.install_optional_training_stubs()
        from edge_sam import SamPredictor, sam_model_registry
        # Full production state strictly replaces every parameter/buffer. No base-weight download.
        segmenter = sam_model_registry["edge_sam"](checkpoint=None)
        state = torch.load(paths["segmenter"], map_location="cpu", weights_only=False)
        segmenter.load_state_dict(state["model"], strict=True)
        del state
        audit_state(segmenter)
        self.segmenter = segmenter.to(self.device).eval()
        self.predictor = SamPredictor(self.segmenter)

    @torch.inference_mode()
    def detect_rgb(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Detector expects upright HWC RGB uint8 pixels")
        height, width = rgb.shape[:2]
        letterbox, _, _, _ = ops.letterbox_image(rgb, 768)
        image = ops.normalized_images(letterbox.transpose(2, 0, 1)[None].copy(), self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            output = self.detector(image)
        assert_fp32(output["pred_logits"], "detector logits")
        assert_fp32(output["pred_boxes"], "detector boxes")
        scores = output["pred_logits"][0, :, 0].sigmoid().cpu().numpy()
        boxes = ops.cxcywh_to_xyxy(output["pred_boxes"][0].cpu().numpy())
        return [{"query_id": int(index), "score": float(scores[index]),
                 "box_xyxy": ops.restore_original_box(boxes[index], width, height).tolist()}
                for index in np.flatnonzero(scores >= THRESHOLD)]

    @torch.inference_mode()
    def segment_box(self, rgb_fp32, box):
        box = np.asarray(box, dtype=np.float32)
        roi = ops.square20_fp32(box)
        side, matrix = ops.square_transform_fp32(roi)
        crop = cv2.warpPerspective(rgb_fp32, matrix, (side, side), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=PAD_RGB)
        new_h, new_w = self.predictor.transform.get_preprocess_shape(side, side, self.segmenter.image_encoder.img_size)
        resized = ops.resize_rgb_fp32(crop, new_h, new_w)
        pixels = torch.from_numpy(resized.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        x1, y1, x2, y2 = box
        corners = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        transformed = ops.transform_points_fp32(corners, matrix)
        prompt = np.asarray([*transformed.min(axis=0), *transformed.max(axis=0)], np.float32)
        scale = np.asarray([new_w, new_h, new_w, new_h], np.float32) / np.float32(side)
        box_tensor = torch.from_numpy((prompt * scale)[None]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            features = self.predictor.set_torch_image(pixels, (side, side))
            masks, quality, low_res = self.predictor.predict_torch(
                features=None, point_coords=None, point_labels=None, boxes=box_tensor,
                mask_input=None, num_multimask_outputs=1, return_logits=True)
        for value, name in ((features, "embedding"), (masks, "mask"),
                            (quality, "quality"), (low_res, "low-resolution mask")):
            assert_fp32(value, name)
        binary = (masks[0, 0] > self.segmenter.mask_threshold).cpu().numpy()
        quad, error = None, None
        try:
            crop_quad = order_quad(mask_to_quad_ransac(binary))
            inverse = np.linalg.inv(matrix.astype(np.float64))
            original = order_quad(ops.transform_points(crop_quad, inverse))
            if not np.isfinite(original).all():
                raise ValueError("Nonfinite final quadrilateral")
            quad = np.round(original, 6).tolist()
        except MemoryError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {"quad": quad, "error": error, "edge_sam_quality_score": float(quality[0, 0]),
                "square20_roi": roi.tolist()}

    def predict_rgb(self, rgb):
        boxes = self.detect_rgb(rgb)
        rgb_fp32 = rgb.astype(np.float32)
        predictions = [dict(item, **self.segment_box(rgb_fp32, item["box_xyxy"])) for item in boxes]
        return {"algorithm": "ChekiEdgeFit-RT v2", "threshold": THRESHOLD,
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "predictions": predictions}

    def predict(self, image_path):
        with Image.open(image_path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        return self.predict_rgb(rgb)
