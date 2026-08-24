from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from pattern_encoder_v3 import FrozenFeatureBackbone, image_to_views, load_rgb


class AugmentedViewDataset(Dataset):
    def __init__(
        self,
        root: Path,
        records: list[dict],
        variants: int,
        seed: int,
    ) -> None:
        self.root = root
        self.records = records
        self.variants = variants
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.variants

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record_index, variant = divmod(index, self.variants)
        random.seed(self.seed + record_index * 10_007 + variant * 1_000_003)
        image = load_rgb(self.root / self.records[record_index]["path"])
        try:
            return (
                image_to_views(image, training=True),
                record_index,
                variant,
            )
        finally:
            image.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--backbone",
        choices=("dinov2_vits14", "convnext_tiny_in22k"),
        required=True,
    )
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_stage1_feature_banks"),
    )
    return parser.parse_args()


def load_eval_features(
    backbone: str, expected_paths: list[str]
) -> tuple[torch.Tensor, torch.Tensor, Path]:
    cache_path = (
        Path("evaluations/pattern_6541_stage0_v1")
        / backbone
        / "features.pt"
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["paths"] != expected_paths:
        raise RuntimeError("Stage0 cache paths do not match manifest records.")
    variants = cache.get("feature_variants")
    if variants is None:
        key = "cls" if backbone == "dinov2_vits14" else "default"
        variants = {key: {"full": cache["full"], "bottom": cache["bottom"]}}
    key = "last4_patch_mean" if backbone == "dinov2_vits14" else "default"
    if key not in variants:
        raise RuntimeError(f"Required Stage0 feature is missing: {key}")
    return variants[key]["full"], variants[key]["bottom"], cache_path


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    train_records = manifest["train_records"]
    eval_records = (
        train_records
        + manifest["validation_records"]
        + manifest["novel_dev_records"]
    )
    eval_paths = [record["path"] for record in eval_records]
    eval_full, eval_bottom, eval_cache_path = load_eval_features(
        args.backbone, eval_paths
    )
    device = torch.device(args.device)
    model = FrozenFeatureBackbone(args.backbone, pretrained=True).to(device).eval()
    feature_key = (
        "last4_patch_mean" if args.backbone == "dinov2_vits14" else "default"
    )
    feature_dim = model.feature_dim
    full = torch.empty(
        (len(train_records), args.variants, feature_dim), dtype=torch.float16
    )
    bottom = torch.empty_like(full)
    loader = DataLoader(
        AugmentedViewDataset(root, train_records, args.variants, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=1,
    )
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (views, record_indices, variant_indices) in enumerate(
        loader, start=1
    ):
        views = views.to(device, non_blocking=True)
        batch, regions, channels, height, width = views.shape
        extracted = model.feature_variants(
            views.reshape(batch * regions, channels, height, width)
        )[feature_key].reshape(batch, regions, feature_dim)
        full[record_indices, variant_indices] = extracted[:, 0].cpu().half()
        bottom[record_indices, variant_indices] = extracted[:, 1].cpu().half()
        if batch_index % 20 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "batches": len(loader),
                        "encoded_augmented_samples": min(
                            batch_index * args.batch_size, len(loader.dataset)
                        ),
                        "total_augmented_samples": len(loader.dataset),
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    output_dir = args.output_root / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "feature_bank.pt"
    torch.save(
        {
            "format": "pattern_6541_stage1_feature_bank_v1",
            "backbone": args.backbone,
            "backbone_feature": feature_key,
            "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
            "seed": args.seed,
            "augmentation_variants": args.variants,
            "train_paths": [record["path"] for record in train_records],
            "train_full": full,
            "train_bottom": bottom,
            "eval_paths": eval_paths,
            "eval_full": eval_full.half(),
            "eval_bottom": eval_bottom.half(),
            "eval_stage0_cache": str(eval_cache_path.resolve()),
        },
        output,
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "backbone": args.backbone,
                "train_shape": list(full.shape),
                "eval_shape": list(eval_full.shape),
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / 2**30,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                / 2**30,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
