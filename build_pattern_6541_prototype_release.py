from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = next(
    (
        candidate
        for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents)
        if (candidate / "pattern_encoder_v3.py").exists()
        and (candidate / "training_plans" / "pattern_6541_v1.json").exists()
    ),
    SCRIPT_DIR,
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pattern_encoder_v3 import PatternEncoderV3, image_to_views, load_rgb


DEFAULT_MANIFEST = ROOT / "training_plans" / "pattern_6541_v1.json"
DEFAULT_CHECKPOINT = (
    ROOT
    / "evaluations"
    / "pattern_6541_stage3_v1_retry2"
    / "dinov2_vits14"
    / "best_encoder.pt"
)
DEFAULT_RELEASE = ROOT / "release" / "pattern_6541_encoder_v1"
DEFAULT_CACHE_DIR = ROOT / "evaluations" / "pattern_6541_prototype_release_v1"
SPLIT_KEYS = (
    "train_records",
    "validation_records",
    "test_records",
    "novel_dev_records",
    "novel_test_records",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class ManifestImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, dataset_root: Path, records: list[dict[str, Any]]) -> None:
        self.dataset_root = dataset_root
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        image = load_rgb(self.dataset_root / record["path"])
        try:
            views = image_to_views(image, training=False)
        finally:
            image.close()
        return views, record["path"]


def load_encoder(checkpoint_path: Path, device: torch.device) -> tuple[PatternEncoderV3, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "pattern_6541_stage3_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')}")
    backbone = checkpoint.get("backbone")
    model = PatternEncoderV3(backbone_kind=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, checkpoint


@torch.inference_mode()
def encode_records(
    model: PatternEncoderV3,
    dataset_root: Path,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[str], torch.Tensor]:
    loader = DataLoader(
        ManifestImageDataset(dataset_root, records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    paths: list[str] = []
    chunks: list[torch.Tensor] = []
    for batch_index, (views, batch_paths) in enumerate(loader, start=1):
        views = views.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            embeddings = model(views)
        chunks.append(F.normalize(embeddings.float(), dim=1).cpu())
        paths.extend(batch_paths)
        if batch_index == 1 or batch_index % 10 == 0 or len(paths) == len(records):
            print(f"Encoded {len(paths):>4}/{len(records)} images", flush=True)
    return paths, torch.cat(chunks, dim=0)


def load_embedding_cache(
    path: Path,
    expected_records: list[dict[str, Any]],
    checkpoint_sha256: str,
    manifest_fingerprint: str,
) -> tuple[list[str], torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    expected_paths = [record["path"] for record in expected_records]
    paths = list(data["paths"])
    if paths != expected_paths:
        raise ValueError(f"Cache path order/content mismatch: {path}")
    if data.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Cache checkpoint mismatch: {path}")
    if data.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError(f"Cache manifest mismatch: {path}")
    embeddings = F.normalize(data["embeddings"].float(), dim=1)
    if embeddings.shape != (len(paths), 256):
        raise ValueError(f"Unexpected embedding shape in {path}: {embeddings.shape}")
    return paths, embeddings


def add_embeddings(
    destination: dict[str, torch.Tensor], paths: list[str], embeddings: torch.Tensor
) -> None:
    for path, embedding in zip(paths, embeddings, strict=True):
        if path in destination:
            raise ValueError(f"Duplicate embedded path: {path}")
        destination[path] = embedding


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the >=30-image pattern prototype release."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--minimum-images", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    release_dir = args.release_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest = read_json(manifest_path)
    dataset_root = Path(manifest["dataset_root"]).resolve()
    manifest_fingerprint = manifest["manifest_fingerprint_sha256"]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_header.get("manifest_fingerprint") != manifest_fingerprint:
        raise ValueError("Checkpoint and manifest fingerprints differ.")

    all_records = [record for split in SPLIT_KEYS for record in manifest[split]]
    record_by_path = {record["path"]: record for record in all_records}
    if len(record_by_path) != len(all_records):
        raise ValueError("Manifest contains duplicate paths across splits.")

    embedding_by_path: dict[str, torch.Tensor] = {}
    fixed_caches = [
        (
            ROOT / "evaluations/pattern_6541_candidate12_known_v1/seen_train_test_embeddings.pt",
            manifest["train_records"] + manifest["test_records"],
        ),
        (
            ROOT / "evaluations/pattern_6541_candidate12_unknown_v1/seen_validation_embeddings.pt",
            manifest["validation_records"],
        ),
        (
            ROOT / "evaluations/pattern_6541_final_v1/novel_test_embeddings.pt",
            manifest["novel_test_records"],
        ),
    ]
    for cache_path, records in fixed_caches:
        paths, embeddings = load_embedding_cache(
            cache_path,
            records,
            checkpoint_sha256,
            manifest_fingerprint,
        )
        add_embeddings(embedding_by_path, paths, embeddings)
        print(f"Loaded {len(paths):>4} embeddings from {cache_path.name}")

    novel_dev_cache = cache_dir / "novel_dev_embeddings.pt"
    novel_dev_records = manifest["novel_dev_records"]
    if novel_dev_cache.exists():
        paths, embeddings = load_embedding_cache(
            novel_dev_cache,
            novel_dev_records,
            checkpoint_sha256,
            manifest_fingerprint,
        )
        print(f"Loaded {len(paths):>4} embeddings from {novel_dev_cache.name}")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        model, _ = load_encoder(checkpoint_path, device)
        print(f"Encoding novel-dev on {device} ...", flush=True)
        paths, embeddings = encode_records(
            model,
            dataset_root,
            novel_dev_records,
            device,
            args.batch_size,
            args.workers,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "pattern_6541_embedding_cache_v1",
                "checkpoint_sha256": checkpoint_sha256,
                "manifest_fingerprint": manifest_fingerprint,
                "paths": paths,
                "embeddings": embeddings.half(),
            },
            novel_dev_cache,
        )
        print(f"Saved cache: {novel_dev_cache}")
    add_embeddings(embedding_by_path, paths, embeddings)

    if set(embedding_by_path) != set(record_by_path):
        missing = set(record_by_path) - set(embedding_by_path)
        extra = set(embedding_by_path) - set(record_by_path)
        raise ValueError(f"Embedding coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    qualifying_ids = sorted(
        pattern_id
        for pattern_id, info in manifest["per_pattern"].items()
        if int(info["usable_instances"]) >= args.minimum_images
    )
    prototypes: list[torch.Tensor] = []
    catalog: list[dict[str, Any]] = []
    selected_records: dict[str, list[dict[str, Any]]] = {}

    for pattern_id in qualifying_ids:
        info = manifest["per_pattern"][pattern_id]
        records = [r for r in all_records if r["pattern"] == pattern_id]
        if len(records) != int(info["usable_instances"]):
            raise ValueError(f"Image count mismatch for {pattern_id}")
        by_source: dict[str, list[torch.Tensor]] = defaultdict(list)
        for record in records:
            by_source[record["source"]].append(embedding_by_path[record["path"]])
        if len(by_source) != int(info["source_groups"]):
            raise ValueError(f"Source-group count mismatch for {pattern_id}")

        source_embeddings = []
        for source in sorted(by_source):
            source_embeddings.append(
                F.normalize(torch.stack(by_source[source]).mean(dim=0), dim=0)
            )
        prototype = F.normalize(torch.stack(source_embeddings).mean(dim=0), dim=0)
        prototypes.append(prototype)
        catalog.append(
            {
                "index": len(catalog),
                "pattern_id": pattern_id,
                "name": info["name"],
                "image_count": len(records),
                "source_group_count": len(by_source),
                "original_split": info["split"],
            }
        )
        selected_records[pattern_id] = records

    prototype_tensor = torch.stack(prototypes).float()
    norms = prototype_tensor.norm(dim=1)
    if not torch.isfinite(prototype_tensor).all() or not torch.allclose(
        norms, torch.ones_like(norms), atol=1e-5
    ):
        raise ValueError("Prototype bank contains invalid or non-unit vectors.")

    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "model").mkdir(exist_ok=True)
    (release_dir / "prototypes").mkdir(exist_ok=True)
    (release_dir / "metadata").mkdir(exist_ok=True)

    bank = {
        "format": "pattern_prototype_bank_v1",
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "manifest_fingerprint_sha256": manifest_fingerprint,
        "minimum_usable_instances_inclusive": args.minimum_images,
        "embedding_dim": 256,
        "similarity": "cosine",
        "aggregation": "L2(mean over images in source), then L2(mean over sources)",
        "pattern_ids": qualifying_ids,
        "pattern_names": [item["name"] for item in catalog],
        "image_counts": torch.tensor([item["image_count"] for item in catalog]),
        "source_group_counts": torch.tensor(
            [item["source_group_count"] for item in catalog]
        ),
        "prototypes": prototype_tensor,
    }
    torch.save(bank, release_dir / "prototypes" / "pattern_prototypes_ge30.pt")
    write_json(
        release_dir / "prototypes" / "pattern_prototypes_ge30.json",
        {
            key: value
            for key, value in bank.items()
            if key not in {"image_counts", "source_group_counts", "prototypes"}
        }
        | {
            "image_counts": bank["image_counts"].tolist(),
            "source_group_counts": bank["source_group_counts"].tolist(),
            "prototypes": prototype_tensor.tolist(),
        },
    )
    write_json(release_dir / "metadata" / "prototype_catalog.json", catalog)
    write_json(
        release_dir / "metadata" / "prototype_source_records.json",
        {
            "format": "pattern_prototype_source_records_v1",
            "minimum_usable_instances_inclusive": args.minimum_images,
            "records_by_pattern": selected_records,
        },
    )

    shutil.copy2(checkpoint_path, release_dir / "model" / "pattern_encoder_v3.pt")
    shutil.copy2(ROOT / "pattern_encoder_v3.py", release_dir / "model" / "pattern_encoder_v3.py")
    shutil.copy2(manifest_path, release_dir / "metadata" / "training_manifest.json")
    packaged_builder = release_dir / "build_prototypes.py"
    if Path(__file__).resolve() != packaged_builder.resolve():
        shutil.copy2(Path(__file__), packaged_builder)

    build_manifest = {
        "format": "pattern_6541_encoder_release_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pattern_count": len(qualifying_ids),
        "image_count": sum(item["image_count"] for item in catalog),
        "source_group_count": sum(item["source_group_count"] for item in catalog),
        "minimum_usable_instances_inclusive": args.minimum_images,
        "model": {
            "architecture": "DINOv2 ViT-S/14 dual-view metric encoder",
            "embedding_dim": 256,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "data": {
            "manifest_fingerprint_sha256": manifest_fingerprint,
            "prototype_uses_all_available_splits": list(SPLIT_KEYS),
            "prototype_uses_all_available_instances": True,
        },
        "prototype_algorithm": bank["aggregation"],
    }
    write_json(release_dir / "release_manifest.json", build_manifest)

    catalog_lines = [
        "# Prototype 目录（可用图片数 ≥ 30）",
        "",
        f"共 {len(catalog)} 个 pattern，{build_manifest['image_count']} 张图片，"
        f"{build_manifest['source_group_count']} 个 source group。",
        "",
        "| Index | Pattern ID | Name | 图片数 | Source group | 原划分 |",
        "|---:|---|---|---:|---:|---|",
    ]
    for item in catalog:
        catalog_lines.append(
            f"| {item['index']} | {item['pattern_id']} | {item['name']} | "
            f"{item['image_count']} | {item['source_group_count']} | {item['original_split']} |"
        )
    (release_dir / "PROTOTYPE_CATALOG.md").write_text(
        "\n".join(catalog_lines) + "\n", encoding="utf-8"
    )

    checksums = []
    for path in sorted(p for p in release_dir.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        checksums.append(f"{sha256_file(path)}  {path.relative_to(release_dir).as_posix()}")
    (release_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    print(
        f"Built {len(catalog)} prototypes from {build_manifest['image_count']} images / "
        f"{build_manifest['source_group_count']} sources."
    )
    print(f"Release: {release_dir}")


if __name__ == "__main__":
    main()
