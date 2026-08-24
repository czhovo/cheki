from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_pattern_6541_stage0 import (
    evaluate_novel_dev,
    evaluate_seen_validation,
)
from pattern_encoder_v3 import PatternEncoderV3
from train_pattern_6541_stage1 import embed_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--mode", choices=("validation", "locked-test"), default="validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
        raise RuntimeError("Checkpoint manifest mismatch.")
    backbone = checkpoint.get("backbone", "dinov2_vits14")
    model = PatternEncoderV3(backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    train_records = manifest["train_records"]
    if args.mode == "validation":
        query_records = manifest["validation_records"]
        novel_records = manifest["novel_dev_records"]
    else:
        query_records = manifest["test_records"]
        novel_records = manifest["novel_test_records"]
    all_records = train_records + query_records + novel_records
    print(
        json.dumps(
            {
                "mode": args.mode,
                "checkpoint": str(args.checkpoint.resolve()),
                "records": len(all_records),
                "locked_test_evaluated": args.mode == "locked-test",
            }
        ),
        flush=True,
    )
    embeddings = embed_records(
        model,
        Path(manifest["dataset_root"]),
        all_records,
        device,
        args.batch_size,
        args.workers,
    )
    train_stop = len(train_records)
    query_stop = train_stop + len(query_records)
    seen = evaluate_seen_validation(
        train_records,
        embeddings[:train_stop],
        query_records,
        embeddings[train_stop:query_stop],
    )
    novel = evaluate_novel_dev(
        novel_records,
        embeddings[query_stop:],
        {record["name"] for record in train_records},
        args.novel_trials,
        args.seed,
    )
    one = novel["common_set"]["1"]["primary_accuracy"]["mean"]
    five = novel["common_set"]["5"]["primary_accuracy"]["mean"]
    report = {
        "format": "pattern_6541_checkpoint_evaluation_v1",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_format": checkpoint.get("format"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "locked_test_evaluated": args.mode == "locked-test",
        "test_used_for_model_selection": False,
        "selection_score": (
            0.5 * seen["macro_accuracy"] + 0.25 * one + 0.25 * five
        ),
        "seen": seen,
        "novel_fewshot": novel,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "mode": args.mode,
                "seen_top1": seen["top1_accuracy"],
                "seen_macro": seen["macro_accuracy"],
                "novel_common_1shot": one,
                "novel_common_5shot": five,
                "score": report["selection_score"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
