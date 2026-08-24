from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from evaluate_pattern_6541_stage0 import (
    grouped_indices,
    summarize_trials,
    trial_classification,
)
from pattern_encoder_v3 import PatternEncoderV3
from train_pattern_6541_stage1 import embed_records


SHOTS = (6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_stage3_v1_retry2/"
            "dinov2_vits14/best_encoder.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_final_v1/"
            "novel_test_extra_shots_6_8_10.json"
        ),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_final_v1/"
            "novel_test_embeddings.pt"
        ),
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embeddings(
    args: argparse.Namespace, manifest: dict
) -> torch.Tensor:
    records = manifest["novel_test_records"]
    paths = [record["path"] for record in records]
    checkpoint_hash = sha256(args.checkpoint)
    if args.embedding_cache.is_file() and not args.recompute:
        cache = torch.load(
            args.embedding_cache, map_location="cpu", weights_only=False
        )
        if cache["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Embedding cache checkpoint mismatch.")
        if cache["paths"] != paths:
            raise RuntimeError("Embedding cache path mismatch.")
        return cache["embeddings"].float()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["manifest_fingerprint"] != manifest[
        "manifest_fingerprint_sha256"
    ]:
        raise RuntimeError("Checkpoint manifest mismatch.")
    model = PatternEncoderV3(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    embeddings = embed_records(
        model,
        Path(manifest["dataset_root"]),
        records,
        device,
        args.batch_size,
        args.workers,
    )
    args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pattern_6541_novel_test_embeddings_v1",
            "checkpoint_sha256": checkpoint_hash,
            "manifest_fingerprint": manifest[
                "manifest_fingerprint_sha256"
            ],
            "paths": paths,
            "embeddings": embeddings.half(),
        },
        args.embedding_cache,
    )
    model.to("cpu")
    return embeddings


def evaluate_extra_shots(
    records: list[dict],
    embeddings: torch.Tensor,
    seen_names: set[str],
    trials: int,
    seed: int,
) -> dict:
    groups = grouped_indices(records)
    result = {
        "all_eligible": {},
        "common_set": {},
        "common_set_protocol": {
            "minimum_source_groups": 15,
            "support_pool_source_groups": 10,
            "minimum_fixed_query_source_groups": 5,
        },
    }

    for shot in SHOTS:
        patterns = sorted(
            pattern
            for pattern, sources in groups.items()
            if len(sources) >= shot + 2
        )
        shot_trials = []
        for trial in range(trials):
            rng = random.Random(seed + shot * 100_003 + trial)
            support = {}
            query = {}
            for pattern in patterns:
                sources = sorted(groups[pattern])
                rng.shuffle(sources)
                support[pattern] = sources[:shot]
                query[pattern] = sources[shot:]
            shot_trials.append(
                trial_classification(
                    records,
                    embeddings,
                    patterns,
                    support,
                    query,
                    seen_names,
                )
            )
        result["all_eligible"][str(shot)] = summarize_trials(shot_trials)

    common_patterns = sorted(
        pattern for pattern, sources in groups.items() if len(sources) >= 15
    )
    common_trials: dict[int, list[dict]] = {
        shot: [] for shot in SHOTS
    }
    for trial in range(trials):
        rng = random.Random(seed + 900_001 + trial)
        support_pool = {}
        query = {}
        for pattern in common_patterns:
            sources = sorted(groups[pattern])
            rng.shuffle(sources)
            support_pool[pattern] = sources[:10]
            query[pattern] = sources[10:]
        for shot in SHOTS:
            support = {
                pattern: support_pool[pattern][:shot]
                for pattern in common_patterns
            }
            common_trials[shot].append(
                trial_classification(
                    records,
                    embeddings,
                    common_patterns,
                    support,
                    query,
                    seen_names,
                )
            )
    for shot, values in common_trials.items():
        result["common_set"][str(shot)] = summarize_trials(values)
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    embeddings = load_embeddings(args, manifest)
    metrics = evaluate_extra_shots(
        manifest["novel_test_records"],
        embeddings,
        {record["name"] for record in manifest["train_records"]},
        args.trials,
        args.seed,
    )
    report = {
        "format": "pattern_6541_novel_test_extra_shots_v1",
        "posthoc_user_requested": True,
        "model_retrained_or_reselected": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "shots": list(SHOTS),
        "trials": args.trials,
        "candidate_scope": (
            "all eligible novel-test patterns for each shot; not 12-candidate"
        ),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "output": str(args.output.resolve()),
        "all_eligible": {
            shot: {
                "patterns": values["pattern_count"],
                "mean": values["primary_accuracy"]["mean"],
                "std": values["primary_accuracy"]["std"],
            }
            for shot, values in metrics["all_eligible"].items()
        },
        "common_set": {
            shot: {
                "patterns": values["pattern_count"],
                "mean": values["primary_accuracy"]["mean"],
                "std": values["primary_accuracy"]["std"],
            }
            for shot, values in metrics["common_set"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
