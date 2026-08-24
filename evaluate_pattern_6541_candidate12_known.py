from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from pattern_encoder_v3 import PatternEncoderV3
from train_pattern_6541_stage1 import embed_records


SHOTS = (1, 2, 3, 5, 6, 8, 10)


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
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_candidate12_known_v1"),
    )
    parser.add_argument(
        "--novel-cache",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_final_v1/"
            "novel_test_embeddings.pt"
        ),
    )
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--recompute-seen", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_indices(records: list[dict]) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        result[record["pattern"]][record["source"]].append(index)
    return result


def source_mean(features: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return torch.nn.functional.normalize(features[indices].mean(0), dim=0)


def prototype(
    features: torch.Tensor,
    sources: dict[str, list[int]],
    selected_sources: list[str],
) -> torch.Tensor:
    values = [source_mean(features, sources[source]) for source in selected_sources]
    return torch.nn.functional.normalize(torch.stack(values).mean(0), dim=0)


def load_model(args: argparse.Namespace, manifest: dict) -> PatternEncoderV3:
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["manifest_fingerprint"] != manifest[
        "manifest_fingerprint_sha256"
    ]:
        raise RuntimeError("Checkpoint manifest mismatch.")
    model = PatternEncoderV3(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_seen_embeddings(
    args: argparse.Namespace, manifest: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = args.output_root / "seen_train_test_embeddings.pt"
    train_records = manifest["train_records"]
    test_records = manifest["test_records"]
    all_records = train_records + test_records
    paths = [record["path"] for record in all_records]
    checkpoint_hash = sha256(args.checkpoint)
    if cache_path.is_file() and not args.recompute_seen:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Seen cache checkpoint mismatch.")
        if cache["paths"] != paths:
            raise RuntimeError("Seen cache path mismatch.")
        embeddings = cache["embeddings"].float()
    else:
        model = load_model(args, manifest)
        device = torch.device(args.device)
        model.to(device).eval()
        embeddings = embed_records(
            model,
            Path(manifest["dataset_root"]),
            all_records,
            device,
            args.batch_size,
            args.workers,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "pattern_6541_seen_train_test_embeddings_v1",
                "checkpoint_sha256": checkpoint_hash,
                "manifest_fingerprint": manifest[
                    "manifest_fingerprint_sha256"
                ],
                "paths": paths,
                "train_count": len(train_records),
                "embeddings": embeddings.half(),
            },
            cache_path,
        )
        model.to("cpu")
    split = len(train_records)
    return embeddings[:split], embeddings[split:]


def load_novel_embeddings(
    args: argparse.Namespace, manifest: dict
) -> torch.Tensor:
    cache = torch.load(
        args.novel_cache, map_location="cpu", weights_only=False
    )
    if cache["checkpoint_sha256"] != sha256(args.checkpoint):
        raise RuntimeError("Novel cache checkpoint mismatch.")
    if cache["paths"] != [
        record["path"] for record in manifest["novel_test_records"]
    ]:
        raise RuntimeError("Novel cache path mismatch.")
    return cache["embeddings"].float()


def known_trial(
    support_records: list[dict],
    support_features: torch.Tensor,
    query_records: list[dict],
    query_features: torch.Tensor,
    shot: int,
    candidate_count: int,
    rng: random.Random,
    shared_pool: bool,
) -> dict:
    support_groups = grouped_indices(support_records)
    query_groups = support_groups if shared_pool else grouped_indices(query_records)
    eligible = sorted(
        pattern
        for pattern, sources in support_groups.items()
        if len(sources) >= (shot + 2 if shared_pool else shot)
        and pattern in query_groups
    )
    if len(eligible) < candidate_count:
        raise RuntimeError(
            f"Only {len(eligible)} patterns eligible for {shot}-shot."
        )
    candidates = rng.sample(eligible, candidate_count)
    prototypes = []
    query_indices = []
    truth = []
    per_pattern_counts = []
    for label, pattern in enumerate(candidates):
        sources = sorted(support_groups[pattern])
        rng.shuffle(sources)
        selected = sources[:shot]
        prototypes.append(
            prototype(support_features, support_groups[pattern], selected)
        )
        before = len(query_indices)
        if shared_pool:
            for source in sources[shot:]:
                indices = support_groups[pattern][source]
                query_indices.extend(indices)
                truth.extend([label] * len(indices))
        else:
            for indices in query_groups[pattern].values():
                query_indices.extend(indices)
                truth.extend([label] * len(indices))
        per_pattern_counts.append(len(query_indices) - before)
    prototype_tensor = torch.stack(prototypes)
    predictions = (query_features[query_indices] @ prototype_tensor.T).argmax(1)
    truth_tensor = torch.tensor(truth)
    correct = predictions.eq(truth_tensor)
    per_pattern_accuracy = []
    start = 0
    for count in per_pattern_counts:
        per_pattern_accuracy.append(float(correct[start : start + count].float().mean()))
        start += count
    return {
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": float(np.mean(per_pattern_accuracy)),
        "query_count": len(query_indices),
        "candidates": candidates,
    }


def fixed_candidate_trial(
    support_groups: dict[str, dict[str, list[int]]],
    support_features: torch.Tensor,
    query_groups: dict[str, dict[str, list[int]]],
    query_features: torch.Tensor,
    candidates: list[str],
    support_pool: dict[str, list[str]],
    fixed_query_sources: dict[str, list[str]] | None,
    shot: int,
) -> dict:
    prototypes = []
    query_indices = []
    truth = []
    per_pattern_counts = []
    for label, pattern in enumerate(candidates):
        prototypes.append(
            prototype(
                support_features,
                support_groups[pattern],
                support_pool[pattern][:shot],
            )
        )
        before = len(query_indices)
        sources = (
            fixed_query_sources[pattern]
            if fixed_query_sources is not None
            else sorted(query_groups[pattern])
        )
        for source in sources:
            indices = query_groups[pattern][source]
            query_indices.extend(indices)
            truth.extend([label] * len(indices))
        per_pattern_counts.append(len(query_indices) - before)
    prototype_tensor = torch.stack(prototypes)
    predictions = (query_features[query_indices] @ prototype_tensor.T).argmax(1)
    truth_tensor = torch.tensor(truth)
    correct = predictions.eq(truth_tensor)
    per_pattern_accuracy = []
    start = 0
    for count in per_pattern_counts:
        per_pattern_accuracy.append(
            float(correct[start : start + count].float().mean())
        )
        start += count
    return {
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": float(np.mean(per_pattern_accuracy)),
        "query_count": len(query_indices),
        "candidates": candidates,
    }


def full_prototype_seen_trial(
    support_records: list[dict],
    support_features: torch.Tensor,
    query_records: list[dict],
    query_features: torch.Tensor,
    candidates: list[str],
) -> dict:
    support_groups = grouped_indices(support_records)
    query_groups = grouped_indices(query_records)
    prototypes = []
    query_indices = []
    truth = []
    per_pattern_counts = []
    for label, pattern in enumerate(candidates):
        all_sources = sorted(support_groups[pattern])
        prototypes.append(
            prototype(
                support_features,
                support_groups[pattern],
                all_sources,
            )
        )
        before = len(query_indices)
        for source in sorted(query_groups[pattern]):
            indices = query_groups[pattern][source]
            query_indices.extend(indices)
            truth.extend([label] * len(indices))
        per_pattern_counts.append(len(query_indices) - before)
    prototype_tensor = torch.stack(prototypes)
    predictions = (query_features[query_indices] @ prototype_tensor.T).argmax(1)
    truth_tensor = torch.tensor(truth)
    correct = predictions.eq(truth_tensor)
    per_pattern_accuracy = []
    start = 0
    for count in per_pattern_counts:
        per_pattern_accuracy.append(
            float(correct[start : start + count].float().mean())
        )
        start += count
    return {
        "accuracy": float(correct.float().mean()),
        "macro_accuracy": float(np.mean(per_pattern_accuracy)),
        "query_count": len(query_indices),
        "candidates": candidates,
    }


def run_seen_full_prototypes(
    support_records: list[dict],
    support_features: torch.Tensor,
    query_records: list[dict],
    query_features: torch.Tensor,
    args: argparse.Namespace,
) -> dict:
    support_patterns = sorted({record["pattern"] for record in support_records})
    query_patterns = {record["pattern"] for record in query_records}
    eligible = [pattern for pattern in support_patterns if pattern in query_patterns]
    trials = []
    for trial in range(args.trials):
        rng = random.Random(args.seed + 10_000 + trial)
        candidates = rng.sample(eligible, args.candidate_count)
        trials.append(
            full_prototype_seen_trial(
                support_records,
                support_features,
                query_records,
                query_features,
                candidates,
            )
        )
    result = {
        "eligible_pattern_count": len(eligible),
        "prototype_scope": "all seen-train source groups per candidate pattern",
        "accuracy": summarize([trial["accuracy"] for trial in trials]),
        "macro_accuracy": summarize(
            [trial["macro_accuracy"] for trial in trials]
        ),
        "mean_query_count": float(
            np.mean([trial["query_count"] for trial in trials])
        ),
        "candidate_examples": [trial["candidates"] for trial in trials[:3]],
    }
    print(
        json.dumps(
            {
                "domain": "seen",
                "prototype_scope": "full_train",
                "accuracy": result["accuracy"]["mean"],
                "macro_accuracy": result["macro_accuracy"]["mean"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_domain(
    domain: str,
    support_records: list[dict],
    support_features: torch.Tensor,
    query_records: list[dict],
    query_features: torch.Tensor,
    shared_pool: bool,
    args: argparse.Namespace,
) -> dict:
    result = {}
    offset = 10_000 if domain == "seen" else 20_000
    support_groups = grouped_indices(support_records)
    query_groups = support_groups if shared_pool else grouped_indices(query_records)
    maximum_shot = max(SHOTS)
    eligible = sorted(
        pattern
        for pattern, sources in support_groups.items()
        if len(sources) >= (maximum_shot + 2 if shared_pool else maximum_shot)
        and pattern in query_groups
    )
    if len(eligible) < args.candidate_count:
        raise RuntimeError("Insufficient patterns for paired 10-shot trials.")
    trials_by_shot: dict[int, list[dict]] = {
        shot: [] for shot in SHOTS
    }
    for trial in range(args.trials):
        rng = random.Random(args.seed + offset + trial)
        candidates = rng.sample(eligible, args.candidate_count)
        support_pool = {}
        fixed_query_sources = {} if shared_pool else None
        for pattern in candidates:
            sources = sorted(support_groups[pattern])
            rng.shuffle(sources)
            support_pool[pattern] = sources[:maximum_shot]
            if shared_pool:
                fixed_query_sources[pattern] = sources[maximum_shot:]
        for shot in SHOTS:
            trials_by_shot[shot].append(
                fixed_candidate_trial(
                    support_groups,
                    support_features,
                    query_groups,
                    query_features,
                    candidates,
                    support_pool,
                    fixed_query_sources,
                    shot,
                )
            )
    for shot in SHOTS:
        trials = trials_by_shot[shot]
        result[str(shot)] = {
            "accuracy": summarize([trial["accuracy"] for trial in trials]),
            "macro_accuracy": summarize(
                [trial["macro_accuracy"] for trial in trials]
            ),
            "mean_query_count": float(
                np.mean([trial["query_count"] for trial in trials])
            ),
            "candidate_examples": [
                trial["candidates"] for trial in trials[:3]
            ],
        }
        print(
            json.dumps(
                {
                    "domain": domain,
                    "shot": shot,
                    "accuracy": result[str(shot)]["accuracy"]["mean"],
                    "macro_accuracy": result[str(shot)][
                        "macro_accuracy"
                    ]["mean"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {
        "eligible_pattern_count": len(eligible),
        "paired_candidate_sets_across_shots": True,
        "nested_support_pool": True,
        "fixed_query_pool_across_shots": True,
        "shots": result,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    seen_train, seen_test = load_seen_embeddings(args, manifest)
    novel = load_novel_embeddings(args, manifest)
    seen_result = run_seen_full_prototypes(
        manifest["train_records"],
        seen_train,
        manifest["test_records"],
        seen_test,
        args,
    )
    novel_result = run_domain(
        "novel",
        manifest["novel_test_records"],
        novel,
        manifest["novel_test_records"],
        novel,
        True,
        args,
    )
    report = {
        "format": "pattern_6541_candidate12_known_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "protocol": {
            "candidate_count": args.candidate_count,
            "known_membership_prior": True,
            "forced_output": "one of 12 candidate patterns",
            "unassigned_enabled": False,
            "unknown_condition_evaluated": False,
            "seen_prototype_scope": "all seen-train source groups",
            "novel_shots": list(SHOTS),
            "trials": args.trials,
            "same_name_sibling_patterns_compete_normally": True,
            "paired_candidate_sets_across_shots": True,
            "nested_support_pool": True,
            "fixed_query_pool_across_shots": True,
        },
        "seen_test": seen_result,
        "novel_test": novel_result,
    }
    output = args.output_root / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
