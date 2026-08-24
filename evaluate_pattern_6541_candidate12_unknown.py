from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_pattern_6541_candidate12_known import (
    grouped_indices,
    prototype,
)
from pattern_encoder_v3 import PatternEncoderV3
from train_pattern_6541_stage1 import embed_records


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
        "--known-cache-root",
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_candidate12_unknown_v1"),
    )
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--novel-shot", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--in-set-weight", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--recompute-validation", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_seen_train_test(
    args: argparse.Namespace, manifest: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    path = args.known_cache_root / "seen_train_test_embeddings.pt"
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache["checkpoint_sha256"] != sha256(args.checkpoint):
        raise RuntimeError("Seen cache checkpoint mismatch.")
    expected = [
        record["path"]
        for record in manifest["train_records"] + manifest["test_records"]
    ]
    if cache["paths"] != expected:
        raise RuntimeError("Seen cache path mismatch.")
    embeddings = cache["embeddings"].float()
    split = cache["train_count"]
    return embeddings[:split], embeddings[split:]


def load_seen_validation(
    args: argparse.Namespace, manifest: dict
) -> torch.Tensor:
    cache_path = args.output_root / "seen_validation_embeddings.pt"
    records = manifest["validation_records"]
    paths = [record["path"] for record in records]
    checkpoint_hash = sha256(args.checkpoint)
    if cache_path.is_file() and not args.recompute_validation:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Validation cache checkpoint mismatch.")
        if cache["paths"] != paths:
            raise RuntimeError("Validation cache path mismatch.")
        return cache["embeddings"].float()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
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
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pattern_6541_seen_validation_embeddings_v1",
            "checkpoint_sha256": checkpoint_hash,
            "manifest_fingerprint": manifest[
                "manifest_fingerprint_sha256"
            ],
            "paths": paths,
            "embeddings": embeddings.half(),
        },
        cache_path,
    )
    model.to("cpu")
    return embeddings


def load_novel(
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


def seen_prototypes(
    records: list[dict], features: torch.Tensor
) -> dict[str, torch.Tensor]:
    groups = grouped_indices(records)
    return {
        pattern: prototype(features, sources, sorted(sources))
        for pattern, sources in groups.items()
    }


def balance_sample(
    indices: list[int], count: int, rng: random.Random
) -> list[int]:
    if len(indices) >= count:
        return rng.sample(indices, count)
    return [rng.choice(indices) for _ in range(count)]


def score_in_queries(
    features: torch.Tensor,
    indices: list[int],
    truth: list[int],
    prototypes: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    scores = features[indices] @ prototypes.T
    maxima, predictions = scores.max(1)
    correct = predictions.eq(torch.tensor(truth))
    return maxima.numpy(), correct.numpy()


def seen_trial(
    prototypes_by_pattern: dict[str, torch.Tensor],
    query_records: list[dict],
    query_features: torch.Tensor,
    candidate_count: int,
    rng: random.Random,
) -> dict:
    candidates = rng.sample(sorted(prototypes_by_pattern), candidate_count)
    candidate_set = set(candidates)
    prototypes = torch.stack([prototypes_by_pattern[p] for p in candidates])
    label = {pattern: index for index, pattern in enumerate(candidates)}
    in_indices = [
        index
        for index, record in enumerate(query_records)
        if record["pattern"] in candidate_set
    ]
    truth = [label[query_records[index]["pattern"]] for index in in_indices]
    in_scores, in_correct = score_in_queries(
        query_features, in_indices, truth, prototypes
    )
    out_indices = [
        index
        for index, record in enumerate(query_records)
        if record["pattern"] not in candidate_set
    ]
    sampled_out = balance_sample(out_indices, len(in_indices), rng)
    out_scores = (query_features[sampled_out] @ prototypes.T).max(1).values.numpy()
    return {
        "candidates": candidates,
        "in_scores": in_scores,
        "in_correct": in_correct,
        "out_scores": out_scores,
    }


def novel_candidate_state(
    records: list[dict],
    features: torch.Tensor,
    shot: int,
    candidate_count: int,
    rng: random.Random,
) -> dict:
    groups = grouped_indices(records)
    eligible = sorted(
        pattern for pattern, sources in groups.items() if len(sources) >= shot + 2
    )
    candidates = rng.sample(eligible, candidate_count)
    candidate_set = set(candidates)
    prototypes = []
    in_indices = []
    truth = []
    for label, pattern in enumerate(candidates):
        sources = sorted(groups[pattern])
        rng.shuffle(sources)
        support = sources[:shot]
        query_sources = sources[shot:]
        prototypes.append(prototype(features, groups[pattern], support))
        for source in query_sources:
            indices = groups[pattern][source]
            in_indices.extend(indices)
            truth.extend([label] * len(indices))
    out_indices = [
        index
        for index, record in enumerate(records)
        if record["pattern"] not in candidate_set
    ]
    return {
        "candidates": candidates,
        "candidate_set": candidate_set,
        "prototypes": torch.stack(prototypes),
        "in_indices": in_indices,
        "truth": truth,
        "out_indices": out_indices,
    }


def novel_trial(
    records: list[dict],
    features: torch.Tensor,
    shot: int,
    candidate_count: int,
    rng: random.Random,
) -> dict:
    state = novel_candidate_state(
        records, features, shot, candidate_count, rng
    )
    in_scores, in_correct = score_in_queries(
        features,
        state["in_indices"],
        state["truth"],
        state["prototypes"],
    )
    sampled_out = balance_sample(
        state["out_indices"], len(state["in_indices"]), rng
    )
    out_scores = (
        features[sampled_out] @ state["prototypes"].T
    ).max(1).values.numpy()
    return {
        "candidates": state["candidates"],
        "in_scores": in_scores,
        "in_correct": in_correct,
        "out_scores": out_scores,
    }


def mixed_trial(
    seen_proto: dict[str, torch.Tensor],
    seen_records: list[dict],
    seen_features: torch.Tensor,
    novel_records: list[dict],
    novel_features: torch.Tensor,
    novel_shot: int,
    rng: random.Random,
) -> dict:
    seen_candidates = rng.sample(sorted(seen_proto), 6)
    novel_state = novel_candidate_state(
        novel_records, novel_features, novel_shot, 6, rng
    )
    candidates = seen_candidates + novel_state["candidates"]
    prototypes = torch.stack(
        [seen_proto[p] for p in seen_candidates]
        + [value for value in novel_state["prototypes"]]
    )
    seen_label = {pattern: index for index, pattern in enumerate(seen_candidates)}
    seen_in = [
        index
        for index, record in enumerate(seen_records)
        if record["pattern"] in seen_label
    ]
    seen_truth = [seen_label[seen_records[index]["pattern"]] for index in seen_in]
    novel_truth = [value + 6 for value in novel_state["truth"]]
    in_scores = torch.cat(
        [
            seen_features[seen_in] @ prototypes.T,
            novel_features[novel_state["in_indices"]] @ prototypes.T,
        ]
    )
    truth = torch.tensor(seen_truth + novel_truth)
    maxima, predictions = in_scores.max(1)
    in_correct = predictions.eq(truth)

    seen_set = set(seen_candidates)
    seen_out = [
        index
        for index, record in enumerate(seen_records)
        if record["pattern"] not in seen_set
    ]
    novel_out = novel_state["out_indices"]
    target = len(truth)
    seen_count = target // 2
    novel_count = target - seen_count
    sampled_seen = balance_sample(seen_out, seen_count, rng)
    sampled_novel = balance_sample(novel_out, novel_count, rng)
    out_scores = torch.cat(
        [
            seen_features[sampled_seen] @ prototypes.T,
            novel_features[sampled_novel] @ prototypes.T,
        ]
    ).max(1).values
    return {
        "candidates": candidates,
        "in_scores": maxima.numpy(),
        "in_correct": in_correct.numpy(),
        "out_scores": out_scores.numpy(),
    }


def choose_threshold(
    trials: list[dict], in_weight: float
) -> tuple[float, dict]:
    in_scores = np.concatenate([trial["in_scores"] for trial in trials])
    in_correct = np.concatenate([trial["in_correct"] for trial in trials])
    out_scores = np.concatenate([trial["out_scores"] for trial in trials])
    unique = np.unique(np.concatenate([in_scores, out_scores]))
    thresholds = np.concatenate(
        [
            [unique[0] - 1e-6],
            (unique[:-1] + unique[1:]) / 2,
            [unique[-1] + 1e-6],
        ]
    )
    correct_scores = np.sort(in_scores[in_correct])
    in_correct_rate = (
        len(correct_scores)
        - np.searchsorted(correct_scores, thresholds, side="left")
    ) / len(in_scores)
    sorted_out = np.sort(out_scores)
    out_reject_rate = np.searchsorted(
        sorted_out, thresholds, side="left"
    ) / len(out_scores)
    objective = in_weight * in_correct_rate + (1 - in_weight) * out_reject_rate
    best = objective.max()
    index = np.flatnonzero(np.isclose(objective, best))[-1]
    return float(thresholds[index]), {
        "objective": float(objective[index]),
        "in_set_correct_assignment_rate": float(in_correct_rate[index]),
        "out_set_unassigned_rate": float(out_reject_rate[index]),
        "in_queries": len(in_scores),
        "out_queries": len(out_scores),
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate(trials: list[dict], threshold: float, in_weight: float) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        assigned = trial["in_scores"] >= threshold
        in_correct = float((trial["in_correct"] & assigned).mean())
        in_accept = float(assigned.mean())
        out_reject = float((trial["out_scores"] < threshold).mean())
        balanced = 0.5 * (in_correct + out_reject)
        weighted = in_weight * in_correct + (1 - in_weight) * out_reject
        values["in_set_correct_assignment_rate"].append(in_correct)
        values["in_set_accept_rate"].append(in_accept)
        values["in_set_unassigned_rate"].append(1 - in_accept)
        values["out_set_unassigned_rate"].append(out_reject)
        values["out_set_false_accept_rate"].append(1 - out_reject)
        values["balanced_accuracy_50_50"].append(balanced)
        values["weighted_accuracy_70_30"].append(weighted)
    return {key: summarize(metric) for key, metric in values.items()}


def main() -> None:
    args = parse_args()
    if not 0 < args.in_set_weight < 1:
        raise ValueError("--in-set-weight must be in (0,1).")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seen_train, seen_test = load_seen_train_test(args, manifest)
    seen_validation = load_seen_validation(args, manifest)
    novel = load_novel(args, manifest)
    seen_proto = seen_prototypes(manifest["train_records"], seen_train)

    calibration_trials = []
    seen_trials = []
    novel_trials = []
    mixed_trials = []
    for trial in range(args.trials):
        calibration_trials.append(
            seen_trial(
                seen_proto,
                manifest["validation_records"],
                seen_validation,
                args.candidate_count,
                random.Random(args.seed + 10_000 + trial),
            )
        )
        seen_trials.append(
            seen_trial(
                seen_proto,
                manifest["test_records"],
                seen_test,
                args.candidate_count,
                random.Random(args.seed + 20_000 + trial),
            )
        )
        novel_trials.append(
            novel_trial(
                manifest["novel_test_records"],
                novel,
                args.novel_shot,
                args.candidate_count,
                random.Random(args.seed + 30_000 + trial),
            )
        )
        mixed_trials.append(
            mixed_trial(
                seen_proto,
                manifest["test_records"],
                seen_test,
                manifest["novel_test_records"],
                novel,
                args.novel_shot,
                random.Random(args.seed + 40_000 + trial),
            )
        )
    threshold, calibration = choose_threshold(
        calibration_trials, args.in_set_weight
    )
    domains = {
        "seen": evaluate(seen_trials, threshold, args.in_set_weight),
        "novel": evaluate(novel_trials, threshold, args.in_set_weight),
        "mixed_6_seen_6_novel": evaluate(
            mixed_trials, threshold, args.in_set_weight
        ),
    }
    report = {
        "format": "pattern_6541_candidate12_unknown_single_threshold_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "protocol": {
            "candidate_count": args.candidate_count,
            "single_global_threshold": True,
            "threshold_selected_only_on": "seen-validation",
            "threshold_in_set_weight": args.in_set_weight,
            "threshold_out_set_weight": 1 - args.in_set_weight,
            "seen_prototypes": "all seen-train source groups",
            "novel_shot": args.novel_shot,
            "mixed_candidates": "6 seen + 6 novel",
            "trials": args.trials,
            "candidate_in_out_query_counts_balanced": True,
            "test_used_for_threshold_selection": False,
        },
        "threshold": threshold,
        "seen_validation_calibration": calibration,
        "test": domains,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "output": str(output.resolve()),
        "threshold": threshold,
        "calibration": calibration,
        "test": {
            domain: {
                key: value["mean"]
                for key, value in metrics.items()
            }
            for domain, metrics in domains.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
