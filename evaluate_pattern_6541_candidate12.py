from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pattern_encoder_v3 import PatternEncoderV3
from train_pattern_6541_stage1 import EmbeddingDataset


SHOTS = (1, 2, 3, 5, 10)


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
        default=Path("evaluations/pattern_6541_candidate12_v1"),
    )
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--recompute-embeddings", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def encode_records(
    model: PatternEncoderV3,
    dataset_root: Path,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        EmbeddingDataset(dataset_root, records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    output = torch.empty((len(records), 256), dtype=torch.float32)
    model.eval()
    for batch_index, (views, indices) in enumerate(loader, start=1):
        output[indices] = model(views.to(device, non_blocking=True)).cpu()
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {
                        "embedding_batch": batch_index,
                        "embedding_batches": len(loader),
                        "encoded": min(batch_index * batch_size, len(records)),
                        "total": len(records),
                    }
                ),
                flush=True,
            )
    return output


def build_embedding_cache(args: argparse.Namespace, manifest: dict) -> dict:
    cache_path = args.output_root / "embeddings.pt"
    split_keys = (
        "train_records",
        "validation_records",
        "test_records",
        "novel_dev_records",
        "novel_test_records",
    )
    all_records = [record for key in split_keys for record in manifest[key]]
    expected_paths = [record["path"] for record in all_records]
    checkpoint_hash = sha256(args.checkpoint)
    if cache_path.is_file() and not args.recompute_embeddings:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("Embedding cache checkpoint mismatch.")
        if cache["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
            raise RuntimeError("Embedding cache manifest mismatch.")
        if cache["paths"] != expected_paths:
            raise RuntimeError("Embedding cache path order mismatch.")
        return cache

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
        raise RuntimeError("Checkpoint manifest mismatch.")
    model = PatternEncoderV3(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    embeddings = encode_records(
        model,
        Path(manifest["dataset_root"]),
        all_records,
        device,
        args.batch_size,
        args.workers,
    )
    ranges = {}
    start = 0
    for key in split_keys:
        stop = start + len(manifest[key])
        ranges[key] = [start, stop]
        start = stop
    cache = {
        "format": "pattern_6541_all_embeddings_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "paths": expected_paths,
        "ranges": ranges,
        "embeddings": embeddings.half(),
    }
    torch.save(cache, cache_path)
    model.to("cpu")
    return cache


def split_cache(
    cache: dict, manifest: dict
) -> dict[str, tuple[list[dict], torch.Tensor]]:
    result = {}
    for key, (start, stop) in cache["ranges"].items():
        result[key] = (
            manifest[key],
            cache["embeddings"][start:stop].float(),
        )
    return result


def group_indices(records: list[dict]) -> dict[str, dict[str, list[int]]]:
    groups: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        groups[record["pattern"]][record["source"]].append(index)
    return groups


def source_mean(
    embeddings: torch.Tensor, indices: list[int]
) -> torch.Tensor:
    return torch.nn.functional.normalize(embeddings[indices].mean(0), dim=0)


def prototype_from_sources(
    embeddings: torch.Tensor,
    sources: dict[str, list[int]],
    selected: list[str],
) -> torch.Tensor:
    values = [source_mean(embeddings, sources[source]) for source in selected]
    return torch.nn.functional.normalize(torch.stack(values).mean(0), dim=0)


def trial_arrays(
    support_records: list[dict],
    support_embeddings: torch.Tensor,
    query_records: list[dict],
    query_embeddings: torch.Tensor,
    candidate_count: int,
    shot: int,
    rng: random.Random,
    shared_support_and_query: bool,
) -> dict:
    support_groups = group_indices(support_records)
    query_groups = (
        support_groups if shared_support_and_query else group_indices(query_records)
    )
    if shared_support_and_query:
        eligible = sorted(
            pattern
            for pattern, sources in support_groups.items()
            if len(sources) >= shot + 2
        )
    else:
        eligible = sorted(
            pattern
            for pattern, sources in support_groups.items()
            if len(sources) >= shot and pattern in query_groups
        )
    if len(eligible) < candidate_count:
        raise RuntimeError(
            f"Only {len(eligible)} patterns eligible for {shot}-shot."
        )
    candidates = rng.sample(eligible, candidate_count)
    prototypes = []
    in_indices = []
    in_truth = []
    for label, pattern in enumerate(candidates):
        sources = sorted(support_groups[pattern])
        rng.shuffle(sources)
        selected = sources[:shot]
        prototypes.append(
            prototype_from_sources(
                support_embeddings, support_groups[pattern], selected
            )
        )
        if shared_support_and_query:
            query_sources = sources[shot:]
            for source in query_sources:
                indices = support_groups[pattern][source]
                in_indices.extend(indices)
                in_truth.extend([label] * len(indices))
        else:
            for indices in query_groups[pattern].values():
                in_indices.extend(indices)
                in_truth.extend([label] * len(indices))
    prototypes_tensor = torch.stack(prototypes)
    in_scores_matrix = query_embeddings[in_indices] @ prototypes_tensor.T
    in_scores, in_predictions = in_scores_matrix.max(1)
    in_truth_tensor = torch.tensor(in_truth)
    in_correct = in_predictions.eq(in_truth_tensor)

    candidate_set = set(candidates)
    out_indices = [
        index
        for index, record in enumerate(query_records)
        if record["pattern"] not in candidate_set
    ]
    if not out_indices:
        raise RuntimeError("No out-of-candidate queries available.")
    # Equalize candidate-in and candidate-out query counts so threshold
    # accuracy cannot be dominated by either side.
    if len(out_indices) >= len(in_indices):
        sampled_out = rng.sample(out_indices, len(in_indices))
    else:
        sampled_out = [rng.choice(out_indices) for _ in range(len(in_indices))]
    out_scores = (query_embeddings[sampled_out] @ prototypes_tensor.T).max(1).values
    return {
        "candidates": candidates,
        "in_scores": in_scores.numpy(),
        "in_correct": in_correct.numpy(),
        "out_scores": out_scores.numpy(),
    }


def choose_threshold(trials: list[dict]) -> tuple[float, dict]:
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
    balanced = 0.5 * (in_correct_rate + out_reject_rate)
    best_value = balanced.max()
    # Prefer the largest equally optimal threshold, reducing false accepts.
    best_index = np.flatnonzero(np.isclose(balanced, best_value))[-1]
    threshold = float(thresholds[best_index])
    return threshold, {
        "balanced_accuracy": float(balanced[best_index]),
        "in_set_correct_assignment_rate": float(in_correct_rate[best_index]),
        "out_set_unassigned_rate": float(out_reject_rate[best_index]),
        "calibration_in_queries": len(in_scores),
        "calibration_out_queries": len(out_scores),
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_trials(trials: list[dict], threshold: float) -> dict:
    metrics: dict[str, list[float]] = defaultdict(list)
    for trial in trials:
        in_scores = trial["in_scores"]
        in_correct = trial["in_correct"]
        out_scores = trial["out_scores"]
        known_accuracy = float(in_correct.mean())
        in_assigned = in_scores >= threshold
        in_correct_assigned = float((in_correct & in_assigned).mean())
        in_unassigned = float((~in_assigned).mean())
        out_unassigned = float((out_scores < threshold).mean())
        balanced = 0.5 * (in_correct_assigned + out_unassigned)
        metrics["known_12way_accuracy"].append(known_accuracy)
        metrics["unknown_balanced_accuracy"].append(balanced)
        metrics["in_set_correct_assignment_rate"].append(in_correct_assigned)
        metrics["in_set_unassigned_rate"].append(in_unassigned)
        metrics["out_set_unassigned_rate"].append(out_unassigned)
        metrics["out_set_false_accept_rate"].append(1.0 - out_unassigned)
    return {key: summarize(values) for key, values in metrics.items()}


def run_domain(
    name: str,
    calibration_support: tuple[list[dict], torch.Tensor],
    calibration_query: tuple[list[dict], torch.Tensor],
    test_support: tuple[list[dict], torch.Tensor],
    test_query: tuple[list[dict], torch.Tensor],
    shared_calibration: bool,
    shared_test: bool,
    args: argparse.Namespace,
) -> dict:
    result = {
        "domain": name,
        "candidate_count": args.candidate_count,
        "trials": args.trials,
        "threshold_selected_only_on": (
            "seen-validation" if name == "seen" else "novel-dev"
        ),
        "shots": {},
    }
    for shot in SHOTS:
        calibration_trials = []
        test_trials = []
        for trial in range(args.trials):
            calibration_rng = random.Random(
                args.seed + (10_000 if name == "seen" else 20_000)
                + shot * 1_000_003 + trial
            )
            test_rng = random.Random(
                args.seed + (30_000 if name == "seen" else 40_000)
                + shot * 1_000_003 + trial
            )
            calibration_trials.append(
                trial_arrays(
                    *calibration_support,
                    *calibration_query,
                    args.candidate_count,
                    shot,
                    calibration_rng,
                    shared_calibration,
                )
            )
            test_trials.append(
                trial_arrays(
                    *test_support,
                    *test_query,
                    args.candidate_count,
                    shot,
                    test_rng,
                    shared_test,
                )
            )
        threshold, calibration_metrics = choose_threshold(calibration_trials)
        test_metrics = evaluate_trials(test_trials, threshold)
        result["shots"][str(shot)] = {
            "threshold": threshold,
            "calibration": calibration_metrics,
            "test": test_metrics,
            "test_candidate_examples": [
                trial["candidates"] for trial in test_trials[:3]
            ],
        }
        print(
            json.dumps(
                {
                    "domain": name,
                    "shot": shot,
                    "threshold": threshold,
                    "known_accuracy": test_metrics[
                        "known_12way_accuracy"
                    ]["mean"],
                    "unknown_accuracy": test_metrics[
                        "unknown_balanced_accuracy"
                    ]["mean"],
                    "in_correct": test_metrics[
                        "in_set_correct_assignment_rate"
                    ]["mean"],
                    "out_unassigned": test_metrics[
                        "out_set_unassigned_rate"
                    ]["mean"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    cache = build_embedding_cache(args, manifest)
    splits = split_cache(cache, manifest)
    started = time.perf_counter()
    seen = run_domain(
        "seen",
        splits["train_records"],
        splits["validation_records"],
        splits["train_records"],
        splits["test_records"],
        False,
        False,
        args,
    )
    novel = run_domain(
        "novel",
        splits["novel_dev_records"],
        splits["novel_dev_records"],
        splits["novel_test_records"],
        splits["novel_test_records"],
        True,
        True,
        args,
    )
    report = {
        "format": "pattern_6541_candidate12_open_set_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": cache["checkpoint_sha256"],
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "protocol": {
            "candidate_count": args.candidate_count,
            "shots": list(SHOTS),
            "trials": args.trials,
            "known_condition": (
                "query is guaranteed in candidate set; force one of 12"
            ),
            "unknown_condition": (
                "query may be outside candidate set; output one of 12 or unassigned"
            ),
            "threshold_objective": (
                "maximize 0.5 * in-set correct assignment rate + "
                "0.5 * out-set unassigned rate on calibration only"
            ),
            "candidate_in_out_query_balance": "equal image counts per trial",
            "test_used_for_threshold_selection": False,
        },
        "seen": seen,
        "novel": novel,
        "elapsed_seconds_excluding_embedding_cache": time.perf_counter()
        - started,
    }
    output = args.output_root / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
