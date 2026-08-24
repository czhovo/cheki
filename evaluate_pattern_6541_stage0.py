from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, Dataset

from pattern_encoder_v3 import (
    FrozenFeatureBackbone,
    fixed_fusion,
    image_to_views,
    load_rgb,
    seed_everything,
)


class ViewDataset(Dataset):
    def __init__(
        self, root: Path, records: list[dict], image_size: int
    ) -> None:
        self.root = root
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = load_rgb(self.root / self.records[index]["path"])
        try:
            return (
                image_to_views(
                    image, training=False, image_size=self.image_size
                ),
                index,
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
        choices=(
            "resnet18_imagenet",
            "dinov2_vits14",
            "convnext_tiny_in22k",
        ),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_stage0_v1"),
    )
    parser.add_argument("--recompute", action="store_true")
    return parser.parse_args()


def cache_key(manifest: dict, backbone: str, input_size: int) -> str:
    value = (
        f"{manifest['manifest_fingerprint_sha256']}\0{backbone}\0"
        f"stage0_eval_views_{input_size}_bottom45_v2"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@torch.inference_mode()
def extract_features(
    model: FrozenFeatureBackbone,
    dataset: ViewDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, dict[str, torch.Tensor]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
    for batch_number, (views, indices) in enumerate(loader, start=1):
        views = views.to(device, non_blocking=True)
        batch, regions, channels, height, width = views.shape
        batch_variants = model.feature_variants(
            views.reshape(batch * regions, channels, height, width)
        )
        for name, flat_features in batch_variants.items():
            features = flat_features.reshape(batch, regions, -1)
            if name not in outputs:
                feature_dim = features.shape[-1]
                outputs[name] = {
                    "full": torch.empty(
                        (len(dataset), feature_dim), dtype=torch.float32
                    ),
                    "bottom": torch.empty(
                        (len(dataset), feature_dim), dtype=torch.float32
                    ),
                }
            outputs[name]["full"][indices] = features[:, 0].cpu()
            outputs[name]["bottom"][indices] = features[:, 1].cpu()
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(
                json.dumps(
                    {
                        "feature_batches": batch_number,
                        "total_batches": len(loader),
                        "encoded": min(batch_number * batch_size, len(dataset)),
                        "total": len(dataset),
                    }
                ),
                flush=True,
            )
    return outputs


def source_weighted_prototypes(
    records: list[dict],
    features: torch.Tensor,
    patterns: list[str],
) -> torch.Tensor:
    by_pattern_source: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        by_pattern_source[record["pattern"]][record["source"]].append(index)
    prototypes = []
    for pattern in patterns:
        source_means = [
            torch_f.normalize(features[indices].mean(dim=0), dim=0)
            for indices in by_pattern_source[pattern].values()
        ]
        prototypes.append(
            torch_f.normalize(torch.stack(source_means).mean(dim=0), dim=0)
        )
    return torch.stack(prototypes)


def macro_f1(truth: np.ndarray, predictions: np.ndarray, count: int) -> float:
    values = []
    for label in range(count):
        true_positive = int(((truth == label) & (predictions == label)).sum())
        false_positive = int(((truth != label) & (predictions == label)).sum())
        false_negative = int(((truth == label) & (predictions != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(values))


def classification_metrics(
    truth: torch.Tensor,
    predictions: torch.Tensor,
    patterns: list[str],
) -> dict:
    truth_np = truth.numpy()
    prediction_np = predictions.numpy()
    correct = truth_np == prediction_np
    per_pattern = {}
    for label, pattern in enumerate(patterns):
        mask = truth_np == label
        per_pattern[pattern] = {
            "count": int(mask.sum()),
            "correct": int(correct[mask].sum()),
            "accuracy": float(correct[mask].mean()),
        }
    confusion = np.zeros((len(patterns), len(patterns)), dtype=np.int64)
    np.add.at(confusion, (truth_np, prediction_np), 1)
    macro_accuracy = float(
        np.mean([item["accuracy"] for item in per_pattern.values()])
    )
    worst = sorted(
        (
            {"pattern": pattern, **values}
            for pattern, values in per_pattern.items()
        ),
        key=lambda item: (item["accuracy"], item["pattern"]),
    )[:10]
    return {
        "count": len(truth_np),
        "top1_accuracy": float(correct.mean()),
        "macro_accuracy": macro_accuracy,
        "macro_f1": macro_f1(truth_np, prediction_np, len(patterns)),
        "per_pattern": per_pattern,
        "confusion_matrix": {
            truth_pattern: {
                prediction_pattern: int(confusion[truth_index, prediction_index])
                for prediction_index, prediction_pattern in enumerate(patterns)
            }
            for truth_index, truth_pattern in enumerate(patterns)
        },
        "worst_10": worst,
    }


def evaluate_seen_validation(
    train_records: list[dict],
    train_features: torch.Tensor,
    validation_records: list[dict],
    validation_features: torch.Tensor,
) -> dict:
    patterns = sorted({record["pattern"] for record in train_records})
    pattern_to_label = {pattern: index for index, pattern in enumerate(patterns)}
    prototypes = source_weighted_prototypes(
        train_records, train_features, patterns
    )
    scores = validation_features @ prototypes.T
    predictions = scores.argmax(dim=1).cpu()
    truth = torch.tensor(
        [pattern_to_label[record["pattern"]] for record in validation_records]
    )
    result = classification_metrics(truth, predictions, patterns)
    result["prototype_source"] = "train_only_source_group_weighted"
    return result


def grouped_indices(records: list[dict]) -> dict[str, dict[str, list[int]]]:
    result: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        result[record["pattern"]][record["source"]].append(index)
    return result


def trial_classification(
    records: list[dict],
    features: torch.Tensor,
    patterns: list[str],
    support_sources: dict[str, list[str]],
    query_sources: dict[str, list[str]],
    seen_names: set[str],
) -> dict:
    groups = grouped_indices(records)
    pattern_to_label = {pattern: index for index, pattern in enumerate(patterns)}
    pattern_to_name = {
        record["pattern"]: record["name"] for record in records
    }
    prototypes = []
    query_indices = []
    truth = []
    for pattern in patterns:
        support_means = [
            torch_f.normalize(features[groups[pattern][source]].mean(0), dim=0)
            for source in support_sources[pattern]
        ]
        prototypes.append(
            torch_f.normalize(torch.stack(support_means).mean(0), dim=0)
        )
        for source in query_sources[pattern]:
            indices = groups[pattern][source]
            query_indices.extend(indices)
            truth.extend([pattern_to_label[pattern]] * len(indices))
    prototypes_tensor = torch.stack(prototypes)
    query_features = features[query_indices]
    scores = query_features @ prototypes_tensor.T
    truth_tensor = torch.tensor(truth)
    primary_predictions = scores.argmax(dim=1)

    secondary_scores = scores.clone()
    for row, truth_label in enumerate(truth):
        truth_name = pattern_to_name[patterns[truth_label]]
        for candidate_label, candidate_pattern in enumerate(patterns):
            if (
                candidate_label != truth_label
                and pattern_to_name[candidate_pattern] == truth_name
            ):
                secondary_scores[row, candidate_label] = float("-inf")
    secondary_predictions = secondary_scores.argmax(dim=1)
    primary_correct = primary_predictions.eq(truth_tensor)
    secondary_correct = secondary_predictions.eq(truth_tensor)

    subgroup = {}
    for key, predicate in (
        ("novel_pattern_seen_name", lambda pattern: pattern_to_name[pattern] in seen_names),
        ("novel_pattern_novel_name", lambda pattern: pattern_to_name[pattern] not in seen_names),
    ):
        mask = torch.tensor([predicate(patterns[label]) for label in truth])
        subgroup[key] = {
            "count": int(mask.sum()),
            "primary_accuracy": (
                float(primary_correct[mask].float().mean())
                if bool(mask.any())
                else None
            ),
            "name_masked_accuracy": (
                float(secondary_correct[mask].float().mean())
                if bool(mask.any())
                else None
            ),
        }
    return {
        "query_count": len(truth),
        "pattern_count": len(patterns),
        "primary_accuracy": float(primary_correct.float().mean()),
        "name_masked_accuracy": float(secondary_correct.float().mean()),
        "subgroups": subgroup,
    }


def summarize_trials(trials: list[dict]) -> dict:
    result = {
        "trials": len(trials),
        "pattern_count": trials[0]["pattern_count"],
        "mean_query_count": float(np.mean([item["query_count"] for item in trials])),
    }
    for metric in ("primary_accuracy", "name_masked_accuracy"):
        values = np.array([item[metric] for item in trials], dtype=np.float64)
        result[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    for subgroup in ("novel_pattern_seen_name", "novel_pattern_novel_name"):
        result.setdefault("subgroups", {})[subgroup] = {}
        for metric in ("primary_accuracy", "name_masked_accuracy"):
            values = [
                item["subgroups"][subgroup][metric]
                for item in trials
                if item["subgroups"][subgroup][metric] is not None
            ]
            result["subgroups"][subgroup][metric] = (
                {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "trials": len(values),
                }
                if values
                else None
            )
    return result


def evaluate_novel_dev(
    records: list[dict],
    features: torch.Tensor,
    seen_names: set[str],
    trial_count: int,
    seed: int,
) -> dict:
    groups = grouped_indices(records)
    result = {"all_eligible": {}, "common_set": {}}
    for shot in (1, 2, 3, 5):
        patterns = sorted(
            pattern
            for pattern, sources in groups.items()
            if len(sources) >= shot + 2
        )
        trials = []
        for trial in range(trial_count):
            rng = random.Random(seed + shot * 100_003 + trial)
            support = {}
            query = {}
            for pattern in patterns:
                sources = sorted(groups[pattern])
                rng.shuffle(sources)
                support[pattern] = sources[:shot]
                query[pattern] = sources[shot:]
            trials.append(
                trial_classification(
                    records, features, patterns, support, query, seen_names
                )
            )
        result["all_eligible"][str(shot)] = summarize_trials(trials)

    common_patterns = sorted(
        pattern for pattern, sources in groups.items() if len(sources) >= 10
    )
    common_trials: dict[int, list[dict]] = {shot: [] for shot in (1, 2, 3, 5)}
    for trial in range(trial_count):
        rng = random.Random(seed + 900_001 + trial)
        support_pool = {}
        query = {}
        for pattern in common_patterns:
            sources = sorted(groups[pattern])
            rng.shuffle(sources)
            support_pool[pattern] = sources[:5]
            query[pattern] = sources[5:]
        for shot in (1, 2, 3, 5):
            support = {
                pattern: support_pool[pattern][:shot]
                for pattern in common_patterns
            }
            common_trials[shot].append(
                trial_classification(
                    records,
                    features,
                    common_patterns,
                    support,
                    query,
                    seen_names,
                )
            )
    for shot, trials in common_trials.items():
        result["common_set"][str(shot)] = summarize_trials(trials)
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    train_records = manifest["train_records"]
    validation_records = manifest["validation_records"]
    novel_records = manifest["novel_dev_records"]
    records = train_records + validation_records + novel_records
    ranges = {
        "train": (0, len(train_records)),
        "validation": (
            len(train_records),
            len(train_records) + len(validation_records),
        ),
        "novel_dev": (
            len(train_records) + len(validation_records), len(records)
        ),
    }
    output_name = (
        args.backbone
        if args.input_size == 224
        else f"{args.backbone}_{args.input_size}"
    )
    output_dir = args.output_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = output_dir / "features.pt"
    expected_key = cache_key(manifest, args.backbone, args.input_size)
    if feature_cache.is_file() and not args.recompute:
        cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
        if cache.get("cache_key") != expected_key:
            raise RuntimeError("Feature cache key mismatch; use --recompute.")
        if "feature_variants" in cache:
            feature_variants = cache["feature_variants"]
        else:
            feature_variants = {
                (
                    "cls"
                    if args.backbone == "dinov2_vits14"
                    else "default"
                ): {
                    "full": cache["full"],
                    "bottom": cache["bottom"],
                }
            }
    else:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = FrozenFeatureBackbone(
            args.backbone,
            pretrained=True,
            image_size=args.input_size,
        ).to(device)
        feature_variants = extract_features(
            model,
            ViewDataset(root, records, args.input_size),
            device,
            args.batch_size,
            args.num_workers,
        )
        torch.save(
            {
                "cache_key": expected_key,
                "backbone": args.backbone,
                "input_size": args.input_size,
                "manifest_fingerprint": manifest[
                    "manifest_fingerprint_sha256"
                ],
                "paths": [record["path"] for record in records],
                "feature_variants": feature_variants,
            },
            feature_cache,
        )
        model.to("cpu")

    variants = {}
    for feature_name, feature_views in feature_variants.items():
        full = feature_views["full"]
        bottom = feature_views["bottom"]
        prefix = "" if len(feature_variants) == 1 else f"{feature_name}/"
        variants.update(
            {
                f"{prefix}full": torch_f.normalize(full, dim=1),
                f"{prefix}bottom": torch_f.normalize(bottom, dim=1),
                f"{prefix}fusion_0.3_0.7": fixed_fusion(full, bottom),
            }
        )
    train_slice = slice(*ranges["train"])
    validation_slice = slice(*ranges["validation"])
    novel_slice = slice(*ranges["novel_dev"])
    seen_names = {record["name"] for record in train_records}
    report = {
        "format": "pattern_6541_stage0_baseline_v1",
        "backbone": args.backbone,
        "input_size": args.input_size,
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "test_sets_evaluated": False,
        "records_encoded": len(records),
        "variants": {},
    }
    for name, features in variants.items():
        print(json.dumps({"evaluating_variant": name}), flush=True)
        report["variants"][name] = {
            "seen_validation": evaluate_seen_validation(
                train_records,
                features[train_slice],
                validation_records,
                features[validation_slice],
            ),
            "novel_dev_fewshot": evaluate_novel_dev(
                novel_records,
                features[novel_slice],
                seen_names,
                args.novel_trials,
                args.seed,
            ),
        }
    output = output_dir / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "output": str(output.resolve()),
        "backbone": args.backbone,
        "input_size": args.input_size,
        "test_sets_evaluated": False,
        "variants": {
            name: {
                "seen_validation_top1": values["seen_validation"][
                    "top1_accuracy"
                ],
                "seen_validation_macro": values["seen_validation"][
                    "macro_accuracy"
                ],
                "novel_dev_common_1shot": values["novel_dev_fewshot"][
                    "common_set"
                ]["1"]["primary_accuracy"]["mean"],
                "novel_dev_common_5shot": values["novel_dev_fewshot"][
                    "common_set"
                ]["5"]["primary_accuracy"]["mean"],
            }
            for name, values in report["variants"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
