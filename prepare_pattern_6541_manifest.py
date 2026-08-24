from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
PATTERN_SUFFIX = re.compile(r"_P[0-9]+$", re.IGNORECASE)
SOURCE_SUFFIX = re.compile(r"_p[0-9]+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, deduplicated pattern_6541 manifest with "
            "source-grouped seen splits and fully held-out low-shot patterns."
        )
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("pattern_6541")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--duplicate-audit",
        type=Path,
        default=Path("reports/pattern_6541_duplicate_audit_v1.json"),
    )
    parser.add_argument("--min-positive-instances", type=int, default=50)
    parser.add_argument("--min-positive-sources", type=int, default=15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--novel-dev-patterns", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def name_from_pattern(pattern: str) -> str:
    return PATTERN_SUFFIX.sub("", pattern)


def source_from_path(path: Path) -> str:
    return SOURCE_SUFFIX.sub("", path.stem)


def deterministic_rng(seed: int, value: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def discover_records(dataset_root: Path) -> tuple[list[dict], dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    raw_pattern_counts: Counter[str] = Counter()
    for pattern_dir in sorted(
        (path for path in dataset_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        pattern = pattern_dir.name
        name = name_from_pattern(pattern)
        paths = sorted(
            path
            for path in pattern_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        raw_pattern_counts[pattern] = len(paths)
        for path in paths:
            relative = path.relative_to(dataset_root).as_posix()
            file_hash = sha256(path)
            by_hash[file_hash].append(
                {
                    "path": relative,
                    "pattern": pattern,
                    "name": name,
                    "source": source_from_path(path),
                    "sha256": file_hash,
                }
            )

    records: list[dict] = []
    duplicate_groups = []
    conflicting_duplicate_groups = []
    for file_hash, items in sorted(by_hash.items()):
        patterns = {item["pattern"] for item in items}
        if len(items) > 1:
            duplicate_groups.append([item["path"] for item in items])
        if len(patterns) > 1:
            conflicting_duplicate_groups.append(
                {
                    "sha256": file_hash,
                    "paths": [item["path"] for item in items],
                    "patterns": sorted(patterns),
                }
            )
            # Conflicting labels cannot safely participate in either training
            # or evaluation, so the whole exact-duplicate group is excluded.
            continue
        records.append(sorted(items, key=lambda item: item["path"])[0])

    audit = {
        "raw_image_count": sum(raw_pattern_counts.values()),
        "raw_pattern_count": len(raw_pattern_counts),
        "raw_name_count": len(
            {name_from_pattern(pattern) for pattern in raw_pattern_counts}
        ),
        "raw_pattern_counts": dict(sorted(raw_pattern_counts.items())),
        "unique_hash_count": len(by_hash),
        "same_label_duplicate_group_count": len(duplicate_groups)
        - len(conflicting_duplicate_groups),
        "conflicting_duplicate_group_count": len(
            conflicting_duplicate_groups
        ),
        "conflicting_duplicate_groups": conflicting_duplicate_groups,
        "usable_image_count": len(records),
    }
    return records, audit


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, item: tuple[str, str]) -> tuple[str, str]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        smaller, larger = sorted((first_root, second_root))
        self.parent[larger] = smaller


def merge_near_duplicate_sources(
    records: list[dict], audit_path: Path | None
) -> dict:
    if audit_path is None:
        return {
            "audit_path": None,
            "same_pattern_pairs_merged": 0,
            "cross_pattern_pairs_ignored": 0,
        }
    audit_path = audit_path.resolve(strict=True)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = payload["phash"]["high_confidence_near_duplicates"]
    by_path = {record["path"]: record for record in records}
    disjoint = DisjointSet()
    merged = 0
    ignored = 0
    for candidate in candidates:
        first = by_path.get(candidate["first"])
        second = by_path.get(candidate["second"])
        if first is None or second is None:
            continue
        if first["pattern"] != second["pattern"]:
            ignored += 1
            continue
        disjoint.union(
            (first["pattern"], first["source"]),
            (second["pattern"], second["source"]),
        )
        merged += 1
    for record in records:
        original_source = record["source"]
        root = disjoint.find((record["pattern"], original_source))
        record["source_original"] = original_source
        record["source"] = root[1]
    return {
        "audit_path": str(audit_path),
        "same_pattern_pairs_merged": merged,
        "cross_pattern_pairs_ignored": ignored,
        "merge_policy": (
            "merge source groups only for high-confidence pairs within the "
            "same pattern; never auto-merge cross-pattern pHash candidates"
        ),
    }


def select_split_groups(
    pattern: str,
    records: list[dict],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    groups = list(by_source)
    rng = deterministic_rng(seed, pattern)
    rng.shuffle(groups)
    test_target = max(1, round(len(records) * test_fraction))
    validation_target = max(1, round(len(records) * validation_fraction))
    assignments: dict[str, str] = {}
    test_images = 0
    validation_images = 0
    for source in groups:
        remaining_groups = len(groups) - len(assignments)
        if test_images < test_target and remaining_groups > 2:
            assignments[source] = "test"
            test_images += len(by_source[source])
        elif validation_images < validation_target and remaining_groups > 1:
            assignments[source] = "validation"
            validation_images += len(by_source[source])
        else:
            assignments[source] = "train"
    if set(assignments.values()) != {"train", "validation", "test"}:
        raise RuntimeError(f"Could not create three-way split for {pattern}.")
    return assignments


def split_novel_patterns(
    patterns: list[str],
    by_pattern: dict[str, list[dict]],
    target_dev_patterns: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for pattern in patterns:
        by_name[by_pattern[pattern][0]["name"]].append(pattern)
    names = sorted(by_name)
    rng = deterministic_rng(seed, "novel-dev-pattern-split")
    rng.shuffle(names)
    dev: set[str] = set()
    for name in names:
        if len(dev) >= target_dev_patterns:
            break
        dev.update(by_name[name])
    return dev, set(patterns) - dev


def build_manifest(args: argparse.Namespace) -> dict:
    dataset_root = args.dataset_root.resolve(strict=True)
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if not 0.0 < args.test_fraction < 0.5:
        raise ValueError("--test-fraction must be in (0, 0.5).")
    if args.validation_fraction + args.test_fraction >= 0.5:
        raise ValueError("Validation and test fractions must sum to less than 0.5.")
    if args.min_positive_instances < 2:
        raise ValueError("--min-positive-instances must be at least 2.")

    records, audit = discover_records(dataset_root)
    near_duplicate_audit = merge_near_duplicate_sources(
        records, args.duplicate_audit
    )
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_pattern[record["pattern"]].append(record)

    seen_patterns = sorted(
        pattern
        for pattern, items in by_pattern.items()
        if len(items) >= args.min_positive_instances
        and len({item["source"] for item in items})
        >= args.min_positive_sources
    )
    held_out_patterns = sorted(set(by_pattern) - set(seen_patterns))
    novel_dev_patterns, novel_test_patterns = split_novel_patterns(
        held_out_patterns,
        by_pattern,
        args.novel_dev_patterns,
        args.seed,
    )
    train_records: list[dict] = []
    validation_records: list[dict] = []
    test_records: list[dict] = []
    novel_dev_records: list[dict] = []
    novel_test_records: list[dict] = []
    per_pattern = {}

    for pattern in sorted(by_pattern):
        items = sorted(by_pattern[pattern], key=lambda item: item["path"])
        sources = {item["source"] for item in items}
        if pattern in seen_patterns:
            split_assignments = select_split_groups(
                pattern,
                items,
                args.validation_fraction,
                args.test_fraction,
                args.seed,
            )
            pattern_train = [
                item
                for item in items
                if split_assignments[item["source"]] == "train"
            ]
            pattern_validation = [
                item
                for item in items
                if split_assignments[item["source"]] == "validation"
            ]
            pattern_test = [
                item
                for item in items
                if split_assignments[item["source"]] == "test"
            ]
            if not pattern_train or not pattern_validation or not pattern_test:
                raise RuntimeError(f"Invalid split for {pattern}.")
            train_records.extend(pattern_train)
            validation_records.extend(pattern_validation)
            test_records.extend(pattern_test)
            split_kind = "seen_positive"
        elif pattern in novel_dev_patterns:
            pattern_train = []
            pattern_validation = []
            pattern_test = []
            novel_dev_records.extend(items)
            split_kind = "novel_dev"
        else:
            pattern_train = []
            pattern_validation = []
            pattern_test = []
            novel_test_records.extend(items)
            split_kind = "novel_test_locked"
        per_pattern[pattern] = {
            "name": items[0]["name"],
            "usable_instances": len(items),
            "source_groups": len(sources),
            "split": split_kind,
            "train_instances": len(pattern_train),
            "validation_instances": len(pattern_validation),
            "test_instances": len(pattern_test),
        }

    train_names = {record["name"] for record in train_records}
    seen_name_siblings = {
        name: sorted(
            pattern
            for pattern in seen_patterns
            if name_from_pattern(pattern) == name
        )
        for name in sorted(train_names)
    }
    seen_name_siblings = {
        name: patterns
        for name, patterns in seen_name_siblings.items()
        if len(patterns) > 1
    }

    manifest = {
        "format": "pattern_6541_metric_manifest_v1",
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "min_positive_instances": args.min_positive_instances,
        "min_positive_sources": args.min_positive_sources,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "novel_dev_target_patterns": args.novel_dev_patterns,
        "split_unit": "source filename group after exact-byte deduplication",
        "name_rule": "strip terminal _P<digits> from pattern directory",
        "negative_rule": (
            "patterns sharing the same name are never sampled in one "
            "training episode and therefore never act as negatives"
        ),
        "audit": audit,
        "near_duplicate_source_merge": near_duplicate_audit,
        "summary": {
            "seen_pattern_count": len(seen_patterns),
            "held_out_pattern_count": len(held_out_patterns),
            "novel_dev_pattern_count": len(novel_dev_patterns),
            "novel_test_pattern_count": len(novel_test_patterns),
            "seen_name_count": len(train_names),
            "seen_multi_pattern_name_count": len(seen_name_siblings),
            "train_image_count": len(train_records),
            "validation_image_count": len(validation_records),
            "test_image_count": len(test_records),
            "novel_dev_image_count": len(novel_dev_records),
            "novel_test_image_count": len(novel_test_records),
            "novel_pattern_seen_name_count": sum(
                by_pattern[pattern][0]["name"] in train_names
                for pattern in held_out_patterns
            ),
        },
        "seen_patterns": seen_patterns,
        "held_out_patterns": held_out_patterns,
        "novel_dev_patterns": sorted(novel_dev_patterns),
        "novel_test_patterns": sorted(novel_test_patterns),
        "seen_same_name_siblings": seen_name_siblings,
        "per_pattern": per_pattern,
        "train_records": sorted(
            train_records, key=lambda item: item["path"]
        ),
        "validation_records": sorted(
            validation_records, key=lambda item: item["path"]
        ),
        "test_records": sorted(test_records, key=lambda item: item["path"]),
        "novel_dev_records": sorted(
            novel_dev_records, key=lambda item: item["path"]
        ),
        "novel_test_records": sorted(
            novel_test_records, key=lambda item: item["path"]
        ),
    }
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["manifest_fingerprint_sha256"] = hashlib.sha256(
        canonical
    ).hexdigest()
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "fingerprint": manifest["manifest_fingerprint_sha256"],
                **manifest["summary"],
                "audit": {
                    key: value
                    for key, value in manifest["audit"].items()
                    if key
                    not in {
                        "raw_pattern_counts",
                        "conflicting_duplicate_groups",
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
