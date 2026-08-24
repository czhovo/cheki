from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pattern_6541_manifest_verification_v1.json"),
    )
    return parser.parse_args()


def record_index(records: list[dict]) -> dict[str, dict]:
    return {record["path"]: record for record in records}


def source_counts(records: list[dict]) -> dict[str, int]:
    by_pattern: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_pattern[record["pattern"]].add(record["source"])
    return {pattern: len(sources) for pattern, sources in by_pattern.items()}


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_fingerprint = manifest.pop("manifest_fingerprint_sha256")
    canonical = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    actual_fingerprint = hashlib.sha256(canonical).hexdigest()
    manifest["manifest_fingerprint_sha256"] = expected_fingerprint
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError("Manifest fingerprint mismatch.")

    split_keys = (
        "train_records",
        "validation_records",
        "test_records",
        "novel_dev_records",
        "novel_test_records",
    )
    indices = {key: record_index(manifest[key]) for key in split_keys}
    all_paths = [path for index in indices.values() for path in index]
    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("A path appears in multiple splits.")
    if len(all_paths) != manifest["audit"]["usable_image_count"]:
        raise RuntimeError("Usable image count does not match split rows.")

    seen = set(manifest["seen_patterns"])
    novel_dev = set(manifest["novel_dev_patterns"])
    novel_test = set(manifest["novel_test_patterns"])
    if seen & novel_dev or seen & novel_test or novel_dev & novel_test:
        raise RuntimeError("Pattern universes overlap.")
    if seen | novel_dev | novel_test != set(manifest["per_pattern"]):
        raise RuntimeError("Pattern universes do not cover every pattern.")

    for key in ("train_records", "validation_records", "test_records"):
        if {item["pattern"] for item in manifest[key]} != seen:
            raise RuntimeError(f"{key} does not contain exactly seen patterns.")
    if {item["pattern"] for item in manifest["novel_dev_records"]} != novel_dev:
        raise RuntimeError("Novel-dev records do not match pattern split.")
    if {item["pattern"] for item in manifest["novel_test_records"]} != novel_test:
        raise RuntimeError("Novel-test records do not match pattern split.")

    split_sources: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for split in ("train", "validation", "test"):
        for record in manifest[f"{split}_records"]:
            split_sources[record["pattern"]][split].add(record["source"])
    for pattern in seen:
        train_sources = split_sources[pattern]["train"]
        validation_sources = split_sources[pattern]["validation"]
        test_sources = split_sources[pattern]["test"]
        if train_sources & validation_sources:
            raise RuntimeError(f"Train/validation source leakage: {pattern}")
        if train_sources & test_sources:
            raise RuntimeError(f"Train/test source leakage: {pattern}")
        if validation_sources & test_sources:
            raise RuntimeError(f"Validation/test source leakage: {pattern}")

    novel_dev_names = {
        record["name"] for record in manifest["novel_dev_records"]
    }
    novel_test_names = {
        record["name"] for record in manifest["novel_test_records"]
    }
    if novel_dev_names & novel_test_names:
        raise RuntimeError("A novel name is split across dev and test.")

    path_to_split = {
        path: key.removesuffix("_records")
        for key, index in indices.items()
        for path in index
    }
    duplicate_audit = json.loads(
        Path(manifest["near_duplicate_source_merge"]["audit_path"])
        .read_text(encoding="utf-8")
    )
    for pair in duplicate_audit["phash"]["high_confidence_near_duplicates"]:
        if pair["first_pattern"] != pair["second_pattern"]:
            continue
        if (
            pair["first"] not in path_to_split
            or pair["second"] not in path_to_split
        ):
            # An exact byte duplicate is intentionally omitted from the
            # manifest, so it has no split assignment to compare.
            continue
        if path_to_split[pair["first"]] != path_to_split[pair["second"]]:
            raise RuntimeError(
                "Same-pattern near duplicate crosses split: "
                f"{pair['first']} / {pair['second']}"
            )

    seen_names = {record["name"] for record in manifest["train_records"]}
    fewshot = {}
    for universe in ("novel_dev", "novel_test"):
        records = manifest[f"{universe}_records"]
        counts = source_counts(records)
        patterns = sorted(counts)
        fewshot[universe] = {
            "pattern_count": len(patterns),
            "image_count": len(records),
            "novel_pattern_seen_name_count": len(
                {
                    record["pattern"]
                    for record in records
                    if record["name"] in seen_names
                }
            ),
            "novel_pattern_novel_name_count": len(
                {
                    record["pattern"]
                    for record in records
                    if record["name"] not in seen_names
                }
            ),
            "all_eligible": {
                str(shot): sum(count >= shot + 2 for count in counts.values())
                for shot in (1, 2, 3, 5)
            },
            "common_set_min_10_sources": sum(
                count >= 10 for count in counts.values()
            ),
            "source_count_histogram": dict(
                sorted(Counter(counts.values()).items())
            ),
        }

    report = {
        "format": "pattern_6541_manifest_verification_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_fingerprint_sha256": expected_fingerprint,
        "verified": True,
        "checks": {
            "fingerprint": True,
            "path_disjointness": True,
            "pattern_universe_disjointness": True,
            "source_group_disjointness": True,
            "novel_name_group_disjointness": True,
            "same_pattern_near_duplicates_share_split": True,
        },
        "summary": manifest["summary"],
        "fewshot_eligibility": fewshot,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
