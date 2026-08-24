from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evaluations/pattern_6541_final_v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pattern_6541_final_verification.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    selection = json.loads(
        (args.root / "selection.json").read_text(encoding="utf-8")
    )
    test = json.loads(
        (args.root / "test_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint_path = Path(selection["checkpoint"])
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    checks = {
        "selection_precedes_test": (
            selection["selected_before_test"]
            and not selection["test_evaluated_at_selection_time"]
        ),
        "test_is_locked_mode": (
            test["mode"] == "locked-test"
            and test["locked_test_evaluated"]
            and not test["test_used_for_model_selection"]
        ),
        "checkpoint_hash_matches_selection": (
            checkpoint_hash == selection["checkpoint_sha256"]
        ),
        "checkpoint_hash_matches_test": (
            checkpoint_hash == test["checkpoint_sha256"]
        ),
        "manifest_fingerprint_matches_selection": (
            manifest["manifest_fingerprint_sha256"]
            == selection["manifest_fingerprint"]
        ),
        "manifest_fingerprint_matches_test": (
            manifest["manifest_fingerprint_sha256"]
            == test["manifest_fingerprint"]
        ),
        "manifest_fingerprint_matches_checkpoint": (
            manifest["manifest_fingerprint_sha256"]
            == checkpoint["manifest_fingerprint"]
        ),
        "checkpoint_is_accepted_stage3": (
            checkpoint.get("format") == "pattern_6541_stage3_checkpoint_v1"
            and checkpoint.get("accepted_over_stage2") is True
            and checkpoint.get("epoch") == selection["checkpoint_epoch"]
        ),
        "seen_test_count_matches_manifest": (
            test["seen"]["count"] == len(manifest["test_records"])
        ),
        "novel_test_patterns_match_manifest": (
            test["novel_fewshot"]["all_eligible"]["1"]["pattern_count"]
            <= len(manifest["novel_test_patterns"])
            and test["novel_fewshot"]["common_set"]["1"]["pattern_count"]
            == 61
        ),
    }
    metrics = {
        "seen_top1": test["seen"]["top1_accuracy"],
        "seen_macro": test["seen"]["macro_accuracy"],
        "seen_macro_f1": test["seen"]["macro_f1"],
        "novel_common": {
            shot: test["novel_fewshot"]["common_set"][shot][
                "primary_accuracy"
            ]["mean"]
            for shot in ("1", "2", "3", "5")
        },
    }
    all_values = [
        metrics["seen_top1"],
        metrics["seen_macro"],
        metrics["seen_macro_f1"],
        *metrics["novel_common"].values(),
    ]
    checks["metrics_finite_and_bounded"] = all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in all_values
    )
    checks["novel_accuracy_monotonic_with_shots"] = (
        metrics["novel_common"]["1"]
        < metrics["novel_common"]["2"]
        < metrics["novel_common"]["3"]
        < metrics["novel_common"]["5"]
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Final verification failed: {failed}")

    report = {
        "format": "pattern_6541_final_verification_v1",
        "verified": True,
        "checks": checks,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "metrics": metrics,
        "artifacts": {
            "selection": str((args.root / "selection.json").resolve()),
            "test_report": str((args.root / "test_report.json").resolve()),
            "manifest": str(args.manifest.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
