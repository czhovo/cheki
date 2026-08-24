from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pattern_encoder_v3 import PatternEncoderV3, seed_everything
from train_pattern_6541_stage1 import (
    NameAwareEpisodeDataset,
    validate,
)
from train_pattern_6541_stage2 import (
    split_decay_parameters,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--stage2-checkpoint",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_stage2_v1/"
            "dinov2_vits14/best_encoder.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-score-delta", type=float, default=0.002)
    parser.add_argument("--accept-score-delta", type=float, default=0.005)
    parser.add_argument("--episodes-per-epoch", type=int, default=80)
    parser.add_argument("--n-way", type=int, default=8)
    parser.add_argument("--support-groups", type=int, default=2)
    parser.add_argument("--query-groups", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--supcon-weight", type=float, default=0.25)
    parser.add_argument("--head-lr", type=float, default=3e-5)
    parser.add_argument("--final-norm-lr", type=float, default=5e-6)
    parser.add_argument("--block11-lr", type=float, default=4e-6)
    parser.add_argument("--block10-lr", type=float, default=3e-6)
    parser.add_argument("--block9-lr", type=float, default=2.25e-6)
    parser.add_argument("--block8-lr", type=float, default=1.7e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-batch-size", type=int, default=12)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--cooldown-before-validation", type=int, default=60)
    parser.add_argument("--episode-workers", type=int, default=2)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype", choices=("fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--dry-run-episodes", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/pattern_6541_stage3_v1/dinov2_vits14"),
    )
    return parser.parse_args()


def optimizer_groups(model: PatternEncoderV3, args: argparse.Namespace) -> list[dict]:
    named_groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "head": [],
        "final_norm": [],
        "block11": [],
        "block10": [],
        "block9": [],
        "block8": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("full_norm.", "bottom_norm.", "gate.", "projector.")):
            key = "head"
        elif name.startswith("backbone.norm."):
            key = "final_norm"
        else:
            key = next(
                (
                    f"block{index}"
                    for index in (11, 10, 9, 8)
                    if name.startswith(f"backbone.blocks.{index}.")
                ),
                None,
            )
            if key is None:
                raise RuntimeError(f"Unexpected Stage3 trainable parameter: {name}")
        named_groups[key].append((name, parameter))
    learning_rates = {
        "head": args.head_lr,
        "final_norm": args.final_norm_lr,
        "block11": args.block11_lr,
        "block10": args.block10_lr,
        "block9": args.block9_lr,
        "block8": args.block8_lr,
    }
    result = []
    for key, named in named_groups.items():
        decay, no_decay = split_decay_parameters(named)
        if decay:
            result.append(
                {
                    "name": f"{key}_decay",
                    "params": decay,
                    "lr": learning_rates[key],
                    "weight_decay": args.weight_decay,
                }
            )
        if no_decay:
            result.append(
                {
                    "name": f"{key}_no_decay",
                    "params": no_decay,
                    "lr": learning_rates[key],
                    "weight_decay": 0.0,
                }
            )
    return result


def compact_metrics(metrics: dict) -> dict:
    return {
        "score": metrics["selection_score"],
        "seen_top1": metrics["seen_validation"]["top1_accuracy"],
        "seen_macro": metrics["seen_validation"]["macro_accuracy"],
        "novel1": metrics["novel_dev_fewshot"]["common_set"]["1"][
            "primary_accuracy"
        ]["mean"],
        "novel5": metrics["novel_dev_fewshot"]["common_set"]["5"][
            "primary_accuracy"
        ]["mean"],
    }


def checkpoint_payload(
    model: PatternEncoderV3,
    manifest: dict,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
    accepted: bool,
) -> dict:
    return {
        "format": "pattern_6541_stage3_checkpoint_v1",
        "backbone": "dinov2_vits14",
        "backbone_feature": "last4_normalized_patch_mean",
        "model_state_dict": model.state_dict(),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "epoch": epoch,
        "metrics": metrics,
        "accepted_over_stage2": accepted,
        "training_args": vars(args),
        "test_sets_evaluated": False,
    }


def prepare_for_validation(device: torch.device, cooldown_seconds: int) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    if cooldown_seconds > 0:
        print(
            json.dumps(
                {"cooldown_before_validation_seconds": cooldown_seconds}
            ),
            flush=True,
        )
        time.sleep(cooldown_seconds)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    if source["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
        raise RuntimeError("Stage2 checkpoint manifest mismatch.")
    device = torch.device(args.device)
    model = PatternEncoderV3("dinov2_vits14", pretrained=False)
    model.load_state_dict(source["model_state_dict"])
    model.set_trainable_stage(3)
    model.to(device)
    dataset = NameAwareEpisodeDataset(
        Path(manifest["dataset_root"]),
        manifest["train_records"],
        args.episodes_per_epoch,
        args.n_way,
        args.support_groups,
        args.query_groups,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.episode_workers,
        prefetch_factor=1 if args.episode_workers > 0 else None,
        persistent_workers=False,
    )
    groups = optimizer_groups(model, args)
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.amp_dtype == "fp16",
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "optimizer_groups": [
                    {
                        "name": group["name"],
                        "lr": group["lr"],
                        "weight_decay": group["weight_decay"],
                        "parameters": sum(p.numel() for p in group["params"]),
                    }
                    for group in groups
                ],
                "test_sets_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run_episodes > 0:
        dataset.episodes = args.dry_run_episodes
        loader = DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.episode_workers,
            prefetch_factor=1 if args.episode_workers > 0 else None,
            persistent_workers=False,
        )
        torch.cuda.reset_peak_memory_stats(device)
        values = train_epoch(
            model, loader, optimizer, scaler, device, args, epoch=0
        )
        torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "optimizer_steps": math.ceil(
                        args.dry_run_episodes / args.gradient_accumulation
                    ),
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                    "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                    / 2**30,
                    **values,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = source["metrics"]
    baseline_values = compact_metrics(baseline)
    prepare_for_validation(device, min(10, args.cooldown_before_validation))
    epoch_zero = validate(
        model,
        Path(manifest["dataset_root"]),
        manifest,
        device,
        args,
    )
    if abs(epoch_zero["selection_score"] - baseline["selection_score"]) > 0.003:
        raise RuntimeError("Stage3 epoch 0 failed to reproduce Stage2.")
    (args.output_dir / "validation_epoch_000.json").write_text(
        json.dumps(epoch_zero, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        checkpoint_payload(model, manifest, args, 0, epoch_zero, False),
        args.output_dir / "stage2_fallback.pt",
    )
    print(
        json.dumps(
            {"epoch": 0, "reproduced": True, **compact_metrics(epoch_zero)}
        ),
        flush=True,
    )

    def lr_factor(epoch: int) -> float:
        if epoch == 0:
            return 0.10
        progress = min(epoch - 1, args.epochs - 2) / max(1, args.epochs - 2)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lr_factor for _ in optimizer.param_groups]
    )
    best_score = baseline_values["score"]
    best_epoch = 0
    best_accepted = False
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        train_values = train_epoch(
            model, loader, optimizer, scaler, device, args, epoch
        )
        prepare_for_validation(device, args.cooldown_before_validation)
        metrics = validate(
            model,
            Path(manifest["dataset_root"]),
            manifest,
            device,
            args,
        )
        values = compact_metrics(metrics)
        hard_novel_ok = (
            values["novel1"] >= baseline_values["novel1"] - 0.01
            and values["novel5"] >= baseline_values["novel5"] - 0.01
        )
        accepted = (
            hard_novel_ok
            and values["score"] >= baseline_values["score"] + args.accept_score_delta
            and values["seen_macro"] >= 0.768
            and values["novel1"] >= 0.365
            and values["novel5"] >= 0.582
        )
        if hard_novel_ok and values["score"] > best_score:
            improvement = values["score"] - best_score
            best_score = values["score"]
            best_epoch = epoch
            best_accepted = accepted
            torch.save(
                checkpoint_payload(
                    model, manifest, args, epoch, metrics, accepted
                ),
                args.output_dir / "best_encoder.pt",
            )
            stale = 0 if improvement >= args.min_score_delta else stale + 1
        else:
            stale += 1
        torch.save(
            checkpoint_payload(model, manifest, args, epoch, metrics, accepted),
            args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
        )
        (args.output_dir / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record = {
            "epoch": epoch,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **train_values,
            **values,
            "hard_novel_ok": hard_novel_ok,
            "accepted_over_stage2": accepted,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_accepted": best_accepted,
        }
        history.append(record)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        scheduler.step()
        if epoch >= args.min_epochs and stale >= args.patience:
            print("stage3_early_stopping", flush=True)
            break


if __name__ == "__main__":
    main()
