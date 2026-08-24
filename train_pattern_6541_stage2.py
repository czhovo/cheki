from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pattern_encoder_v3 import PatternEncoderV3, seed_everything
from train_pattern_6541_stage1 import (
    NameAwareEpisodeDataset,
    episode_loss,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=Path(
            "evaluations/pattern_6541_stage1_v1/"
            "dinov2_vits14/best_encoder.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience-validations", type=int, default=3)
    parser.add_argument("--min-score-delta", type=float, default=0.002)
    parser.add_argument("--episodes-per-epoch", type=int, default=80)
    parser.add_argument("--n-way", type=int, default=8)
    parser.add_argument("--support-groups", type=int, default=2)
    parser.add_argument("--query-groups", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--supcon-weight", type=float, default=0.25)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--final-norm-lr", type=float, default=1e-5)
    parser.add_argument("--last-block-lr", type=float, default=5e-6)
    parser.add_argument("--penultimate-block-lr", type=float, default=3.5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-every", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--episode-workers", type=int, default=2)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run-episodes", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/pattern_6541_stage2_v1/dinov2_vits14"),
    )
    return parser.parse_args()


def split_decay_parameters(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    decay = []
    no_decay = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return decay, no_decay


def optimizer_groups(model: PatternEncoderV3, args: argparse.Namespace) -> list[dict]:
    categories = {
        "head": [],
        "final_norm": [],
        "last_block": [],
        "penultimate_block": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("full_norm.", "bottom_norm.", "gate.", "projector.")):
            categories["head"].append((name, parameter))
        elif name.startswith("backbone.norm."):
            categories["final_norm"].append((name, parameter))
        elif name.startswith("backbone.blocks.11."):
            categories["last_block"].append((name, parameter))
        elif name.startswith("backbone.blocks.10."):
            categories["penultimate_block"].append((name, parameter))
        else:
            raise RuntimeError(f"Unexpected trainable Stage2 parameter: {name}")
    learning_rates = {
        "head": args.head_lr,
        "final_norm": args.final_norm_lr,
        "last_block": args.last_block_lr,
        "penultimate_block": args.penultimate_block_lr,
    }
    groups = []
    for category, named in categories.items():
        decay, no_decay = split_decay_parameters(named)
        if decay:
            groups.append(
                {
                    "name": f"{category}_decay",
                    "params": decay,
                    "lr": learning_rates[category],
                    "weight_decay": args.weight_decay,
                }
            )
        if no_decay:
            groups.append(
                {
                    "name": f"{category}_no_decay",
                    "params": no_decay,
                    "lr": learning_rates[category],
                    "weight_decay": 0.0,
                }
            )
    return groups


def checkpoint_payload(
    model: PatternEncoderV3,
    manifest: dict,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict | None,
) -> dict:
    return {
        "format": "pattern_6541_stage2_checkpoint_v1",
        "backbone": "dinov2_vits14",
        "backbone_feature": "last4_normalized_patch_mean",
        "model_state_dict": model.state_dict(),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "epoch": epoch,
        "metrics": metrics,
        "training_args": vars(args),
        "test_sets_evaluated": False,
    }


def train_epoch(
    model: PatternEncoderV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict:
    model.train()
    totals = Counter()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for episode, batch in enumerate(loader, start=1):
        loss, values = episode_loss(model, batch, device, args)
        scaler.scale(loss / args.gradient_accumulation).backward()
        if episode % args.gradient_accumulation == 0 or episode == len(loader):
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            gradient_is_finite = bool(torch.isfinite(gradient_norm))
            if gradient_is_finite:
                totals["gradient_norm"] += float(gradient_norm)
            else:
                totals["nonfinite_gradient_steps"] += 1
            if scaler.is_enabled():
                # GradScaler skips optimizer.step when unscale_ observed
                # non-finite gradients and then lowers its scale.
                scaler.step(optimizer)
                scaler.update()
            elif gradient_is_finite:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for key, value in values.items():
            totals[key] += value
        if episode % 10 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "episode": episode,
                        "episodes": len(loader),
                        "mean_loss": totals["loss"] / episode,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    return {
        **{key: value / len(loader) for key, value in totals.items()},
        "train_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
        raise RuntimeError("Stage1 checkpoint manifest mismatch.")
    device = torch.device(args.device)
    model = PatternEncoderV3("dinov2_vits14", pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.set_trainable_stage(2)
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
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    # Epoch 0 validates that the Stage1 checkpoint is reproduced through the
    # on-the-fly Stage2 code path before any update.
    epoch_zero = validate(
        model,
        Path(manifest["dataset_root"]),
        manifest,
        device,
        args,
    )
    (args.output_dir / "validation_epoch_000.json").write_text(
        json.dumps(epoch_zero, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stage1_score = float(checkpoint["metrics"]["selection_score"])
    if abs(epoch_zero["selection_score"] - stage1_score) > 0.003:
        raise RuntimeError(
            "Stage2 epoch-0 score does not reproduce Stage1 within tolerance: "
            f"{epoch_zero['selection_score']} vs {stage1_score}"
        )
    print(
        json.dumps(
            {
                "epoch": 0,
                "selection_score": epoch_zero["selection_score"],
                "stage1_score": stage1_score,
                "reproduced": True,
            }
        ),
        flush=True,
    )
    torch.save(
        checkpoint_payload(model, manifest, args, 0, epoch_zero),
        args.output_dir / "best_encoder.pt",
    )
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_factor(epoch: int) -> float:
        if epoch == 0:
            return 0.10
        progress = min(epoch - 1, args.epochs - 2) / max(1, args.epochs - 2)
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[lr_factor for _ in optimizer.param_groups]
    )
    best_score = epoch_zero["selection_score"]
    best_epoch = 0
    stale_validations = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        train_values = train_epoch(
            model, loader, optimizer, scaler, device, args, epoch
        )
        metrics = None
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            metrics = validate(
                model,
                Path(manifest["dataset_root"]),
                manifest,
                device,
                args,
            )
            (args.output_dir / f"validation_epoch_{epoch:03d}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            score = metrics["selection_score"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    checkpoint_payload(model, manifest, args, epoch, metrics),
                    args.output_dir / "best_encoder.pt",
                )
            if score >= best_score - 1e-12 and score >= (
                history[-1].get("best_validated_score", stage1_score)
                + args.min_score_delta
                if history
                else stage1_score + args.min_score_delta
            ):
                stale_validations = 0
            else:
                stale_validations += 1
        torch.save(
            checkpoint_payload(model, manifest, args, epoch, metrics),
            args.output_dir / f"checkpoint_epoch_{epoch:03d}.pt",
        )
        record = {
            "epoch": epoch,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **train_values,
            "selection_score": (
                metrics["selection_score"] if metrics is not None else None
            ),
            "best_validated_score": best_score,
            "best_epoch": best_epoch,
        }
        history.append(record)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
        scheduler.step()
        if (
            epoch >= args.min_epochs
            and metrics is not None
            and stale_validations >= args.patience_validations
        ):
            print("early_stopping", flush=True)
            break


if __name__ == "__main__":
    main()
