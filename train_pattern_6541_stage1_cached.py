from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, Dataset

from evaluate_pattern_6541_stage0 import (
    evaluate_novel_dev,
    evaluate_seen_validation,
)
from pattern_encoder_v3 import PatternEncoderV3, seed_everything
from train_pattern_6541_stage1 import masked_supervised_contrastive_loss


class CachedNameAwareEpisodeDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        full: torch.Tensor,
        bottom: torch.Tensor,
        episodes: int,
        n_way: int,
        support_groups: int,
        query_groups: int,
        seed: int,
    ) -> None:
        self.records = records
        self.full = full
        self.bottom = bottom
        self.episodes = episodes
        self.n_way = n_way
        self.support_groups = support_groups
        self.query_groups = query_groups
        self.seed = seed
        self.epoch = 0
        self.patterns = sorted({record["pattern"] for record in records})
        self.names = sorted({record["name"] for record in records})
        self.pattern_to_id = {
            pattern: index for index, pattern in enumerate(self.patterns)
        }
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        nested: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for index, record in enumerate(records):
            nested[record["name"]][record["pattern"]][record["source"]].append(
                index
            )
        self.by_name_pattern_source = {
            name: {
                pattern: dict(sources)
                for pattern, sources in patterns.items()
            }
            for name, patterns in nested.items()
        }
        if n_way > len(self.names):
            raise ValueError("n-way exceeds available names.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.episodes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + index * 97_409
        )
        selected_names = rng.sample(self.names, self.n_way)
        values = {
            key: []
            for key in (
                "support_full",
                "support_bottom",
                "query_full",
                "query_bottom",
                "support_labels",
                "query_labels",
                "support_pattern_ids",
                "query_pattern_ids",
                "support_name_ids",
                "query_name_ids",
            )
        }
        for local_label, name in enumerate(selected_names):
            pattern = rng.choice(sorted(self.by_name_pattern_source[name]))
            sources = rng.sample(
                sorted(self.by_name_pattern_source[name][pattern]),
                self.support_groups + self.query_groups,
            )
            for source_index, source in enumerate(sources):
                record_index = rng.choice(
                    self.by_name_pattern_source[name][pattern][source]
                )
                variant = rng.randrange(self.full.shape[1])
                prefix = (
                    "support" if source_index < self.support_groups else "query"
                )
                values[f"{prefix}_full"].append(
                    self.full[record_index, variant]
                )
                values[f"{prefix}_bottom"].append(
                    self.bottom[record_index, variant]
                )
                values[f"{prefix}_labels"].append(local_label)
                values[f"{prefix}_pattern_ids"].append(
                    self.pattern_to_id[pattern]
                )
                values[f"{prefix}_name_ids"].append(self.name_to_id[name])
        result = {}
        for key, items in values.items():
            result[key] = (
                torch.stack(items)
                if key.endswith(("_full", "_bottom"))
                else torch.tensor(items, dtype=torch.long)
            )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_plans/pattern_6541_v1.json"),
    )
    parser.add_argument(
        "--backbone",
        choices=("dinov2_vits14", "convnext_tiny_in22k"),
        required=True,
    )
    parser.add_argument(
        "--feature-bank-root",
        type=Path,
        default=Path("evaluations/pattern_6541_stage1_feature_banks"),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=8)
    parser.add_argument("--support-groups", type=int, default=2)
    parser.add_argument("--query-groups", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--supcon-temperature", type=float, default=0.10)
    parser.add_argument("--supcon-weight", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--head-batch-size", type=int, default=512)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_stage1_v1"),
    )
    return parser.parse_args()


def episode_loss(
    model: PatternEncoderV3,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    support_full = batch["support_full"].to(device).float()
    support_bottom = batch["support_bottom"].to(device).float()
    query_full = batch["query_full"].to(device).float()
    query_bottom = batch["query_bottom"].to(device).float()
    support_labels = batch["support_labels"].to(device)
    query_labels = batch["query_labels"].to(device)
    pattern_ids = torch.cat(
        [batch["support_pattern_ids"], batch["query_pattern_ids"]]
    ).to(device)
    name_ids = torch.cat(
        [batch["support_name_ids"], batch["query_name_ids"]]
    ).to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        embeddings = model.fuse_features(
            torch.cat([support_full, query_full]),
            torch.cat([support_bottom, query_bottom]),
        )
        support_count = len(support_full)
        support_embeddings = embeddings[:support_count]
        query_embeddings = embeddings[support_count:]
        prototypes = torch.stack(
            [
                torch_f.normalize(
                    support_embeddings[support_labels == label].mean(0), dim=0
                )
                for label in range(args.n_way)
            ]
        )
        prototype_loss = torch_f.cross_entropy(
            query_embeddings @ prototypes.T / args.prototype_temperature,
            query_labels,
        )
        supcon_loss = masked_supervised_contrastive_loss(
            embeddings, pattern_ids, name_ids, args.supcon_temperature
        )
        loss = prototype_loss + args.supcon_weight * supcon_loss
    return loss, {
        "loss": float(loss.detach()),
        "prototype_loss": float(prototype_loss.detach()),
        "supcon_loss": float(supcon_loss.detach()),
    }


@torch.inference_mode()
def apply_head(
    model: PatternEncoderV3,
    full: torch.Tensor,
    bottom: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty((len(full), 256), dtype=torch.float32)
    model.eval()
    for start in range(0, len(full), batch_size):
        stop = min(len(full), start + batch_size)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output[start:stop] = model.fuse_features(
                full[start:stop].to(device).float(),
                bottom[start:stop].to(device).float(),
            ).cpu()
    return output


def validate(
    model: PatternEncoderV3,
    bank: dict,
    manifest: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    embeddings = apply_head(
        model,
        bank["eval_full"],
        bank["eval_bottom"],
        device,
        args.head_batch_size,
    )
    train_records = manifest["train_records"]
    validation_records = manifest["validation_records"]
    novel_records = manifest["novel_dev_records"]
    train_stop = len(train_records)
    validation_stop = train_stop + len(validation_records)
    seen = evaluate_seen_validation(
        train_records,
        embeddings[:train_stop],
        validation_records,
        embeddings[train_stop:validation_stop],
    )
    novel = evaluate_novel_dev(
        novel_records,
        embeddings[validation_stop:],
        {record["name"] for record in train_records},
        args.novel_trials,
        args.seed,
    )
    one = novel["common_set"]["1"]["primary_accuracy"]["mean"]
    five = novel["common_set"]["5"]["primary_accuracy"]["mean"]
    return {
        "selection_score": 0.5 * seen["macro_accuracy"] + 0.25 * one + 0.25 * five,
        "seen_validation": seen,
        "novel_dev_fewshot": novel,
        "test_sets_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank_path = args.feature_bank_root / args.backbone / "feature_bank.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if bank["manifest_fingerprint"] != manifest["manifest_fingerprint_sha256"]:
        raise RuntimeError("Feature bank manifest mismatch.")
    if bank["backbone"] != args.backbone:
        raise RuntimeError("Feature bank backbone mismatch.")
    train_records = manifest["train_records"]
    if bank["train_paths"] != [record["path"] for record in train_records]:
        raise RuntimeError("Feature bank train paths mismatch.")
    device = torch.device(args.device)
    model = PatternEncoderV3(args.backbone, pretrained=True).to(device)
    model.set_trainable_stage(1)
    dataset = CachedNameAwareEpisodeDataset(
        train_records,
        bank["train_full"],
        bank["train_bottom"],
        args.episodes_per_epoch,
        args.n_way,
        args.support_groups,
        args.query_groups,
        args.seed,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    first_batch = next(iter(loader))
    model.train()
    loss, parts = episode_loss(model, first_batch, device, args)
    if args.dry_run:
        loss.backward()
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "backbone": args.backbone,
                    "feature_bank": str(bank_path.resolve()),
                    "trainable_parameters": sum(p.numel() for p in trainable),
                    "optimizer_steps": 0,
                    **parts,
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_root / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            0.20
            if epoch == 0
            else 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * min(epoch - 1, args.epochs - 2)
                    / max(1, args.epochs - 2)
                )
            )
        ),
    )
    best_score = float("-inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        for episode, batch in enumerate(loader, start=1):
            loss, values = episode_loss(model, batch, device, args)
            scaler.scale(loss / args.gradient_accumulation).backward()
            if episode % args.gradient_accumulation == 0 or episode == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key, value in values.items():
                totals[key] += value
        metrics = validate(model, bank, manifest, device, args)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{key: value / len(loader) for key, value in totals.items()},
            "selection_score": metrics["selection_score"],
            "seen_macro": metrics["seen_validation"]["macro_accuracy"],
            "novel_common_1shot": metrics["novel_dev_fewshot"]["common_set"]["1"]["primary_accuracy"]["mean"],
            "novel_common_5shot": metrics["novel_dev_fewshot"]["common_set"]["5"]["primary_accuracy"]["mean"],
        }
        history.append(record)
        (output_dir / f"validation_epoch_{epoch:03d}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            stale = 0
            torch.save(
                {
                    "format": "pattern_6541_stage1_cached_checkpoint_v1",
                    "backbone": args.backbone,
                    "backbone_feature": bank["backbone_feature"],
                    "model_state_dict": model.state_dict(),
                    "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
                    "feature_bank": str(bank_path.resolve()),
                    "epoch": epoch,
                    "metrics": metrics,
                    "training_args": vars(args),
                    "test_sets_evaluated": False,
                },
                output_dir / "best_encoder.pt",
            )
        else:
            stale += 1
        print(json.dumps(record, ensure_ascii=False), flush=True)
        scheduler.step()
        if epoch >= args.min_epochs and stale >= args.patience:
            print("early_stopping", flush=True)
            break


if __name__ == "__main__":
    main()
