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
from pattern_encoder_v3 import (
    PatternEncoderV3,
    image_to_views,
    load_rgb,
    seed_everything,
)


class NameAwareEpisodeDataset(Dataset):
    def __init__(
        self,
        root: Path,
        records: list[dict],
        episodes: int,
        n_way: int,
        support_groups: int,
        query_groups: int,
        seed: int,
    ) -> None:
        self.root = root
        self.records = records
        self.episodes = episodes
        self.n_way = n_way
        self.support_groups = support_groups
        self.query_groups = query_groups
        self.seed = seed
        self.epoch = 0
        nested: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for record in records:
            nested[record["name"]][record["pattern"]][
                record["source"]
            ].append(record)
        self.by_name_pattern_source = {
            name: {
                pattern: dict(sources)
                for pattern, sources in patterns.items()
            }
            for name, patterns in nested.items()
        }
        self.names = sorted(self.by_name_pattern_source)
        if n_way > len(self.names):
            raise ValueError("n-way exceeds available names.")
        minimum = support_groups + query_groups
        too_small = {}
        for name, patterns in self.by_name_pattern_source.items():
            for pattern, sources in patterns.items():
                if len(sources) < minimum:
                    too_small[pattern] = len(sources)
        if too_small:
            raise ValueError(
                f"Patterns need at least {minimum} source groups: {too_small}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.episodes

    def load_record(self, record: dict) -> torch.Tensor:
        image = load_rgb(self.root / record["path"])
        try:
            return image_to_views(image, training=True)
        finally:
            image.close()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + index * 97_409
        )
        selected_names = rng.sample(self.names, self.n_way)
        support_views = []
        query_views = []
        support_labels = []
        query_labels = []
        support_pattern_ids = []
        query_pattern_ids = []
        support_name_ids = []
        query_name_ids = []
        for local_label, name in enumerate(selected_names):
            patterns = sorted(self.by_name_pattern_source[name])
            pattern = rng.choice(patterns)
            sources = sorted(self.by_name_pattern_source[name][pattern])
            selected_sources = rng.sample(
                sources, self.support_groups + self.query_groups
            )
            pattern_id = sorted(
                {
                    record["pattern"]
                    for record in self.records
                }
            ).index(pattern)
            name_id = self.names.index(name)
            for source_index, source in enumerate(selected_sources):
                record = rng.choice(
                    self.by_name_pattern_source[name][pattern][source]
                )
                if source_index < self.support_groups:
                    support_views.append(self.load_record(record))
                    support_labels.append(local_label)
                    support_pattern_ids.append(pattern_id)
                    support_name_ids.append(name_id)
                else:
                    query_views.append(self.load_record(record))
                    query_labels.append(local_label)
                    query_pattern_ids.append(pattern_id)
                    query_name_ids.append(name_id)
        return {
            "support_views": torch.stack(support_views),
            "query_views": torch.stack(query_views),
            "support_labels": torch.tensor(support_labels, dtype=torch.long),
            "query_labels": torch.tensor(query_labels, dtype=torch.long),
            "support_pattern_ids": torch.tensor(
                support_pattern_ids, dtype=torch.long
            ),
            "query_pattern_ids": torch.tensor(
                query_pattern_ids, dtype=torch.long
            ),
            "support_name_ids": torch.tensor(support_name_ids, dtype=torch.long),
            "query_name_ids": torch.tensor(query_name_ids, dtype=torch.long),
        }


class EmbeddingDataset(Dataset):
    def __init__(self, root: Path, records: list[dict]) -> None:
        self.root = root
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = load_rgb(self.root / self.records[index]["path"])
        try:
            return image_to_views(image, training=False), index
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
        choices=("dinov2_vits14", "convnext_tiny_in22k"),
        required=True,
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
    parser.add_argument("--validation-batch-size", type=int, default=24)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--episode-workers", type=int, default=2)
    parser.add_argument("--novel-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluations/pattern_6541_stage1_v1"),
    )
    return parser.parse_args()


def masked_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    pattern_ids: torch.Tensor,
    name_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    embeddings = torch_f.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    count = len(embeddings)
    self_mask = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positive = pattern_ids[:, None].eq(pattern_ids[None, :]) & ~self_mask
    same_name_different_pattern = (
        name_ids[:, None].eq(name_ids[None, :])
        & ~pattern_ids[:, None].eq(pattern_ids[None, :])
    )
    valid = ~self_mask & ~same_name_different_pattern
    if torch.any(positive.sum(dim=1) == 0):
        raise RuntimeError("Every SupCon anchor requires a positive.")
    masked_logits = logits.masked_fill(~valid, float("-inf"))
    log_prob = logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    return -(
        (log_prob.masked_fill(~positive, 0.0)).sum(dim=1)
        / positive.sum(dim=1)
    ).mean()


def episode_loss(
    model: PatternEncoderV3,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    support_views = batch["support_views"].to(device, non_blocking=True)
    query_views = batch["query_views"].to(device, non_blocking=True)
    support_labels = batch["support_labels"].to(device, non_blocking=True)
    query_labels = batch["query_labels"].to(device, non_blocking=True)
    pattern_ids = torch.cat(
        [batch["support_pattern_ids"], batch["query_pattern_ids"]]
    ).to(device, non_blocking=True)
    name_ids = torch.cat(
        [batch["support_name_ids"], batch["query_name_ids"]]
    ).to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=(
            torch.bfloat16
            if getattr(args, "amp_dtype", "fp16") == "bf16"
            else torch.float16
        ),
        enabled=device.type == "cuda",
    ):
        embeddings, gate_weights = model(
            torch.cat([support_views, query_views]), return_gate=True
        )
        support_count = len(support_views)
        support_embeddings = embeddings[:support_count]
        query_embeddings = embeddings[support_count:]
        prototypes = []
        for label in range(args.n_way):
            prototypes.append(
                torch_f.normalize(
                    support_embeddings[support_labels == label].mean(dim=0),
                    dim=0,
                )
            )
        prototypes = torch.stack(prototypes)
        prototype_logits = (
            query_embeddings @ prototypes.T / args.prototype_temperature
        )
        prototype_loss = torch_f.cross_entropy(prototype_logits, query_labels)
        supcon_loss = masked_supervised_contrastive_loss(
            embeddings, pattern_ids, name_ids, args.supcon_temperature
        )
        loss = prototype_loss + args.supcon_weight * supcon_loss
        query_predictions = prototype_logits.argmax(dim=1)
        sorted_logits = prototype_logits.sort(dim=1, descending=True).values
        query_margin = (
            sorted_logits[:, 0] - sorted_logits[:, 1]
        ) * args.prototype_temperature
        correct_cosine = (
            query_embeddings
            * prototypes[query_labels]
        ).sum(dim=1)
        similarities = embeddings @ embeddings.T
        self_mask = torch.eye(
            len(embeddings), dtype=torch.bool, device=embeddings.device
        )
        positive_mask = pattern_ids[:, None].eq(pattern_ids[None, :]) & ~self_mask
        negative_mask = ~name_ids[:, None].eq(name_ids[None, :])
    return loss, {
        "prototype_loss": float(prototype_loss.detach()),
        "supcon_loss": float(supcon_loss.detach()),
        "loss": float(loss.detach()),
        "query_accuracy": float(query_predictions.eq(query_labels).float().mean()),
        "query_margin": float(query_margin.mean().detach()),
        "correct_class_cosine": float(correct_cosine.mean().detach()),
        "positive_cosine": float(similarities[positive_mask].mean().detach()),
        "negative_cosine": float(similarities[negative_mask].mean().detach()),
        "gate_full_weight": float(gate_weights[:, 0].mean().detach()),
        "gate_bottom_weight": float(gate_weights[:, 1].mean().detach()),
    }


@torch.inference_mode()
def embed_records(
    model: PatternEncoderV3,
    root: Path,
    records: list[dict],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        EmbeddingDataset(root, records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    output = torch.empty((len(records), 256), dtype=torch.float32)
    model.eval()
    for views, indices in loader:
        output[indices] = model(
            views.to(device, non_blocking=True)
        ).cpu()
    return output


def validate(
    model: PatternEncoderV3,
    root: Path,
    manifest: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    train_records = manifest["train_records"]
    validation_records = manifest["validation_records"]
    novel_records = manifest["novel_dev_records"]
    all_records = train_records + validation_records + novel_records
    embeddings = embed_records(
        model,
        root,
        all_records,
        device,
        args.validation_batch_size,
        args.validation_workers,
    )
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
    novel_one = novel["common_set"]["1"]["primary_accuracy"]["mean"]
    novel_five = novel["common_set"]["5"]["primary_accuracy"]["mean"]
    score = (
        0.50 * seen["macro_accuracy"]
        + 0.25 * novel_one
        + 0.25 * novel_five
    )
    return {
        "selection_score": score,
        "seen_validation": seen,
        "novel_dev_fewshot": novel,
        "test_sets_evaluated": False,
    }


def checkpoint_payload(
    model: PatternEncoderV3,
    manifest: dict,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
) -> dict:
    return {
        "format": "pattern_6541_stage1_checkpoint_v1",
        "backbone": args.backbone,
        "backbone_feature": (
            "last4_normalized_patch_mean"
            if args.backbone == "dinov2_vits14"
            else "global_average_pool"
        ),
        "model_state_dict": model.state_dict(),
        "manifest_fingerprint": manifest["manifest_fingerprint_sha256"],
        "epoch": epoch,
        "metrics": metrics,
        "training_args": vars(args),
        "test_sets_evaluated": False,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    train_records = manifest["train_records"]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = PatternEncoderV3(args.backbone, pretrained=True).to(device)
    model.set_trainable_stage(1)
    dataset = NameAwareEpisodeDataset(
        root,
        train_records,
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        json.dumps(
            {
                "device": str(device),
                "backbone": args.backbone,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "seen_patterns": len(manifest["seen_patterns"]),
                "seen_names": manifest["summary"]["seen_name_count"],
                "test_sets_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )
    first_batch = next(iter(loader))
    model.train()
    model.backbone.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, parts = episode_loss(model, first_batch, device, args)
    if args.dry_run:
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "optimizer_steps": 0,
                    "support_shape": list(first_batch["support_views"].shape),
                    "query_shape": list(first_batch["query_views"].shape),
                    "cuda_peak_allocated_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else None
                    ),
                    "cuda_peak_reserved_gib": (
                        torch.cuda.max_memory_reserved(device) / 2**30
                        if device.type == "cuda"
                        else None
                    ),
                    **parts,
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_root / args.backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
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
        model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        for episode, batch in enumerate(loader, start=1):
            loss, parts = episode_loss(model, batch, device, args)
            scaler.scale(loss / args.gradient_accumulation).backward()
            if (
                episode % args.gradient_accumulation == 0
                or episode == len(loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key, value in parts.items():
                totals[key] += value
            if episode % 10 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "episode": episode,
                            "episodes": len(loader),
                            "mean_loss": totals["loss"] / episode,
                        }
                    ),
                    flush=True,
                )
        metrics = validate(model, root, manifest, device, args)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{key: value / len(loader) for key, value in totals.items()},
            "selection_score": metrics["selection_score"],
            "seen_macro": metrics["seen_validation"]["macro_accuracy"],
            "novel_common_1shot": metrics["novel_dev_fewshot"]["common_set"][
                "1"
            ]["primary_accuracy"]["mean"],
            "novel_common_5shot": metrics["novel_dev_fewshot"]["common_set"][
                "5"
            ]["primary_accuracy"]["mean"],
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
                checkpoint_payload(model, manifest, args, epoch, metrics),
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
