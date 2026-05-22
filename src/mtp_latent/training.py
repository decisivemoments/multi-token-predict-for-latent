from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mtp_latent.config import CodecObjectiveConfig, ExperimentConfig
from mtp_latent.metrics import cosine_retrieval_metrics, masked_token_accuracy
from mtp_latent.models import LatentTransitionModel, ReasoningCodec
from mtp_latent.utils import ensure_dir, save_json


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


def _build_summary_writer(config: ExperimentConfig, stage: str) -> SummaryWriter:
    output_dir = ensure_dir(config.train.output_dir)
    tensorboard_root = Path(config.train.tensorboard_dir) if config.train.tensorboard_dir else output_dir / "tensorboard"
    tensorboard_dir = ensure_dir(tensorboard_root / stage)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("experiment/name", config.experiment_name)
    writer.add_text("experiment/config", str(config.dump_dict()))
    return writer


def _write_metrics(writer: SummaryWriter, prefix: str, metrics: dict[str, float], step: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )


def compute_codec_loss(
    codec: ReasoningCodec,
    batch: dict[str, torch.Tensor | list[str] | list[list[str]]],
    objective_config: CodecObjectiveConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    prefix_ids = batch["prefix_ids"].to(device)
    prefix_mask = batch["prefix_mask"].to(device)
    horizon_mask = batch["horizon_mask"].to(device)
    latent = codec.encode(prefix_ids, prefix_mask)

    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}

    for horizon, target_tokens in enumerate(batch["target_steps"]):
        target_tokens = target_tokens.to(device)
        if target_tokens.size(1) <= 1:
            continue
        logits = codec.decode(latent, target_tokens)
        targets = target_tokens[:, 1:]
        sample_loss = _cross_entropy_loss(logits, targets, codec.pad_token_id)

        active = horizon_mask[:, horizon].float().mean()
        if active.item() == 0.0:
            continue

        if objective_config.name == "standard" and horizon > 0:
            continue

        weight = objective_config.horizon_weights[min(horizon, len(objective_config.horizon_weights) - 1)]
        losses.append(weight * sample_loss * active)
        metrics[f"h{horizon + 1}_token_acc"] = masked_token_accuracy(
            logits.detach(),
            targets.detach(),
            codec.pad_token_id,
        )

    if not losses:
        raise ValueError("No active losses were produced for the current batch.")
    return sum(losses), metrics


def evaluate_codec(
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochResult:
    codec.eval()
    total_loss = 0.0
    count = 0
    metric_sums: dict[str, float] = {}

    with torch.no_grad():
        for batch in loader:
            loss, metrics = compute_codec_loss(codec, batch, config.codec_objective, device)
            total_loss += loss.item()
            count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

    averaged = {key: value / max(count, 1) for key, value in metric_sums.items()}
    return EpochResult(loss=total_loss / max(count, 1), metrics=averaged)


def train_codec(
    codec: ReasoningCodec,
    loaders,
    config: ExperimentConfig,
) -> Path:
    device = torch.device(config.train.device)
    codec.to(device)
    codec.load_init_checkpoint()
    optimizer = torch.optim.AdamW(
        codec.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "codec_best.pt"
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    best_valid = float("inf")
    writer = _build_summary_writer(config, "codec")
    global_step = 0

    try:
        for epoch in range(config.train.epochs):
            codec.train()
            progress = tqdm(loaders["train"], desc=f"codec epoch {epoch + 1}/{config.train.epochs}")
            running_loss = 0.0
            metric_sums: dict[str, float] = {}

            for step, batch in enumerate(progress, start=1):
                optimizer.zero_grad()
                loss, metrics = compute_codec_loss(codec, batch, config.codec_objective, device)
                loss.backward()
                nn.utils.clip_grad_norm_(codec.parameters(), config.train.grad_clip_norm)
                optimizer.step()

                global_step += 1
                running_loss += loss.item()
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                _write_metrics(writer, "train_step", metrics, global_step)
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value

                if step % config.train.log_every == 0 or step == len(loaders["train"]):
                    progress.set_postfix(loss=running_loss / step, **metrics)

            averaged_train_metrics = {key: value / max(len(loaders["train"]), 1) for key, value in metric_sums.items()}
            train_result = EpochResult(loss=running_loss / max(len(loaders["train"]), 1), metrics=averaged_train_metrics)
            valid_result = evaluate_codec(codec, loaders["valid"], config, device)
            history["train"].append({"loss": train_result.loss, **train_result.metrics})
            history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
            writer.add_scalar("train/epoch_loss", train_result.loss, epoch + 1)
            writer.add_scalar("valid/loss", valid_result.loss, epoch + 1)
            _write_metrics(writer, "train_epoch", train_result.metrics, epoch + 1)
            _write_metrics(writer, "valid", valid_result.metrics, epoch + 1)

            if valid_result.loss < best_valid:
                best_valid = valid_result.loss
                torch.save(
                    {
                        "model_state": codec.state_dict(),
                        "config": config.dump_dict(),
                        "valid_metrics": valid_result.metrics,
                    },
                    best_path,
                )

        save_json(output_dir / "codec_history.json", history)
        return best_path
    finally:
        writer.close()


def build_transition_pairs(codec: ReasoningCodec, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    codec.eval()

    with torch.no_grad():
        for batch in loader:
            prefix_ids = batch["prefix_ids"].to(device)
            prefix_mask = batch["prefix_mask"].to(device)
            target_steps = batch["target_steps"]
            horizon_mask = batch["horizon_mask"].to(device)

            current_latent = codec.encode(prefix_ids, prefix_mask)
            next_tokens = target_steps[0].to(device)
            next_prefix_ids = torch.cat([prefix_ids, next_tokens[:, 1:]], dim=1)
            max_positions = codec.encoder.config.n_positions
            if next_prefix_ids.size(1) > max_positions:
                next_prefix_ids = next_prefix_ids[:, -max_positions:]
            next_prefix_mask = next_prefix_ids != codec.pad_token_id
            next_latent = codec.encode(next_prefix_ids, next_prefix_mask)

            active = horizon_mask[:, 0]
            if active.any():
                inputs.append(current_latent[active].cpu())
                targets.append(next_latent[active].cpu())

    if not inputs:
        raise ValueError("Transition dataset is empty. Check trace formatting and horizon settings.")
    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


def evaluate_transition(
    transition: LatentTransitionModel,
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    device: torch.device,
) -> EpochResult:
    transition.eval()
    with torch.no_grad():
        predictions = transition(source_latent.to(device)).cpu()
        loss = nn.functional.mse_loss(predictions, target_latent).item()
        metrics = cosine_retrieval_metrics(predictions, target_latent)
    return EpochResult(loss=loss, metrics=metrics)


def train_transition(
    codec: ReasoningCodec,
    transition: LatentTransitionModel,
    loaders,
    config: ExperimentConfig,
) -> Path:
    device = torch.device(config.train.device)
    codec.to(device)
    transition.to(device)
    transition.load_init_checkpoint()

    source_train, target_train = build_transition_pairs(codec, loaders["train"], device)
    source_valid, target_valid = build_transition_pairs(codec, loaders["valid"], device)
    optimizer = torch.optim.AdamW(
        transition.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "transition_best.pt"
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    best_valid = float("inf")
    writer = _build_summary_writer(config, "transition")

    try:
        for epoch in range(config.train.epochs):
            transition.train()
            permutation = torch.randperm(source_train.size(0))
            source_epoch = source_train[permutation].to(device)
            target_epoch = target_train[permutation].to(device)

            optimizer.zero_grad()
            predictions = transition(source_epoch)
            loss = nn.functional.mse_loss(predictions, target_epoch)
            loss.backward()
            nn.utils.clip_grad_norm_(transition.parameters(), config.train.grad_clip_norm)
            optimizer.step()

            train_metrics = cosine_retrieval_metrics(predictions.detach().cpu(), target_epoch.detach().cpu())
            valid_result = evaluate_transition(transition, source_valid, target_valid, device)
            history["train"].append({"loss": loss.item(), **train_metrics})
            history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
            writer.add_scalar("train/loss", loss.item(), epoch + 1)
            writer.add_scalar("valid/loss", valid_result.loss, epoch + 1)
            _write_metrics(writer, "train", train_metrics, epoch + 1)
            _write_metrics(writer, "valid", valid_result.metrics, epoch + 1)

            if valid_result.loss < best_valid:
                best_valid = valid_result.loss
                torch.save(
                    {
                        "transition_state": transition.state_dict(),
                        "config": config.dump_dict(),
                        "valid_metrics": valid_result.metrics,
                    },
                    best_path,
                )

        save_json(output_dir / "transition_history.json", history)
        return best_path
    finally:
        writer.close()
