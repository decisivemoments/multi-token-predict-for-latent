from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mtp_latent.config import CodecObjectiveConfig, ExperimentConfig
from mtp_latent.metrics import cosine_retrieval_metrics, masked_token_accuracy
from mtp_latent.models import ReasoningCodec
from mtp_latent.utils import DistributedContext, cleanup_distributed, distributed_mean, distributed_sum_dict, ensure_dir, init_distributed, save_json


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


def _unwrap_codec(model) -> ReasoningCodec:
    return model.module if isinstance(model, DDP) else model


def _build_summary_writer(config: ExperimentConfig, stage: str, dist_ctx: DistributedContext) -> SummaryWriter | None:
    if not dist_ctx.is_main_process:
        return None
    output_dir = ensure_dir(config.train.output_dir)
    tensorboard_root = Path(config.train.tensorboard_dir) if config.train.tensorboard_dir else output_dir / "tensorboard"
    tensorboard_dir = ensure_dir(tensorboard_root / stage)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    writer.add_text("experiment/name", config.experiment_name)
    writer.add_text("experiment/config", str(config.dump_dict()))
    return writer


def _write_metrics(writer: SummaryWriter | None, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
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
    raw_codec = _unwrap_codec(codec)
    prefix_ids = batch["prefix_ids"].to(device)
    prefix_mask = batch["prefix_mask"].to(device)
    horizon_mask = batch["horizon_mask"].to(device)
    latent = raw_codec.encode(prefix_ids, prefix_mask)

    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}

    for horizon, target_tokens in enumerate(batch["target_steps"]):
        target_tokens = target_tokens.to(device)
        if target_tokens.size(1) <= 1:
            continue
        logits = raw_codec.decode(latent, target_tokens)
        targets = target_tokens[:, 1:]
        sample_loss = _cross_entropy_loss(logits, targets, raw_codec.pad_token_id)

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
            raw_codec.pad_token_id,
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

    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        count = int(loss_tensor[1].item())
        metric_sums = distributed_sum_dict(metric_sums, device)

    averaged = {key: value / max(count, 1) for key, value in metric_sums.items()}
    return EpochResult(loss=total_loss / max(count, 1), metrics=averaged)


def _maybe_wrap_ddp(codec: ReasoningCodec, dist_ctx: DistributedContext):
    if not dist_ctx.enabled:
        return codec
    if dist_ctx.device.type == "cuda":
        return DDP(codec, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, find_unused_parameters=False)
    return DDP(codec, find_unused_parameters=False)


def _clip_grad_norm(model, max_norm: float) -> None:
    params = [param for param in model.parameters() if param.grad is not None]
    if params:
        nn.utils.clip_grad_norm_(params, max_norm)


def _save_model_state(model, config: ExperimentConfig, valid_metrics: dict[str, float], path: Path, dist_ctx: DistributedContext) -> None:
    if dist_ctx.is_main_process:
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save({"model_state": raw_model.state_dict(), "config": config.dump_dict(), "valid_metrics": valid_metrics}, path)


def train_codec(
    codec: ReasoningCodec,
    loaders,
    config: ExperimentConfig,
) -> Path:
    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    device = dist_ctx.device
    if config.model.init_checkpoint:
        raise ValueError("Experiment 1 only supports initialization via model.model_name_or_path. Leave model.init_checkpoint empty.")
    codec.to(device)
    codec = _maybe_wrap_ddp(codec, dist_ctx)
    optimizer = torch.optim.AdamW(
        codec.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "codec_best.pt"
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    best_valid = float("inf")
    writer = _build_summary_writer(config, "codec", dist_ctx)
    global_step = 0

    try:
        for epoch in range(config.train.epochs):
            if dist_ctx.enabled and hasattr(loaders["train"], "sampler") and hasattr(loaders["train"].sampler, "set_epoch"):
                loaders["train"].sampler.set_epoch(epoch)
            codec.train()
            progress = tqdm(loaders["train"], desc=f"codec epoch {epoch + 1}/{config.train.epochs}", disable=not dist_ctx.is_main_process)
            running_loss = 0.0
            metric_sums: dict[str, float] = {}
            steps_in_epoch = 0

            for step, batch in enumerate(progress, start=1):
                optimizer.zero_grad()
                loss, metrics = compute_codec_loss(codec, batch, config.codec_objective, device)
                loss.backward()
                _clip_grad_norm(codec, config.train.grad_clip_norm)
                optimizer.step()

                global_step += 1
                steps_in_epoch += 1
                running_loss += loss.item()
                step_loss_value = distributed_mean(loss.item(), device) if dist_ctx.enabled else loss.item()
                if writer is not None:
                    writer.add_scalar("train/step_loss", step_loss_value, global_step)
                _write_metrics(writer, "train_step", metrics, global_step)
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value

                if dist_ctx.is_main_process and (step % config.train.log_every == 0 or step == len(loaders["train"])):
                    progress.set_postfix(loss=running_loss / step, **metrics)

            if dist_ctx.enabled:
                loss_tensor = torch.tensor([running_loss, steps_in_epoch], device=device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor[0].item()
                steps_in_epoch = int(loss_tensor[1].item())
                metric_sums = distributed_sum_dict(metric_sums, device)

            averaged_train_metrics = {key: value / max(steps_in_epoch, 1) for key, value in metric_sums.items()}
            train_result = EpochResult(loss=running_loss / max(steps_in_epoch, 1), metrics=averaged_train_metrics)
            valid_result = evaluate_codec(codec, loaders["valid"], config, device)
            if dist_ctx.is_main_process:
                history["train"].append({"loss": train_result.loss, **train_result.metrics})
                history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
            if writer is not None:
                writer.add_scalar("train/epoch_loss", train_result.loss, epoch + 1)
                writer.add_scalar("valid/loss", valid_result.loss, epoch + 1)
            _write_metrics(writer, "train_epoch", train_result.metrics, epoch + 1)
            _write_metrics(writer, "valid", valid_result.metrics, epoch + 1)

            if valid_result.loss < best_valid:
                best_valid = valid_result.loss
                _save_model_state(codec, config, valid_result.metrics, best_path, dist_ctx)

        if dist_ctx.is_main_process:
            save_json(output_dir / "codec_history.json", history)
        return best_path
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()
