from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mtp_latent.config import CodecObjectiveConfig, ExperimentConfig
from mtp_latent.metrics import cosine_retrieval_metrics, masked_token_accuracy
from mtp_latent.models import ReasoningCodec, ReasoningTransitionModel, SFTLanguageModel
from mtp_latent.utils import (
    DistributedContext,
    cleanup_distributed,
    configure_torch_runtime,
    distributed_mean,
    distributed_sum_dict,
    ensure_dir,
    init_distributed,
    save_json,
)


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


def _progress_metrics(stage: str, metrics: dict[str, float]) -> dict[str, float]:
    if stage == "transition":
        keys = [
            "type_loss",
            "decode_loss",
            "latent_loss",
            "type_acc",
            "decode_token_acc",
            "answer_rate",
        ]
        return {key: metrics[key] for key in keys if key in metrics}
    return metrics


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def _weighted_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    token_loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(targets)
    valid_mask = (targets != ignore_index).float()
    token_counts = valid_mask.sum(dim=1)
    safe_token_counts = token_counts.clamp_min(1.0)
    sample_loss = (token_loss * valid_mask).sum(dim=1) / safe_token_counts
    active_mask = (token_counts > 0).to(sample_loss.dtype)
    weighted_losses = sample_loss * sample_weights * active_mask
    return weighted_losses.sum() / (sample_weights * active_mask).sum().clamp_min(1e-8)


def _classification_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    predictions = logits.argmax(dim=-1)
    mask = targets != ignore_index
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = ((predictions == targets) & mask).sum().item()
    return correct / total


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    per_example = nn.functional.huber_loss(predictions, targets, reduction="none", delta=delta).mean(dim=-1)
    return _masked_mean(per_example, mask)


def _info_nce_loss(predictions: torch.Tensor, targets: torch.Tensor, temperature: float) -> torch.Tensor:
    normalized_predictions = nn.functional.normalize(predictions, dim=-1)
    normalized_targets = nn.functional.normalize(targets, dim=-1)
    logits = normalized_predictions @ normalized_targets.transpose(0, 1)
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


def _within_question_info_nce_loss(
    predicted_latents: torch.Tensor,
    latent_targets: torch.Tensor,
    latent_target_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for row in range(predicted_latents.size(0)):
        valid_positions = torch.nonzero(latent_target_mask[row], as_tuple=False).flatten()
        if valid_positions.numel() < 2:
            continue
        normalized_predictions = nn.functional.normalize(predicted_latents[row, valid_positions], dim=-1)
        normalized_targets = nn.functional.normalize(latent_targets[row, valid_positions], dim=-1)
        logits = normalized_predictions @ normalized_targets.transpose(0, 1)
        logits = logits / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        losses.append(nn.functional.cross_entropy(logits, labels))
    if losses:
        return torch.stack(losses).mean()
    return predicted_latents.new_zeros(())


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported train.precision={precision}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _answer_sample_weights(batch, horizon: int, device: torch.device, answer_loss_weight: float) -> torch.Tensor:
    weights = torch.ones(len(batch["future_kinds"]), device=device)
    if answer_loss_weight == 1.0:
        return weights
    for index, future_kinds in enumerate(batch["future_kinds"]):
        if horizon < len(future_kinds) and future_kinds[horizon] == "answer":
            weights[index] = answer_loss_weight
    return weights


def compute_codec_loss(
    codec: ReasoningCodec,
    batch: dict[str, torch.Tensor | list[str] | list[list[str]]],
    objective_config: CodecObjectiveConfig,
    device: torch.device,
    precision: str = "fp32",
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_codec = _unwrap_codec(codec)
    prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
    prefix_mask = batch["prefix_mask"].to(device, non_blocking=True)
    horizon_mask = batch["horizon_mask"].to(device, non_blocking=True)
    autocast_dtype = _autocast_dtype(precision) if device.type == "cuda" else None

    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        latent = raw_codec.encode(prefix_ids, prefix_mask)

        for horizon, target_tokens in enumerate(batch["target_steps"]):
            target_tokens = target_tokens.to(device, non_blocking=True)
            target_labels = batch["target_labels"][horizon].to(device, non_blocking=True)
            sample_weights = _answer_sample_weights(batch, horizon, device, objective_config.answer_loss_weight)
            if target_tokens.size(1) == 0:
                continue
            active = horizon_mask[:, horizon].to(latent.dtype).mean()

            if horizon > 0:
                continue

            if objective_config.name == "standard":
                logits = raw_codec.decode(latent, target_tokens)
                targets = target_labels
                sample_loss = _weighted_cross_entropy_loss(logits, targets, sample_weights)
                weight = objective_config.horizon_weights[min(horizon, len(objective_config.horizon_weights) - 1)]
                losses.append(weight * sample_loss * active)
                if compute_metrics:
                    metrics["token_h1_loss"] = sample_loss.detach().item()
                    metrics["token_h1_acc"] = masked_token_accuracy(
                        logits.detach(),
                        targets.detach(),
                    )
                    metrics["primary_loss"] = metrics["token_h1_loss"]
                continue

            if objective_config.name == "decoder_token_mtp":
                token_horizons = objective_config.token_prediction_horizons
                logits_by_horizon = raw_codec.decode_multi_horizon(latent, target_tokens, token_horizons)
                base_targets = target_labels

                for token_horizon in token_horizons:
                    logits = logits_by_horizon[token_horizon]
                    offset = token_horizon - 1
                    if offset > 0:
                        if logits.size(1) <= offset:
                            continue
                        logits = logits[:, :-offset, :]
                        targets = base_targets[:, offset:]
                    else:
                        targets = base_targets

                    weight = objective_config.token_prediction_weights[
                        min(token_horizon - 1, len(objective_config.token_prediction_weights) - 1)
                    ]
                    sample_loss = _weighted_cross_entropy_loss(logits, targets, sample_weights)
                    losses.append(weight * sample_loss * active)
                    if compute_metrics:
                        metrics[f"token_h{token_horizon}_loss"] = sample_loss.detach().item()
                        metrics[f"token_h{token_horizon}_acc"] = masked_token_accuracy(
                            logits.detach(),
                            targets.detach(),
                        )
                if compute_metrics and "token_h1_loss" in metrics:
                    metrics["primary_loss"] = metrics["token_h1_loss"]
                continue

            raise ValueError(f"Unsupported codec objective: {objective_config.name}")

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
    answer_correct = 0.0
    answer_total = 0.0
    answer_token_loss_sum = 0.0
    answer_token_acc_sum = 0.0
    answer_batch_count = 0
    raw_codec = _unwrap_codec(codec)
    tokenizer = loader.dataset.tokenizer
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            loss, metrics = compute_codec_loss(codec, batch, config.codec_objective, device, config.train.precision)
            total_loss += loss.item()
            count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

            answer_indices = [
                index
                for index, future_kinds in enumerate(batch["future_kinds"])
                if future_kinds and future_kinds[0] == "answer"
            ]
            if answer_indices:
                prefix_ids = batch["prefix_ids"][answer_indices].to(device, non_blocking=True)
                prefix_mask = batch["prefix_mask"][answer_indices].to(device, non_blocking=True)
                answer_target_tokens = batch["target_steps"][0][answer_indices].to(device, non_blocking=True)
                answer_target_labels = batch["target_labels"][0][answer_indices].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    latent = raw_codec.encode(prefix_ids, prefix_mask)
                    answer_logits = raw_codec.decode(latent, answer_target_tokens)
                    generated_token_ids, _ = raw_codec.generate_step(
                        latent,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=config.data.max_step_tokens,
                    )
                answer_token_loss_sum += _cross_entropy_loss(answer_logits, answer_target_labels).item()
                answer_token_acc_sum += masked_token_accuracy(answer_logits.detach(), answer_target_labels.detach())
                answer_batch_count += 1
                predictions = _decode_generated_tokens(tokenizer, generated_token_ids)
                gold_answers = [batch["future_texts"][index][0] for index in answer_indices]
                answer_correct += sum(
                    1.0
                    for prediction, gold_answer in zip(predictions, gold_answers)
                    if _normalize_text(prediction) == _normalize_text(gold_answer)
                )
                answer_total += float(len(gold_answers))

    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        count = int(loss_tensor[1].item())
        metric_sums = distributed_sum_dict(metric_sums, device)
        answer_tensor = torch.tensor([answer_correct, answer_total], device=device, dtype=torch.float64)
        dist.all_reduce(answer_tensor, op=dist.ReduceOp.SUM)
        answer_correct = answer_tensor[0].item()
        answer_total = answer_tensor[1].item()
        answer_metric_tensor = torch.tensor(
            [answer_token_loss_sum, answer_token_acc_sum, answer_batch_count],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(answer_metric_tensor, op=dist.ReduceOp.SUM)
        answer_token_loss_sum = answer_metric_tensor[0].item()
        answer_token_acc_sum = answer_metric_tensor[1].item()
        answer_batch_count = int(answer_metric_tensor[2].item())

    averaged = {key: value / max(count, 1) for key, value in metric_sums.items()}
    if answer_total > 0:
        averaged["answer_acc"] = answer_correct / answer_total
        averaged["answer_count"] = answer_total
    if answer_batch_count > 0:
        averaged["answer_token_loss"] = answer_token_loss_sum / answer_batch_count
        averaged["answer_token_acc"] = answer_token_acc_sum / answer_batch_count
    return EpochResult(loss=total_loss / max(count, 1), metrics=averaged)


def _decode_generated_tokens(tokenizer, generated_token_ids: torch.Tensor) -> list[str]:
    eos_id = tokenizer.eos_token_id
    decoded: list[str] = []
    for row in range(generated_token_ids.size(0)):
        token_list = generated_token_ids[row].tolist()
        if eos_id in token_list:
            token_list = token_list[: token_list.index(eos_id)]
        decoded.append(tokenizer.decode(token_list, skip_special_tokens=True).strip())
    return decoded


def _collect_valid_generations(
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    raw_codec = _unwrap_codec(codec)
    tokenizer = loader.dataset.tokenizer
    results: list[dict[str, Any]] = []
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            prefix_ids = batch["prefix_ids"].to(device, non_blocking=True)
            prefix_mask = batch["prefix_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                latent = raw_codec.encode(prefix_ids, prefix_mask)
                generated_token_ids, finished = raw_codec.generate_step(
                    latent,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=config.data.max_step_tokens,
                )

            predictions = _decode_generated_tokens(tokenizer, generated_token_ids)
            current_targets = [future_steps[0] if future_steps else "" for future_steps in batch["future_texts"]]
            for index, prediction in enumerate(predictions):
                target_kind = batch["future_kinds"][index][0] if batch["future_kinds"][index] else ""
                answer_correct = None
                if target_kind == "answer":
                    answer_correct = _normalize_text(prediction) == _normalize_text(current_targets[index])
                results.append(
                    {
                        "prefix_text": batch["prefix_texts"][index],
                        "target_kind": target_kind,
                        "target_text": current_targets[index],
                        "predicted_text": prediction,
                        "finished_with_eos": bool(finished[index].item()),
                        "answer_correct": answer_correct,
                    }
                )
                if len(results) >= max_examples:
                    return results
    return results


def _maybe_wrap_ddp(codec: ReasoningCodec, dist_ctx: DistributedContext, static_graph: bool = True):
    if not dist_ctx.enabled:
        return codec
    kwargs = {"find_unused_parameters": False, "static_graph": static_graph}
    if dist_ctx.device.type == "cuda":
        try:
            return DDP(codec, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, **kwargs)
        except TypeError:
            kwargs.pop("static_graph", None)
            return DDP(codec, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, **kwargs)
    try:
        return DDP(codec, **kwargs)
    except TypeError:
        kwargs.pop("static_graph", None)
        return DDP(codec, **kwargs)


def _build_optimizer(model, config: ExperimentConfig, device: torch.device) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
    }
    if device.type == "cuda" and config.train.fused_optimizer:
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **kwargs)


def _clip_grad_norm(model, max_norm: float) -> None:
    params = [param for param in model.parameters() if param.grad is not None]
    if params:
        nn.utils.clip_grad_norm_(params, max_norm)


def _save_model_state(model, config: ExperimentConfig, valid_metrics: dict[str, float], path: Path, dist_ctx: DistributedContext) -> None:
    if dist_ctx.is_main_process:
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save({"model_state": raw_model.state_dict(), "config": config.dump_dict(), "valid_metrics": valid_metrics}, path)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig, total_steps: int) -> LambdaLR | None:
    if total_steps <= 0 or config.train.scheduler == "none":
        return None
    if config.train.scheduler != "cosine":
        raise ValueError(f"Unsupported train.scheduler={config.train.scheduler}")

    warmup_steps = int(total_steps * config.train.warmup_ratio)
    min_lr_ratio = config.train.min_lr_ratio

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(warmup_steps, 1))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _num_update_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    return max(math.ceil(num_batches / max(grad_accum_steps, 1)), 1)


def _current_accum_divisor(step: int, num_batches: int, grad_accum_steps: int) -> int:
    micro_step_in_accum = (step - 1) % grad_accum_steps + 1
    if step == num_batches and micro_step_in_accum != grad_accum_steps:
        return micro_step_in_accum
    return grad_accum_steps


def _build_compact_valid_entry(
    epoch: int,
    stage: str,
    loss: float,
    metrics: dict[str, float],
) -> dict[str, float | int | str]:
    entry: dict[str, float | int | str] = {
        "epoch": epoch,
        "stage": stage,
        "loss": round(loss, 6),
    }
    preferred_keys = [
        "primary_loss",
        "token_h1_loss",
        "token_h1_acc",
        "step_acc",
        "answer_token_loss",
        "answer_token_acc",
        "answer_acc",
        "type_loss",
        "decode_loss",
        "latent_loss",
        "infonce_loss",
        "latent_huber_loss",
        "type_acc",
        "decode_token_acc",
        "pred_vs_target_latent_cosine",
        "pred_vs_target_latent_mse",
        "pred_vs_random_target_latent_cosine",
        "teacher_forced_answer_acc",
        "rollout_direct_answer_acc",
        "rollout_direct_answer_stop_rate",
    ]
    for key in preferred_keys:
        if key in metrics:
            entry[key] = round(float(metrics[key]), 6)
    return entry


def _build_compact_valid_report(
    experiment_name: str,
    stage: str,
    entries: list[dict[str, float | int | str]],
) -> dict[str, Any]:
    if not entries:
        return {
            "experiment_name": experiment_name,
            "stage": stage,
            "epoch_count": 0,
            "latest": None,
            "best_by_loss": None,
            "best_by_metric": {},
            "recent_epochs": [],
        }

    minimize_metrics = {
        "loss",
        "primary_loss",
        "token_h1_loss",
        "answer_token_loss",
        "type_loss",
        "decode_loss",
        "latent_loss",
        "infonce_loss",
        "latent_huber_loss",
    }
    maximize_metrics = {
        "token_h1_acc",
        "step_acc",
        "answer_token_acc",
        "answer_acc",
        "type_acc",
        "decode_token_acc",
        "pred_vs_target_latent_cosine",
        "pred_vs_target_latent_mse",
        "pred_vs_random_target_latent_cosine",
        "teacher_forced_answer_acc",
        "rollout_direct_answer_acc",
        "rollout_direct_answer_stop_rate",
    }

    best_by_loss = min(entries, key=lambda entry: float(entry["loss"]))
    best_by_metric: dict[str, dict[str, float | int]] = {}

    metric_keys = [
        key
        for key in entries[0].keys()
        if key not in {"epoch", "stage"}
    ]
    for key in metric_keys:
        if key in minimize_metrics:
            best_entry = min(entries, key=lambda entry: float(entry[key]))
        elif key in maximize_metrics:
            best_entry = max(entries, key=lambda entry: float(entry[key]))
        else:
            continue
        best_by_metric[key] = {
            "epoch": int(best_entry["epoch"]),
            "value": float(best_entry[key]),
        }

    return {
        "experiment_name": experiment_name,
        "stage": stage,
        "epoch_count": len(entries),
        "latest": entries[-1],
        "best_by_loss": {
            "epoch": int(best_by_loss["epoch"]),
            "loss": float(best_by_loss["loss"]),
        },
        "best_by_metric": best_by_metric,
        "recent_epochs": entries[-5:],
    }


def compute_sft_loss(
    model: SFTLanguageModel,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    autocast_dtype = _autocast_dtype(precision) if device.type == "cuda" else None

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        logits = model(input_ids, attention_mask)
        loss = _cross_entropy_loss(logits, labels)

    metrics = {
        "token_loss": loss.detach().item(),
        "token_acc": masked_token_accuracy(logits.detach(), labels.detach()),
        "primary_loss": loss.detach().item(),
    }
    return loss, metrics


def _decode_sft_generated(
    tokenizer,
    generated_token_ids: torch.Tensor,
) -> list[str]:
    decoded: list[str] = []
    eos_id = tokenizer.eos_token_id
    for row in range(generated_token_ids.size(0)):
        tokens = generated_token_ids[row].tolist()
        if eos_id in tokens:
            tokens = tokens[:tokens.index(eos_id)]
        decoded.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return decoded


def evaluate_sft(
    model: SFTLanguageModel,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    count = 0
    metric_sums: dict[str, float] = {}
    step_correct = 0.0
    step_total = 0.0
    answer_correct = 0.0
    answer_total = 0.0
    tokenizer = loader.dataset.tokenizer
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            loss, metrics = compute_sft_loss(model, batch, device, config.train.precision)
            total_loss += loss.item()
            count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                generated_token_ids, _ = model.generate_continuation(
                    input_ids,
                    attention_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=config.data.max_step_tokens,
                )
            predictions = _decode_sft_generated(tokenizer, generated_token_ids)
            for prediction, target_text, target_kind in zip(predictions, batch["target_texts"], batch["target_kinds"]):
                if target_kind == "step":
                    step_total += 1.0
                    if _normalize_text(prediction) == _normalize_text(target_text):
                        step_correct += 1.0
                elif target_kind == "answer":
                    answer_total += 1.0
                    if _normalize_text(prediction) == _normalize_text(target_text):
                        answer_correct += 1.0

    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        count = int(loss_tensor[1].item())
        metric_sums = distributed_sum_dict(metric_sums, device)
        counts = torch.tensor([step_correct, step_total, answer_correct, answer_total], device=device, dtype=torch.float64)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        step_correct, step_total, answer_correct, answer_total = counts.tolist()

    averaged = {key: value / max(count, 1) for key, value in metric_sums.items()}
    if step_total > 0:
        averaged["step_acc"] = step_correct / step_total
        averaged["step_count"] = step_total
    if answer_total > 0:
        averaged["answer_acc"] = answer_correct / answer_total
        averaged["answer_count"] = answer_total
    return EpochResult(loss=total_loss / max(count, 1), metrics=averaged)


def _collect_sft_valid_generations(
    model: SFTLanguageModel,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    tokenizer = loader.dataset.tokenizer
    results: list[dict[str, Any]] = []
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                generated_token_ids, finished = model.generate_continuation(
                    input_ids,
                    attention_mask,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=config.data.max_step_tokens,
                )
            predictions = _decode_sft_generated(tokenizer, generated_token_ids)
            for idx, prediction in enumerate(predictions):
                results.append(
                    {
                        "prompt_text": batch["prompt_texts"][idx],
                        "target_kind": batch["target_kinds"][idx],
                        "target_text": batch["target_texts"][idx],
                        "predicted_text": prediction,
                        "finished_with_eos": bool(finished[idx].item()),
                    }
                )
                if len(results) >= max_examples:
                    return results
    return results


def _flatten_active_transition_targets(
    predicted_latents: torch.Tensor,
    target_tokens: torch.Tensor,
    target_labels: torch.Tensor,
    target_type_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = target_type_labels != -100
    return (
        predicted_latents[active_mask],
        target_tokens[active_mask],
        target_labels[active_mask],
        target_type_labels[active_mask],
    )


def _build_transition_latent_targets(
    latent_inputs: torch.Tensor,
    target_type_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_targets = target_type_labels.shape
    latent_dim = latent_inputs.size(-1) if latent_inputs.dim() == 3 and latent_inputs.size(1) > 0 else 0
    latent_targets = torch.zeros(
        (batch_size, max_targets, latent_dim),
        device=latent_inputs.device,
        dtype=latent_inputs.dtype,
    )
    latent_target_mask = torch.zeros((batch_size, max_targets), device=latent_inputs.device, dtype=torch.bool)
    max_available = min(latent_inputs.size(1), max_targets)
    if max_available > 0:
        latent_targets[:, :max_available, :] = latent_inputs[:, :max_available, :]
        latent_target_mask[:, :max_available] = target_type_labels[:, :max_available] == 0
    return latent_targets, latent_target_mask


def _encode_transition_latents(
    codec: ReasoningCodec,
    batch: dict[str, torch.Tensor | list[str] | list[list[str]]],
    device: torch.device,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_prefix_ids = batch["latent_prefix_ids"].to(device, non_blocking=True)
    latent_prefix_mask = batch["latent_prefix_mask"].to(device, non_blocking=True)
    latent_mask = batch["latent_mask"].to(device, non_blocking=True)
    autocast_dtype = _autocast_dtype(precision) if device.type == "cuda" else None

    batch_size, max_latents, seq_len = latent_prefix_ids.shape
    if max_latents == 0:
        return torch.zeros((batch_size, 0, codec.model_config.latent_dim), device=device), latent_mask

    flat_active_mask = latent_mask.view(-1)
    flat_prefix_ids = latent_prefix_ids.view(batch_size * max_latents, seq_len)
    flat_prefix_mask = latent_prefix_mask.view(batch_size * max_latents, seq_len)

    flat_latents = torch.zeros(
        (batch_size * max_latents, codec.model_config.latent_dim),
        device=device,
        dtype=next(codec.parameters()).dtype,
    )
    if flat_active_mask.any():
        active_prefix_ids = flat_prefix_ids[flat_active_mask]
        active_prefix_mask = flat_prefix_mask[flat_active_mask]
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            encoded_latents = codec.encode(active_prefix_ids, active_prefix_mask)
        flat_latents[flat_active_mask] = encoded_latents.to(flat_latents.dtype)
    return flat_latents.view(batch_size, max_latents, -1), latent_mask


def _encode_prefix_text(
    codec: ReasoningCodec,
    tokenizer,
    prefix_text: str,
    max_prefix_tokens: int,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    encoded = tokenizer(
        [prefix_text],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_prefix_tokens,
        return_tensors="pt",
    )
    prefix_ids = encoded["input_ids"].to(device, non_blocking=True)
    prefix_mask = encoded["attention_mask"].bool().to(device, non_blocking=True)
    autocast_dtype = _autocast_dtype(precision) if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        return codec.encode(prefix_ids, prefix_mask)


def _predict_transition_next(
    transition_model: ReasoningTransitionModel,
    tokenizer,
    question_text: str,
    latent_history: list[torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    encoded_question = tokenizer(
        [question_text],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=config.data.max_prefix_tokens,
        return_tensors="pt",
    )
    question_ids = encoded_question["input_ids"].to(device, non_blocking=True)
    question_mask = encoded_question["attention_mask"].bool().to(device, non_blocking=True)

    if latent_history:
        latent_inputs = torch.cat(latent_history, dim=0).unsqueeze(0).to(device, non_blocking=True)
        latent_mask = torch.ones((1, len(latent_history)), dtype=torch.bool, device=device)
    else:
        latent_inputs = torch.zeros((1, 0, config.model.latent_dim), device=device, dtype=next(transition_model.parameters()).dtype)
        latent_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)

    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        next_type_logits, predicted_latents, _ = transition_model(
            question_ids,
            question_mask,
            latent_inputs,
            latent_mask,
        )
    position_index = len(latent_history)
    predicted_type = int(next_type_logits[0, position_index].argmax(dim=-1).item())
    predicted_latent = predicted_latents[0, position_index].unsqueeze(0)
    return predicted_type, predicted_latent


def _decode_latent_text(
    codec: ReasoningCodec,
    tokenizer,
    latent: torch.Tensor,
    max_step_tokens: int,
    precision: str,
) -> tuple[str, bool]:
    device = latent.device
    autocast_dtype = _autocast_dtype(precision) if device.type == "cuda" else None
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        generated_token_ids, finished = codec.generate_step(
            latent,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_step_tokens,
        )
    decoded_text = _decode_generated_tokens(tokenizer, generated_token_ids)[0]
    return decoded_text, bool(finished[0].item())


def _rollout_transition_answer(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    tokenizer,
    question_text: str,
    config: ExperimentConfig,
    device: torch.device,
    mode: str,
    max_rollout_steps: int,
) -> dict[str, Any]:
    if mode not in {"direct", "reencode"}:
        raise ValueError(f"Unsupported rollout mode: {mode}")

    latent_history: list[torch.Tensor] = []
    generated_steps: list[str] = []
    current_prefix_text = question_text
    answer_text = ""
    answer_finished = False
    stopped_by_answer = False

    for _ in range(max_rollout_steps):
        predicted_type, predicted_latent = _predict_transition_next(
            transition_model,
            tokenizer,
            question_text,
            latent_history,
            config,
            device,
        )
        decoded_text, finished = _decode_latent_text(
            codec,
            tokenizer,
            predicted_latent,
            config.data.max_step_tokens,
            config.train.precision,
        )
        if predicted_type == 1:
            answer_text = decoded_text
            answer_finished = finished
            stopped_by_answer = True
            break

        generated_steps.append(decoded_text)
        if mode == "direct":
            latent_history.append(predicted_latent.detach().to(next(codec.parameters()).dtype))
        else:
            current_prefix_text = (
                f"{current_prefix_text}{config.data.text_separator}{decoded_text}".strip()
                if decoded_text
                else current_prefix_text
            )
            reencoded_latent = _encode_prefix_text(
                codec,
                tokenizer,
                current_prefix_text,
                config.data.max_prefix_tokens,
                device,
                config.train.precision,
            )
            latent_history.append(reencoded_latent.detach().to(next(codec.parameters()).dtype))

    return {
        "predicted_answer": answer_text,
        "finished_with_eos": answer_finished,
        "stopped_by_answer_type": stopped_by_answer,
        "generated_steps": generated_steps,
    }


def _evaluate_transition_rollout_metrics(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    raw_transition = transition_model.module if isinstance(transition_model, DDP) else transition_model
    tokenizer = loader.dataset.tokenizer
    counts = {
        "teacher_forced_answer_correct": 0.0,
        "teacher_forced_answer_total": 0.0,
        "rollout_direct_answer_correct": 0.0,
        "rollout_direct_answer_total": 0.0,
        "rollout_direct_answer_stop": 0.0,
        "rollout_reencode_answer_correct": 0.0,
        "rollout_reencode_answer_total": 0.0,
        "rollout_reencode_answer_stop": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            question_ids = batch["question_ids"].to(device, non_blocking=True)
            question_mask = batch["question_mask"].to(device, non_blocking=True)
            latent_inputs, latent_mask = _encode_transition_latents(codec, batch, device, config.train.precision)
            autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                _, predicted_latents, supervision_mask = raw_transition(
                    question_ids,
                    question_mask,
                    latent_inputs,
                    latent_mask,
                )

            for batch_index, answer_text in enumerate(batch["answers"]):
                if not answer_text:
                    continue

                valid_positions = int(supervision_mask[batch_index].sum().item())
                if valid_positions <= 0:
                    continue

                answer_position = valid_positions - 1
                teacher_forced_latent = predicted_latents[batch_index, answer_position].unsqueeze(0)
                teacher_forced_prediction, _ = _decode_latent_text(
                    codec,
                    tokenizer,
                    teacher_forced_latent,
                    config.data.max_step_tokens,
                    config.train.precision,
                )
                counts["teacher_forced_answer_total"] += 1.0
                if _normalize_text(teacher_forced_prediction) == _normalize_text(answer_text):
                    counts["teacher_forced_answer_correct"] += 1.0

                max_rollout_steps = len(batch["target_texts"][batch_index])
                direct_rollout = _rollout_transition_answer(
                    raw_transition,
                    codec,
                    tokenizer,
                    batch["question_texts"][batch_index],
                    config,
                    device,
                    mode="direct",
                    max_rollout_steps=max_rollout_steps,
                )
                counts["rollout_direct_answer_total"] += 1.0
                if direct_rollout["stopped_by_answer_type"]:
                    counts["rollout_direct_answer_stop"] += 1.0
                if _normalize_text(direct_rollout["predicted_answer"]) == _normalize_text(answer_text):
                    counts["rollout_direct_answer_correct"] += 1.0

                reencode_rollout = _rollout_transition_answer(
                    raw_transition,
                    codec,
                    tokenizer,
                    batch["question_texts"][batch_index],
                    config,
                    device,
                    mode="reencode",
                    max_rollout_steps=max_rollout_steps,
                )
                counts["rollout_reencode_answer_total"] += 1.0
                if reencode_rollout["stopped_by_answer_type"]:
                    counts["rollout_reencode_answer_stop"] += 1.0
                if _normalize_text(reencode_rollout["predicted_answer"]) == _normalize_text(answer_text):
                    counts["rollout_reencode_answer_correct"] += 1.0

    if dist.is_available() and dist.is_initialized():
        counts = distributed_sum_dict(counts, device)

    metrics: dict[str, float] = {}
    if counts["teacher_forced_answer_total"] > 0:
        metrics["teacher_forced_answer_acc"] = (
            counts["teacher_forced_answer_correct"] / counts["teacher_forced_answer_total"]
        )
    if counts["rollout_direct_answer_total"] > 0:
        metrics["rollout_direct_answer_acc"] = (
            counts["rollout_direct_answer_correct"] / counts["rollout_direct_answer_total"]
        )
        metrics["rollout_direct_answer_stop_rate"] = (
            counts["rollout_direct_answer_stop"] / counts["rollout_direct_answer_total"]
        )
    if counts["rollout_reencode_answer_total"] > 0:
        metrics["rollout_reencode_answer_acc"] = (
            counts["rollout_reencode_answer_correct"] / counts["rollout_reencode_answer_total"]
        )
        metrics["rollout_reencode_answer_stop_rate"] = (
            counts["rollout_reencode_answer_stop"] / counts["rollout_reencode_answer_total"]
        )
    return metrics


def compute_transition_loss(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    batch: dict[str, torch.Tensor | list[str] | list[list[str]]],
    config: ExperimentConfig,
    device: torch.device,
    compute_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_transition = transition_model.module if isinstance(transition_model, DDP) else transition_model
    question_ids = batch["question_ids"].to(device, non_blocking=True)
    question_mask = batch["question_mask"].to(device, non_blocking=True)
    target_tokens = batch["target_tokens"].to(device, non_blocking=True)
    target_labels = batch["target_labels"].to(device, non_blocking=True)
    target_type_labels = batch["target_type_labels"].to(device, non_blocking=True)
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        latent_inputs, latent_mask = _encode_transition_latents(codec, batch, device, config.train.precision)
        next_type_logits, predicted_latents, supervision_mask = raw_transition(
            question_ids,
            question_mask,
            latent_inputs,
            latent_mask,
        )
        masked_type_labels = target_type_labels.masked_fill(~supervision_mask, -100)
        type_loss = _cross_entropy_loss(next_type_logits, masked_type_labels)

        flat_predicted_latents, flat_target_tokens, flat_target_labels, flat_target_types = _flatten_active_transition_targets(
            predicted_latents,
            target_tokens,
            target_labels,
            masked_type_labels,
        )
        decode_logits = codec.decode(flat_predicted_latents, flat_target_tokens)
        decode_loss = _cross_entropy_loss(decode_logits, flat_target_labels)

        latent_targets, latent_target_mask = _build_transition_latent_targets(latent_inputs, target_type_labels)
        zero = predicted_latents.new_zeros(())
        if latent_target_mask.any():
            cosine_scores = nn.functional.cosine_similarity(predicted_latents, latent_targets, dim=-1)
            cosine_latent_loss = _masked_mean(1.0 - cosine_scores, latent_target_mask)
            latent_mse = _masked_mean(
                ((predicted_latents - latent_targets) ** 2).mean(dim=-1),
                latent_target_mask,
            )
            latent_huber_loss = _masked_huber_loss(
                predicted_latents,
                latent_targets,
                latent_target_mask,
                delta=config.transition.latent_huber_delta,
            )
            pred_vs_target_latent_cosine = _masked_mean(cosine_scores, latent_target_mask)
            flat_predicted_next = predicted_latents[latent_target_mask]
            flat_target_next = latent_targets[latent_target_mask]
            infonce_loss = _within_question_info_nce_loss(
                predicted_latents,
                latent_targets,
                latent_target_mask,
                temperature=config.transition.latent_infonce_temperature,
            )
            if flat_target_next.size(0) > 1:
                pred_vs_random_target_latent_cosine = nn.functional.cosine_similarity(
                    flat_predicted_next,
                    flat_target_next.roll(shifts=1, dims=0),
                    dim=-1,
                ).mean()
            else:
                pred_vs_random_target_latent_cosine = predicted_latents.new_zeros(())
        else:
            cosine_latent_loss = zero
            latent_mse = zero
            latent_huber_loss = zero
            infonce_loss = zero
            pred_vs_target_latent_cosine = zero
            pred_vs_random_target_latent_cosine = zero

        latent_loss_type = config.transition.latent_loss_type
        if latent_loss_type == "cosine":
            latent_loss = cosine_latent_loss
        elif latent_loss_type == "cosine_huber":
            latent_loss = cosine_latent_loss + config.transition.latent_huber_weight * latent_huber_loss
        elif latent_loss_type == "infonce":
            latent_loss = infonce_loss
        elif latent_loss_type == "infonce_huber":
            latent_loss = infonce_loss + config.transition.latent_huber_weight * latent_huber_loss
        else:
            raise ValueError(f"Unsupported transition.latent_loss_type={latent_loss_type}")

        total_loss = (
            config.transition.type_loss_weight * type_loss
            + config.transition.decode_loss_weight * decode_loss
            + config.transition.latent_loss_weight * latent_loss
        )

    metrics = {}
    if compute_metrics:
        metrics = {
            "type_loss": type_loss.detach().item(),
            "decode_loss": decode_loss.detach().item(),
            "latent_loss": latent_loss.detach().item(),
            "infonce_loss": infonce_loss.detach().item(),
            "latent_huber_loss": latent_huber_loss.detach().item(),
            "type_acc": _classification_accuracy(next_type_logits.detach(), masked_type_labels.detach()),
            "decode_token_acc": masked_token_accuracy(decode_logits.detach(), flat_target_labels.detach()),
            "pred_vs_target_latent_cosine": pred_vs_target_latent_cosine.detach().item(),
            "pred_vs_target_latent_mse": latent_mse.detach().item(),
            "pred_vs_random_target_latent_cosine": pred_vs_random_target_latent_cosine.detach().item(),
            "answer_rate": (flat_target_types == 1).float().mean().item() if flat_target_types.numel() > 0 else 0.0,
        }
    return total_loss, metrics


def evaluate_transition(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochResult:
    transition_model.eval()
    total_loss = 0.0
    count = 0
    metric_sums: dict[str, float] = {}

    with torch.no_grad():
        for batch in loader:
            loss, metrics = compute_transition_loss(transition_model, codec, batch, config, device)
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
    averaged.update(_evaluate_transition_rollout_metrics(transition_model, codec, loader, config, device))
    return EpochResult(loss=total_loss / max(count, 1), metrics=averaged)


def _collect_transition_valid_generations(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    raw_transition = transition_model.module if isinstance(transition_model, DDP) else transition_model
    tokenizer = loader.dataset.tokenizer
    results: list[dict[str, Any]] = []
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            question_ids = batch["question_ids"].to(device, non_blocking=True)
            question_mask = batch["question_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                latent_inputs, latent_mask = _encode_transition_latents(codec, batch, device, config.train.precision)
                next_type_logits, predicted_latents, supervision_mask = raw_transition(
                    question_ids,
                    question_mask,
                    latent_inputs,
                    latent_mask,
                )

            predicted_types = next_type_logits.argmax(dim=-1)
            for batch_index in range(question_ids.size(0)):
                valid_positions = int(supervision_mask[batch_index].sum().item())
                for position_index in range(valid_positions):
                    predicted_latent = predicted_latents[batch_index, position_index].unsqueeze(0)
                    predicted_text, finished_with_eos = _decode_latent_text(
                        codec,
                        tokenizer,
                        predicted_latent,
                        config.data.max_step_tokens,
                        config.train.precision,
                    )
                    gold_type = batch["target_kinds"][batch_index][position_index]
                    gold_text = batch["target_texts"][batch_index][position_index]
                    results.append(
                        {
                            "question_text": batch["question_texts"][batch_index],
                            "position_index": position_index,
                            "gold_type": gold_type,
                            "predicted_type": "answer" if int(predicted_types[batch_index, position_index].item()) == 1 else "step",
                            "gold_text": gold_text,
                            "predicted_text": predicted_text,
                            "finished_with_eos": finished_with_eos,
                        }
                    )
                    if len(results) >= max_examples:
                        return results
    return results


def _collect_transition_rollout_samples(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    loader,
    config: ExperimentConfig,
    device: torch.device,
    max_examples: int,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        return []
    raw_transition = transition_model.module if isinstance(transition_model, DDP) else transition_model
    tokenizer = loader.dataset.tokenizer
    samples: list[dict[str, Any]] = []
    autocast_dtype = _autocast_dtype(config.train.precision) if device.type == "cuda" else None

    with torch.no_grad():
        for batch in loader:
            question_ids = batch["question_ids"].to(device, non_blocking=True)
            question_mask = batch["question_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                latent_inputs, latent_mask = _encode_transition_latents(codec, batch, device, config.train.precision)
                _, predicted_latents, supervision_mask = raw_transition(
                    question_ids,
                    question_mask,
                    latent_inputs,
                    latent_mask,
                )
            for batch_index, answer_text in enumerate(batch["answers"]):
                if not answer_text:
                    continue

                question_text = batch["question_texts"][batch_index]
                max_rollout_steps = len(batch["target_texts"][batch_index])
                valid_positions = int(supervision_mask[batch_index].sum().item())
                teacher_forced_answer_prediction = ""
                if valid_positions > 0:
                    teacher_forced_latent = predicted_latents[batch_index, valid_positions - 1].unsqueeze(0)
                    teacher_forced_answer_prediction, teacher_forced_finished = _decode_latent_text(
                        codec,
                        tokenizer,
                        teacher_forced_latent,
                        config.data.max_step_tokens,
                        config.train.precision,
                    )
                else:
                    teacher_forced_finished = False

                direct_rollout = _rollout_transition_answer(
                    raw_transition,
                    codec,
                    tokenizer,
                    question_text,
                    config,
                    device,
                    mode="direct",
                    max_rollout_steps=max_rollout_steps,
                )
                reencode_rollout = _rollout_transition_answer(
                    raw_transition,
                    codec,
                    tokenizer,
                    question_text,
                    config,
                    device,
                    mode="reencode",
                    max_rollout_steps=max_rollout_steps,
                )

                samples.append(
                    {
                        "question_text": question_text,
                        "gold_steps": batch["target_texts"][batch_index][:-1],
                        "gold_answer": answer_text,
                        "teacher_forced_answer_target": batch["target_texts"][batch_index][-1],
                        "teacher_forced_answer_prediction": teacher_forced_answer_prediction,
                        "teacher_forced_answer_finished_with_eos": teacher_forced_finished,
                        "rollout_direct": direct_rollout,
                        "rollout_reencode": reencode_rollout,
                    }
                )
                if len(samples) >= max_examples:
                    return samples
    return samples


def train_codec(
    codec: ReasoningCodec,
    loaders,
    config: ExperimentConfig,
) -> Path:
    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    device = dist_ctx.device
    configure_torch_runtime(device, config.train.allow_tf32)
    if config.model.init_checkpoint:
        raise ValueError("Experiment 1 only supports initialization via model.model_name_or_path. Leave model.init_checkpoint empty.")
    codec.to(device)
    codec = _maybe_wrap_ddp(codec, dist_ctx, config.train.ddp_static_graph)
    optimizer = _build_optimizer(codec, config, device)
    total_steps = config.train.epochs * _num_update_steps_per_epoch(
        len(loaders["train"]),
        config.train.grad_accum_steps,
    )
    scheduler = _build_scheduler(optimizer, config, total_steps)
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "codec_best.pt"
    valid_generation_dir = ensure_dir(output_dir / "valid_generations")
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    valid_summary: list[dict[str, float | int | str]] = []
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
            running_loss_tensor = torch.zeros((), device=device)
            metric_sums: dict[str, float] = {}
            metric_steps = 0
            steps_in_epoch = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(progress, start=1):
                should_log_step = step % config.train.log_every == 0 or step == len(loaders["train"])
                should_step = step % config.train.grad_accum_steps == 0 or step == len(loaders["train"])
                accum_steps = _current_accum_divisor(step, len(loaders["train"]), config.train.grad_accum_steps)
                loss, metrics = compute_codec_loss(
                    codec,
                    batch,
                    config.codec_objective,
                    device,
                    config.train.precision,
                    compute_metrics=should_log_step,
                )
                (loss / accum_steps).backward()
                steps_in_epoch += 1
                running_loss_tensor += loss.detach()

                if should_step:
                    _clip_grad_norm(codec, config.train.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if should_log_step:
                    step_loss_value = distributed_mean(loss.detach().item(), device) if dist_ctx.enabled else loss.detach().item()
                    running_loss = running_loss_tensor.item()
                    if writer is not None:
                        writer.add_scalar("train/step_loss", step_loss_value, global_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    _write_metrics(writer, "train_step", metrics, global_step)
                    if metrics:
                        metric_steps += 1
                        for key, value in metrics.items():
                            metric_sums[key] = metric_sums.get(key, 0.0) + value
                    progress.set_postfix(loss=running_loss / step, **_progress_metrics("codec", metrics))

            if dist_ctx.enabled:
                loss_tensor = torch.stack(
                    [
                        running_loss_tensor.to(torch.float64),
                        torch.tensor(float(steps_in_epoch), device=device, dtype=torch.float64),
                    ]
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor[0].item()
                steps_in_epoch = int(loss_tensor[1].item())
                metric_sums = distributed_sum_dict(metric_sums, device)
                metric_steps_tensor = torch.tensor(metric_steps, device=device, dtype=torch.long)
                dist.all_reduce(metric_steps_tensor, op=dist.ReduceOp.SUM)
                metric_steps = int(metric_steps_tensor.item())
            else:
                running_loss = running_loss_tensor.item()

            averaged_train_metrics = {key: value / max(metric_steps, 1) for key, value in metric_sums.items()}
            train_result = EpochResult(loss=running_loss / max(steps_in_epoch, 1), metrics=averaged_train_metrics)
            valid_result = evaluate_codec(codec, loaders["valid"], config, device)
            if dist_ctx.is_main_process:
                history["train"].append({"loss": train_result.loss, **train_result.metrics})
                history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
                valid_generations = _collect_valid_generations(
                    codec,
                    loaders["valid"],
                    config,
                    device,
                    config.train.valid_generate_examples,
                )
                save_json(
                    valid_generation_dir / f"epoch_{epoch + 1:03d}.json",
                    {
                        "epoch": epoch + 1,
                        "experiment_name": config.experiment_name,
                        "valid_loss": valid_result.loss,
                        "valid_metrics": valid_result.metrics,
                        "samples": valid_generations,
                    },
                )
                valid_summary.append(
                    _build_compact_valid_entry(epoch + 1, "codec", valid_result.loss, valid_result.metrics)
                )
                save_json(
                    output_dir / "codec_valid_compact.json",
                    _build_compact_valid_report(config.experiment_name, "codec", valid_summary),
                )
            if dist_ctx.enabled:
                dist.barrier()
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


def train_transition(
    transition_model: ReasoningTransitionModel,
    codec: ReasoningCodec,
    loaders,
    config: ExperimentConfig,
) -> Path:
    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    device = dist_ctx.device
    configure_torch_runtime(device, config.train.allow_tf32)

    codec.to(device)
    codec.eval()
    codec.requires_grad_(False)

    transition_model.to(device)
    transition_model = _maybe_wrap_ddp(transition_model, dist_ctx, config.train.ddp_static_graph)
    optimizer = _build_optimizer(transition_model, config, device)
    total_steps = config.train.epochs * _num_update_steps_per_epoch(
        len(loaders["train"]),
        config.train.grad_accum_steps,
    )
    scheduler = _build_scheduler(optimizer, config, total_steps)
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "transition_best.pt"
    valid_generation_dir = ensure_dir(output_dir / "transition_valid_generations")
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    valid_summary: list[dict[str, float | int | str]] = []
    best_valid = float("inf")
    writer = _build_summary_writer(config, "transition", dist_ctx)
    global_step = 0

    try:
        for epoch in range(config.train.epochs):
            if dist_ctx.enabled and hasattr(loaders["train"], "sampler") and hasattr(loaders["train"].sampler, "set_epoch"):
                loaders["train"].sampler.set_epoch(epoch)
            transition_model.train()
            progress = tqdm(loaders["train"], desc=f"transition epoch {epoch + 1}/{config.train.epochs}", disable=not dist_ctx.is_main_process)
            running_loss = 0.0
            running_loss_tensor = torch.zeros((), device=device)
            metric_sums: dict[str, float] = {}
            metric_steps = 0
            steps_in_epoch = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(progress, start=1):
                should_log_step = step % config.train.log_every == 0 or step == len(loaders["train"])
                should_step = step % config.train.grad_accum_steps == 0 or step == len(loaders["train"])
                accum_steps = _current_accum_divisor(step, len(loaders["train"]), config.train.grad_accum_steps)
                loss, metrics = compute_transition_loss(
                    transition_model,
                    codec,
                    batch,
                    config,
                    device,
                    compute_metrics=should_log_step,
                )
                (loss / accum_steps).backward()
                steps_in_epoch += 1
                running_loss_tensor += loss.detach()

                if should_step:
                    _clip_grad_norm(transition_model, config.train.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if should_log_step:
                    step_loss_value = distributed_mean(loss.detach().item(), device) if dist_ctx.enabled else loss.detach().item()
                    running_loss = running_loss_tensor.item()
                    if writer is not None:
                        writer.add_scalar("train/step_loss", step_loss_value, global_step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    _write_metrics(writer, "train_step", metrics, global_step)
                    if metrics:
                        metric_steps += 1
                        for key, value in metrics.items():
                            metric_sums[key] = metric_sums.get(key, 0.0) + value
                    progress.set_postfix(loss=running_loss / step, **_progress_metrics("transition", metrics))

            if dist_ctx.enabled:
                loss_tensor = torch.stack(
                    [
                        running_loss_tensor.to(torch.float64),
                        torch.tensor(float(steps_in_epoch), device=device, dtype=torch.float64),
                    ]
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor[0].item()
                steps_in_epoch = int(loss_tensor[1].item())
                metric_sums = distributed_sum_dict(metric_sums, device)
                metric_steps_tensor = torch.tensor(metric_steps, device=device, dtype=torch.long)
                dist.all_reduce(metric_steps_tensor, op=dist.ReduceOp.SUM)
                metric_steps = int(metric_steps_tensor.item())
            else:
                running_loss = running_loss_tensor.item()

            averaged_train_metrics = {key: value / max(metric_steps, 1) for key, value in metric_sums.items()}
            train_result = EpochResult(loss=running_loss / max(steps_in_epoch, 1), metrics=averaged_train_metrics)
            valid_result = evaluate_transition(transition_model, codec, loaders["valid"], config, device)
            if dist_ctx.is_main_process:
                history["train"].append({"loss": train_result.loss, **train_result.metrics})
                history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
                valid_generations = _collect_transition_valid_generations(
                    transition_model,
                    codec,
                    loaders["valid"],
                    config,
                    device,
                    config.train.valid_generate_examples,
                )
                rollout_samples = _collect_transition_rollout_samples(
                    transition_model,
                    codec,
                    loaders["valid"],
                    config,
                    device,
                    config.train.valid_generate_examples,
                )
                save_json(
                    valid_generation_dir / f"epoch_{epoch + 1:03d}.json",
                    {
                        "epoch": epoch + 1,
                        "experiment_name": config.experiment_name,
                        "valid_loss": valid_result.loss,
                        "valid_metrics": valid_result.metrics,
                        "samples": valid_generations,
                        "rollout_samples": rollout_samples,
                    },
                )
                valid_summary.append(
                    _build_compact_valid_entry(epoch + 1, "transition", valid_result.loss, valid_result.metrics)
                )
                save_json(
                    output_dir / "transition_valid_compact.json",
                    _build_compact_valid_report(config.experiment_name, "transition", valid_summary),
                )
            if dist_ctx.enabled:
                dist.barrier()
            if writer is not None:
                writer.add_scalar("train/epoch_loss", train_result.loss, epoch + 1)
                writer.add_scalar("valid/loss", valid_result.loss, epoch + 1)
            _write_metrics(writer, "train_epoch", train_result.metrics, epoch + 1)
            _write_metrics(writer, "valid", valid_result.metrics, epoch + 1)

            if valid_result.loss < best_valid:
                best_valid = valid_result.loss
                _save_model_state(transition_model, config, valid_result.metrics, best_path, dist_ctx)

        if dist_ctx.is_main_process:
            save_json(output_dir / "transition_history.json", history)
        return best_path
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()


def train_sft(
    model: SFTLanguageModel,
    loaders,
    config: ExperimentConfig,
) -> Path:
    dist_ctx = init_distributed(config.train.device, config.train.distributed_backend)
    device = dist_ctx.device
    configure_torch_runtime(device, config.train.allow_tf32)

    model.to(device)
    if dist_ctx.enabled:
        if dist_ctx.device.type == "cuda":
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=False,
                static_graph=config.train.ddp_static_graph,
            )
        else:
            model = DDP(model, find_unused_parameters=False)

    optimizer_kwargs = {
        "lr": config.train.learning_rate,
        "weight_decay": config.train.weight_decay,
    }
    if config.train.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    total_steps = config.train.epochs * _num_update_steps_per_epoch(
        len(loaders["train"]),
        config.train.grad_accum_steps,
    )
    scheduler = _build_scheduler(optimizer, config, total_steps)
    output_dir = ensure_dir(config.train.output_dir)
    best_path = output_dir / "sft_best.pt"
    valid_generation_dir = ensure_dir(output_dir / "sft_valid_generations")
    history: dict[str, list[dict[str, float]]] = {"train": [], "valid": []}
    valid_summary: list[dict[str, float | int | str]] = []
    best_valid = float("inf")
    writer = _build_summary_writer(config, "sft", dist_ctx)
    global_step = 0

    try:
        for epoch in range(config.train.epochs):
            if dist_ctx.enabled and hasattr(loaders["train"], "sampler") and hasattr(loaders["train"].sampler, "set_epoch"):
                loaders["train"].sampler.set_epoch(epoch)
            model.train()
            progress = tqdm(loaders["train"], desc=f"sft epoch {epoch + 1}/{config.train.epochs}", disable=not dist_ctx.is_main_process)
            running_loss = 0.0
            metric_sums: dict[str, float] = {}
            steps_in_epoch = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(progress, start=1):
                should_step = step % config.train.grad_accum_steps == 0 or step == len(loaders["train"])
                accum_steps = _current_accum_divisor(step, len(loaders["train"]), config.train.grad_accum_steps)
                loss, metrics = compute_sft_loss(model.module if isinstance(model, DDP) else model, batch, device, config.train.precision)
                (loss / accum_steps).backward()
                steps_in_epoch += 1
                running_loss += loss.item()

                if should_step:
                    _clip_grad_norm(model, config.train.grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                step_loss_value = distributed_mean(loss.item(), device) if dist_ctx.enabled else loss.item()
                if writer is not None:
                    writer.add_scalar("train/step_loss", step_loss_value, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                _write_metrics(writer, "train_step", metrics, global_step)
                for key, value in metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value

                if dist_ctx.is_main_process and (step % config.train.log_every == 0 or step == len(loaders["train"])):
                    progress.set_postfix(loss=running_loss / step, **_progress_metrics("sft", metrics))

            if dist_ctx.enabled:
                loss_tensor = torch.tensor([running_loss, steps_in_epoch], device=device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                running_loss = loss_tensor[0].item()
                steps_in_epoch = int(loss_tensor[1].item())
                metric_sums = distributed_sum_dict(metric_sums, device)

            averaged_train_metrics = {key: value / max(steps_in_epoch, 1) for key, value in metric_sums.items()}
            train_result = EpochResult(loss=running_loss / max(steps_in_epoch, 1), metrics=averaged_train_metrics)
            valid_result = evaluate_sft(model.module if isinstance(model, DDP) else model, loaders["valid"], config, device)
            if dist_ctx.is_main_process:
                history["train"].append({"loss": train_result.loss, **train_result.metrics})
                history["valid"].append({"loss": valid_result.loss, **valid_result.metrics})
                valid_generations = _collect_sft_valid_generations(
                    model.module if isinstance(model, DDP) else model,
                    loaders["valid"],
                    config,
                    device,
                    config.train.valid_generate_examples,
                )
                save_json(
                    valid_generation_dir / f"epoch_{epoch + 1:03d}.json",
                    {
                        "epoch": epoch + 1,
                        "experiment_name": config.experiment_name,
                        "valid_loss": valid_result.loss,
                        "valid_metrics": valid_result.metrics,
                        "samples": valid_generations,
                    },
                )
                valid_summary.append(_build_compact_valid_entry(epoch + 1, "sft", valid_result.loss, valid_result.metrics))
                save_json(
                    output_dir / "sft_valid_compact.json",
                    _build_compact_valid_report(config.experiment_name, "sft", valid_summary),
                )
            if dist_ctx.enabled:
                dist.barrier()
            if writer is not None:
                writer.add_scalar("train/epoch_loss", train_result.loss, epoch + 1)
                writer.add_scalar("valid/loss", valid_result.loss, epoch + 1)
            _write_metrics(writer, "train_epoch", train_result.metrics, epoch + 1)
            _write_metrics(writer, "valid", valid_result.metrics, epoch + 1)

            if valid_result.loss < best_valid:
                best_valid = valid_result.loss
                _save_model_state(model, config, valid_result.metrics, best_path, dist_ctx)

        if dist_ctx.is_main_process:
            save_json(output_dir / "sft_history.json", history)
        return best_path
    finally:
        if writer is not None:
            writer.close()
        cleanup_distributed()
