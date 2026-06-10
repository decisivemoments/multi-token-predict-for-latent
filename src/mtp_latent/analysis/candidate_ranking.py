from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import re
import torch
import torch.nn.functional as F

from mtp_latent.analysis.config import CandidateRankingConfig
from mtp_latent.config import ExperimentConfig
from mtp_latent.data import load_reasoning_traces
from mtp_latent.models import ReasoningCodec
from mtp_latent.utils import build_tokenizer, ensure_dir, save_json, set_seed


@dataclass
class RankingTarget:
    prefix_text: str
    target_text: str
    target_kind: str
    question_index: int
    target_index: int


class CandidateSamplingConfig(Protocol):
    split: str
    seed: int
    max_samples: int
    random_negatives: int
    same_question_negatives: int
    include_answer_targets: bool
    max_examples: int


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)")


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision={precision}")


def _load_codec(codec: ReasoningCodec, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    codec.load_state_dict(state["model_state"])


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "p25": tensor.quantile(0.25).item(),
        "p50": tensor.quantile(0.50).item(),
        "p75": tensor.quantile(0.75).item(),
        "max": tensor.max().item(),
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _format_number_like(value: float, reference: str) -> str:
    if "." in reference:
        decimals = len(reference.split(".")[-1])
        return f"{value:.{decimals}f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _perturb_number_text(number_text: str, delta: float = 1.0) -> str | None:
    value = _parse_float(number_text)
    if value is None:
        return None
    if abs(value) < 1:
        new_value = value + 0.5
    else:
        new_value = value + delta
    if abs(new_value - value) < 1e-12:
        new_value = value + 1.0
    return _format_number_like(new_value, number_text)


def _replace_first_number(text: str, delta: float = 1.0) -> str | None:
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    replacement = _perturb_number_text(match.group(0), delta=delta)
    if replacement is None:
        return None
    return text[: match.start()] + replacement + text[match.end() :]


def _replace_last_number(text: str, delta: float = 1.0) -> str | None:
    matches = list(_NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    replacement = _perturb_number_text(match.group(0), delta=delta)
    if replacement is None:
        return None
    return text[: match.start()] + replacement + text[match.end() :]


def _replace_first_operator(expression: str) -> str | None:
    replacements = {"+": "-", "-": "+", "*": "/", "/": "*"}
    for index, char in enumerate(expression):
        if char in replacements:
            return expression[:index] + replacements[char] + expression[index + 1 :]
    return None


def _hard_negative_candidates(target_text: str, max_count: int) -> list[dict[str, str]]:
    if max_count <= 0:
        return []

    candidates: list[dict[str, str]] = []
    seen = {_normalize_text(target_text)}

    def add_candidate(text: str | None, candidate_type: str) -> None:
        if text is None:
            return
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            return
        candidates.append({"text": text, "type": candidate_type})
        seen.add(normalized)

    step_match = re.fullmatch(r"\s*<<(.+)=(.+)>>\s*", target_text)
    if step_match:
        expression = step_match.group(1)
        result = step_match.group(2)
        add_candidate(f"<<{expression}={_replace_last_number(result, delta=1.0) or result}>>", "hard_wrong_result")

        changed_operator_expression = _replace_first_operator(expression)
        add_candidate(
            None if changed_operator_expression is None else f"<<{changed_operator_expression}={result}>>",
            "hard_wrong_operator",
        )

        changed_number_expression = _replace_first_number(expression, delta=1.0)
        add_candidate(
            None if changed_number_expression is None else f"<<{changed_number_expression}={result}>>",
            "hard_wrong_operand",
        )

        changed_result_again = _replace_last_number(result, delta=-1.0)
        add_candidate(
            None if changed_result_again is None else f"<<{expression}={changed_result_again}>>",
            "hard_wrong_result",
        )
    else:
        add_candidate(_replace_last_number(target_text, delta=1.0), "hard_wrong_answer")
        add_candidate(_replace_last_number(target_text, delta=-1.0), "hard_wrong_answer")

    return candidates[:max_count]


def _tokenize_target_texts(tokenizer, texts: list[str], max_step_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max(max_step_tokens - 1, 0),
        return_tensors="pt",
    )
    target_tokens = torch.full(
        (len(texts), encoded["input_ids"].size(1) + 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    target_labels = torch.full_like(target_tokens, -100)
    for row in range(len(texts)):
        step_length = int(encoded["attention_mask"][row].sum().item())
        if step_length > 0:
            target_tokens[row, :step_length] = encoded["input_ids"][row, :step_length]
            target_labels[row, :step_length] = encoded["input_ids"][row, :step_length]
        target_tokens[row, step_length] = tokenizer.eos_token_id
        target_labels[row, step_length] = tokenizer.eos_token_id
    return target_tokens, target_labels


def _build_ranking_targets(
    experiment_config: ExperimentConfig,
    analysis_config: CandidateSamplingConfig,
) -> tuple[list[RankingTarget], list[list[RankingTarget]]]:
    split_path = {
        "train": experiment_config.data.train_path,
        "valid": experiment_config.data.valid_path,
        "test": experiment_config.data.test_path,
    }[analysis_config.split]
    split_max_records = {
        "train": experiment_config.data.train_max_records,
        "valid": experiment_config.data.valid_max_records,
        "test": experiment_config.data.test_max_records,
    }[analysis_config.split]
    records = load_reasoning_traces(
        split_path,
        max_records=split_max_records,
        drop_empty_steps=experiment_config.data.drop_empty_steps,
    )

    all_targets: list[RankingTarget] = []
    targets_by_question: list[list[RankingTarget]] = []
    for question_index, record in enumerate(records):
        target_texts = record.steps[:]
        target_kinds = ["step"] * len(record.steps)
        if analysis_config.include_answer_targets and record.answer:
            target_texts.append(record.answer)
            target_kinds.append("answer")

        question_targets: list[RankingTarget] = []
        for target_index, (target_text, target_kind) in enumerate(zip(target_texts, target_kinds)):
            prefix_steps = record.steps[: min(target_index, len(record.steps))]
            prefix_parts = [record.question]
            if prefix_steps:
                prefix_parts.append(experiment_config.data.text_separator.join(prefix_steps))
            target = RankingTarget(
                prefix_text=experiment_config.data.text_separator.join(prefix_parts).strip(),
                target_text=target_text,
                target_kind=target_kind,
                question_index=question_index,
                target_index=target_index,
            )
            question_targets.append(target)
            all_targets.append(target)
        targets_by_question.append(question_targets)
    return all_targets, targets_by_question


def _conditional_candidate_scores(
    codec: ReasoningCodec,
    tokenizer,
    prefix_text: str,
    candidate_texts: list[str],
    experiment_config: ExperimentConfig,
    device: torch.device,
) -> list[float]:
    autocast_dtype = _autocast_dtype(experiment_config.train.precision) if device.type == "cuda" else None
    prefix_batch = tokenizer(
        [prefix_text],
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=experiment_config.data.max_prefix_tokens,
        return_tensors="pt",
    )
    target_tokens, target_labels = _tokenize_target_texts(
        tokenizer,
        candidate_texts,
        experiment_config.data.max_step_tokens,
    )

    prefix_ids = prefix_batch["input_ids"].to(device, non_blocking=True)
    prefix_mask = prefix_batch["attention_mask"].bool().to(device, non_blocking=True)
    target_tokens = target_tokens.to(device, non_blocking=True)
    target_labels = target_labels.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            latent = codec.encode(prefix_ids, prefix_mask)
            repeated_latent = latent.expand(len(candidate_texts), -1)
            logits = codec.decode(repeated_latent, target_tokens)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        target_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(target_labels.shape)
    token_mask = target_labels.ne(-100)
    token_counts = token_mask.sum(dim=1).clamp_min(1)
    mean_nll = (token_loss * token_mask).sum(dim=1) / token_counts
    return (-mean_nll).detach().cpu().tolist()


def _sample_candidates(
    target: RankingTarget,
    all_targets: list[RankingTarget],
    targets_by_question: list[list[RankingTarget]],
    analysis_config: CandidateSamplingConfig,
    generator: torch.Generator,
) -> list[dict[str, str]]:
    candidates = [{"text": target.target_text, "type": "gold"}]
    seen_texts = {_normalize_text(target.target_text)}

    hard_negative_count = int(getattr(analysis_config, "hard_negatives", 0))
    for candidate in _hard_negative_candidates(target.target_text, hard_negative_count):
        normalized = _normalize_text(candidate["text"])
        if normalized in seen_texts:
            continue
        candidates.append(candidate)
        seen_texts.add(normalized)

    same_question_pool = [
        candidate
        for candidate in targets_by_question[target.question_index]
        if candidate.target_index != target.target_index
        and _normalize_text(candidate.target_text) not in seen_texts
    ]
    same_question_order = torch.randperm(len(same_question_pool), generator=generator).tolist()
    for index in same_question_order[: analysis_config.same_question_negatives]:
        candidate = same_question_pool[index]
        candidates.append({"text": candidate.target_text, "type": "same_question"})
        seen_texts.add(_normalize_text(candidate.target_text))

    random_order = torch.randperm(len(all_targets), generator=generator).tolist()
    random_added = 0
    for index in random_order:
        candidate = all_targets[index]
        normalized = _normalize_text(candidate.target_text)
        if candidate.question_index == target.question_index or normalized in seen_texts:
            continue
        candidates.append({"text": candidate.target_text, "type": "cross_question"})
        seen_texts.add(normalized)
        random_added += 1
        if random_added >= analysis_config.random_negatives:
            break

    return candidates


def run_candidate_ranking_analysis(config_path: str) -> Path:
    analysis_config = CandidateRankingConfig.from_yaml(config_path)
    experiment_config = ExperimentConfig.from_yaml(analysis_config.experiment_config_path)
    set_seed(analysis_config.seed)

    tokenizer_name = experiment_config.model.tokenizer_name_or_path or experiment_config.model.model_name_or_path
    tokenizer = build_tokenizer(tokenizer_name)
    experiment_config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(experiment_config.model, pad_token_id=tokenizer.pad_token_id)
    _load_codec(codec, analysis_config.codec_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec.to(device)
    codec.eval()

    all_targets, targets_by_question = _build_ranking_targets(experiment_config, analysis_config)
    if not all_targets:
        results = {
            "experiment_name": analysis_config.experiment_name,
            "sample_count": 0,
            "reason": "No ranking targets were built.",
        }
        output_dir = ensure_dir(analysis_config.output_dir)
        output_path = output_dir / f"{analysis_config.experiment_name}_{analysis_config.split}_candidate_ranking.json"
        save_json(output_path, results)
        return output_path

    generator = torch.Generator().manual_seed(analysis_config.seed)
    sample_count = min(len(all_targets), analysis_config.max_samples)
    selected_indices = torch.randperm(len(all_targets), generator=generator).tolist()[:sample_count]
    selected_targets = [all_targets[index] for index in selected_indices]

    top1_correct = 0.0
    mrr_sum = 0.0
    candidate_counts: list[float] = []
    gold_scores: list[float] = []
    max_negative_scores: list[float] = []
    margins: list[float] = []
    negative_type_margins: dict[str, list[float]] = {}
    negative_type_best_count: dict[str, int] = {}
    negative_type_count: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for sample_position, target in enumerate(selected_targets):
        candidates = _sample_candidates(target, all_targets, targets_by_question, analysis_config, generator)
        scores = _conditional_candidate_scores(
            codec,
            tokenizer,
            target.prefix_text,
            [candidate["text"] for candidate in candidates],
            experiment_config,
            device,
        )
        gold_score = scores[0]
        negative_scores = scores[1:]
        rank = 1 + sum(1 for score in negative_scores if score > gold_score)
        top1_correct += 1.0 if rank == 1 else 0.0
        mrr_sum += 1.0 / rank
        candidate_counts.append(float(len(candidates)))
        gold_scores.append(float(gold_score))

        max_negative_score = None
        best_negative_type = None
        if negative_scores:
            max_negative_score = max(negative_scores)
            max_negative_index = 1 + negative_scores.index(max_negative_score)
            best_negative_type = candidates[max_negative_index]["type"]
            max_negative_scores.append(float(max_negative_score))
            margins.append(float(gold_score - max_negative_score))
            negative_type_best_count[best_negative_type] = negative_type_best_count.get(best_negative_type, 0) + 1

        for candidate, score in zip(candidates[1:], negative_scores):
            negative_type = candidate["type"]
            negative_type_count[negative_type] = negative_type_count.get(negative_type, 0) + 1
            negative_type_margins.setdefault(negative_type, []).append(float(gold_score - score))

        should_log_example = rank > 1 or sample_position < max(analysis_config.max_examples // 2, 0)
        if len(examples) < analysis_config.max_examples and should_log_example:
            sorted_candidates = sorted(
                [
                    {
                        "text": candidate["text"],
                        "type": candidate["type"],
                        "score": float(score),
                    }
                    for candidate, score in zip(candidates, scores)
                ],
                key=lambda item: item["score"],
                reverse=True,
            )
            examples.append(
                {
                    "prefix_text": target.prefix_text,
                    "target_kind": target.target_kind,
                    "gold_text": target.target_text,
                    "gold_score": float(gold_score),
                    "rank": rank,
                    "candidate_count": len(candidates),
                    "best_negative_type": best_negative_type,
                    "best_negative_score": None if max_negative_score is None else float(max_negative_score),
                    "top_candidates": sorted_candidates[: min(5, len(sorted_candidates))],
                }
            )

    negative_type_summary = {
        negative_type: {
            "count": negative_type_count.get(negative_type, 0),
            "best_negative_count": negative_type_best_count.get(negative_type, 0),
            "gold_minus_negative_margin": _summarize(type_margins),
        }
        for negative_type, type_margins in sorted(negative_type_margins.items())
    }

    results = {
        "experiment_name": analysis_config.experiment_name,
        "analysis_config_path": str(config_path),
        "experiment_config_path": analysis_config.experiment_config_path,
        "codec_checkpoint": analysis_config.codec_checkpoint,
        "split": analysis_config.split,
        "device": str(device),
        "implementation_contract": {
            "score": "score(z_prefix, candidate) = - mean teacher-forced token NLL under Decoder(z_prefix)",
            "gold_candidate": "current step or answer for the prefix",
            "same_question_negative": "other step/answer from the same reasoning trace",
            "cross_question_negative": "step/answer from another reasoning trace",
            "rank_direction": "higher score is better",
        },
        "sample_count": sample_count,
        "target_pool_size": len(all_targets),
        "top1_accuracy": top1_correct / max(sample_count, 1),
        "mrr": mrr_sum / max(sample_count, 1),
        "candidate_count": _summarize(candidate_counts),
        "gold_score": _summarize(gold_scores),
        "max_negative_score": _summarize(max_negative_scores),
        "gold_minus_max_negative_margin": _summarize(margins),
        "negative_type_summary": negative_type_summary,
        "examples": examples,
    }

    output_dir = ensure_dir(analysis_config.output_dir)
    output_path = output_dir / f"{analysis_config.experiment_name}_{analysis_config.split}_candidate_ranking.json"
    save_json(output_path, results)
    return output_path
