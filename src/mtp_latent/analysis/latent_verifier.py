from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mtp_latent.analysis.candidate_ranking import (
    _build_ranking_targets,
    _conditional_candidate_scores,
    _load_codec,
    _sample_candidates,
    _summarize,
)
from mtp_latent.analysis.config import LatentVerifierConfig
from mtp_latent.config import ExperimentConfig
from mtp_latent.models import ReasoningCodec
from mtp_latent.utils import build_tokenizer, ensure_dir, save_json, set_seed


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [(score, label) for score, label in zip(scores, labels) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None

    sorted_pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_count = sum(1 for _, label in sorted_pairs[index:end] if label == 1)
        rank_sum += positive_count * average_rank
        index = end

    positive_count = len(positives)
    negative_count = len(negatives)
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _best_threshold_metrics(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    if not labels:
        return {"threshold": None, "accuracy": None, "precision": None, "recall": None, "f1": None}

    thresholds = sorted(set(scores))
    candidates = [thresholds[0] - 1e-6] + thresholds + [thresholds[-1] + 1e-6]
    best = {"threshold": candidates[0], "accuracy": -1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for threshold in candidates:
        tp = fp = tn = fn = 0
        for label, score in zip(labels, scores):
            predicted = 1 if score >= threshold else 0
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        accuracy = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if accuracy > best["accuracy"] or (accuracy == best["accuracy"] and f1 > best["f1"]):
            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def _confusion_by_type(labels: list[int], scores: list[float], candidate_types: list[str], threshold: float) -> dict[str, dict[str, float | int]]:
    by_type: dict[str, dict[str, int]] = {}
    for label, score, candidate_type in zip(labels, scores, candidate_types):
        bucket = by_type.setdefault(candidate_type, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            bucket["tp"] += 1
        elif predicted == 1 and label == 0:
            bucket["fp"] += 1
        elif predicted == 0 and label == 0:
            bucket["tn"] += 1
        else:
            bucket["fn"] += 1

    summary: dict[str, dict[str, float | int]] = {}
    for candidate_type, counts in sorted(by_type.items()):
        total = sum(counts.values())
        negative_total = counts["fp"] + counts["tn"]
        positive_total = counts["tp"] + counts["fn"]
        summary[candidate_type] = {
            **counts,
            "count": total,
            "false_positive_rate": counts["fp"] / max(negative_total, 1),
            "false_negative_rate": counts["fn"] / max(positive_total, 1),
        }
    return summary


def run_latent_verifier_analysis(config_path: str) -> Path:
    verifier_config = LatentVerifierConfig.from_yaml(config_path)
    experiment_config = ExperimentConfig.from_yaml(verifier_config.experiment_config_path)
    set_seed(verifier_config.seed)

    tokenizer_name = experiment_config.model.tokenizer_name_or_path or experiment_config.model.model_name_or_path
    tokenizer = build_tokenizer(tokenizer_name)
    experiment_config.model.vocab_size = len(tokenizer)
    codec = ReasoningCodec(experiment_config.model, pad_token_id=tokenizer.pad_token_id)
    _load_codec(codec, verifier_config.codec_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec.to(device)
    codec.eval()

    all_targets, targets_by_question = _build_ranking_targets(experiment_config, verifier_config)
    generator = torch.Generator().manual_seed(verifier_config.seed)
    sample_count = min(len(all_targets), verifier_config.max_samples)
    selected_indices = torch.randperm(len(all_targets), generator=generator).tolist()[:sample_count]
    selected_targets = [all_targets[index] for index in selected_indices]

    labels: list[int] = []
    scores: list[float] = []
    candidate_types: list[str] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    negative_scores_by_type: dict[str, list[float]] = {}
    examples: list[dict[str, Any]] = []

    for target in selected_targets:
        candidates = _sample_candidates(target, all_targets, targets_by_question, verifier_config, generator)
        candidate_scores = _conditional_candidate_scores(
            codec,
            tokenizer,
            target.prefix_text,
            [candidate["text"] for candidate in candidates],
            experiment_config,
            device,
        )

        for candidate, score in zip(candidates, candidate_scores):
            label = 1 if candidate["type"] == "gold" else 0
            labels.append(label)
            scores.append(float(score))
            candidate_types.append(candidate["type"])
            if label == 1:
                positive_scores.append(float(score))
            else:
                negative_scores.append(float(score))
                negative_scores_by_type.setdefault(candidate["type"], []).append(float(score))

        sorted_candidates = sorted(
            [
                {"text": candidate["text"], "type": candidate["type"], "score": float(score)}
                for candidate, score in zip(candidates, candidate_scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        gold_rank = 1 + sum(1 for candidate in sorted_candidates if candidate["type"] != "gold" and candidate["score"] > candidate_scores[0])
        if len(examples) < verifier_config.max_examples and (gold_rank > 1 or len(examples) < verifier_config.max_examples // 2):
            examples.append(
                {
                    "prefix_text": target.prefix_text,
                    "target_kind": target.target_kind,
                    "gold_text": target.target_text,
                    "gold_score": float(candidate_scores[0]),
                    "gold_rank": gold_rank,
                    "top_candidates": sorted_candidates[: min(5, len(sorted_candidates))],
                }
            )

    auc = _binary_auc(labels, scores)
    threshold_metrics = _best_threshold_metrics(labels, scores)
    threshold = threshold_metrics["threshold"]
    confusion = (
        _confusion_by_type(labels, scores, candidate_types, float(threshold))
        if threshold is not None
        else {}
    )

    results = {
        "experiment_name": verifier_config.experiment_name,
        "analysis_config_path": str(config_path),
        "experiment_config_path": verifier_config.experiment_config_path,
        "codec_checkpoint": verifier_config.codec_checkpoint,
        "split": verifier_config.split,
        "device": str(device),
        "implementation_contract": {
            "score": "score(z_prefix, candidate) = - mean teacher-forced token NLL under Decoder(z_prefix)",
            "positive": "gold current step or answer for the prefix",
            "negative": "hard perturbation, same-question, or cross-question non-gold candidate",
            "hard_negative": "perturb gold step/answer by changing result, operator, operand, or answer value",
            "decision_rule": "candidate is valid if score >= threshold",
            "threshold_selection": "post-hoc threshold maximizing accuracy on this analysis split",
        },
        "prefix_sample_count": sample_count,
        "pair_count": len(labels),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "auc": auc,
        "best_threshold": threshold_metrics,
        "score_summary": {
            "positive": _summarize(positive_scores),
            "negative": _summarize(negative_scores),
            "negative_by_type": {
                candidate_type: _summarize(type_scores)
                for candidate_type, type_scores in sorted(negative_scores_by_type.items())
            },
        },
        "confusion_by_type_at_best_threshold": confusion,
        "examples": examples,
    }

    output_dir = ensure_dir(verifier_config.output_dir)
    output_path = output_dir / f"{verifier_config.experiment_name}_{verifier_config.split}_latent_verifier.json"
    save_json(output_path, results)
    return output_path
