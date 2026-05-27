from __future__ import annotations

import math

import torch

def masked_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    predictions = logits.argmax(dim=-1)
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return correct.sum().item() / total


def cosine_retrieval_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    similarity = torch.nn.functional.cosine_similarity(
        predictions.unsqueeze(1), targets.unsqueeze(0), dim=-1
    )
    positive = similarity.diag()
    masked = similarity.clone()
    masked.fill_diagonal_(-math.inf)
    max_negative = masked.max(dim=1).values
    accuracy = (positive > max_negative).float().mean().item()
    return {
        "retrieval_acc": accuracy,
        "positive_score": positive.mean().item(),
        "max_negative_score": max_negative.mean().item(),
        "margin": (positive - max_negative).mean().item(),
    }
