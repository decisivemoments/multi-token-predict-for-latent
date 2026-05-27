from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from mtp_latent.models import LatentTransitionModel, ReasoningCodec


def rollout_continuous(
    codec: ReasoningCodec,
    transition: LatentTransitionModel,
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    steps: int,
) -> list[torch.Tensor]:
    latent = codec.encode(prefix_ids, prefix_mask)
    outputs = [latent]
    current = latent
    for _ in range(steps):
        current = transition(current)
        outputs.append(current)
    return outputs


def rollout_discretized(
    codec: ReasoningCodec,
    transition: LatentTransitionModel,
    prefix_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    steps: int,
    eos_id: int,
    max_step_tokens: int,
    text_separator: str = "\n",
) -> tuple[list[torch.Tensor], list[list[str]]]:
    encoded_prefix = tokenizer(prefix_texts, add_special_tokens=False, padding=True, return_tensors="pt")
    prefix_ids = encoded_prefix["input_ids"].to(next(codec.parameters()).device)
    prefix_mask = encoded_prefix["attention_mask"].bool().to(next(codec.parameters()).device)
    latent = codec.encode(prefix_ids, prefix_mask)
    outputs = [latent]
    current_prefix_texts = prefix_texts[:]
    generated_steps: list[list[str]] = []

    for _ in range(steps):
        predicted_latent = transition(outputs[-1])
        generated_token_ids, _ = codec.generate_step(
            predicted_latent,
            eos_token_id=eos_id,
            max_new_tokens=max_step_tokens,
        )
        step_texts: list[str] = []
        for row in range(generated_token_ids.size(0)):
            token_list = generated_token_ids[row].tolist()
            if eos_id in token_list:
                token_list = token_list[: token_list.index(eos_id)]
            step_texts.append(tokenizer.decode(token_list, skip_special_tokens=True).strip())
        generated_steps.append(step_texts)

        current_prefix_texts = [
            f"{prefix}{text_separator}{step}".strip() if step else prefix
            for prefix, step in zip(current_prefix_texts, step_texts)
        ]
        encoded_prefix = tokenizer(
            current_prefix_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=codec.encoder.config.n_positions,
            return_tensors="pt",
        )
        current_prefix = encoded_prefix["input_ids"].to(predicted_latent.device)
        current_mask = encoded_prefix["attention_mask"].bool().to(predicted_latent.device)
        outputs.append(codec.encode(current_prefix, current_mask))

    return outputs, generated_steps
