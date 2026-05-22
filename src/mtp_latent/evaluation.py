from __future__ import annotations

import torch

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
    prefix_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    steps: int,
    bos_id: int,
) -> list[torch.Tensor]:
    latent = codec.encode(prefix_ids, prefix_mask)
    outputs = [latent]
    current_prefix = prefix_ids
    current_mask = prefix_mask

    for _ in range(steps):
        predicted_latent = transition(outputs[-1])
        decode_tokens = torch.full((predicted_latent.size(0), 2), bos_id, dtype=torch.long, device=predicted_latent.device)
        logits = codec.decode(predicted_latent, decode_tokens)
        next_token = logits.argmax(dim=-1)
        current_prefix = torch.cat([current_prefix, next_token], dim=1)
        if current_prefix.size(1) > codec.encoder.config.n_positions:
            current_prefix = current_prefix[:, -codec.encoder.config.n_positions :]
        current_mask = current_prefix != codec.pad_token_id
        outputs.append(codec.encode(current_prefix, current_mask))

    return outputs
