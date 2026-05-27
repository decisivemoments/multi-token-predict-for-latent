from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel, GPT2Model

from mtp_latent.config import ModelConfig


class ReasoningCodec(nn.Module):
    def __init__(self, model_config: ModelConfig, pad_token_id: int) -> None:
        super().__init__()
        self.model_config = model_config
        self.pad_token_id = pad_token_id

        if model_config.model_name_or_path:
            self.encoder = GPT2Model.from_pretrained(model_config.model_name_or_path)
            self.decoder = GPT2LMHeadModel.from_pretrained(model_config.model_name_or_path)
        else:
            gpt2_config = HFGPT2Config(
                vocab_size=model_config.vocab_size,
                n_positions=model_config.n_positions,
                n_ctx=model_config.n_positions,
                n_embd=model_config.embedding_dim,
                n_layer=model_config.n_layer,
                n_head=model_config.n_head,
                resid_pdrop=model_config.dropout,
                embd_pdrop=model_config.dropout,
                attn_pdrop=model_config.dropout,
            )
            self.encoder = GPT2Model(gpt2_config)
            self.decoder = GPT2LMHeadModel(gpt2_config)

        hidden_size = self.encoder.config.hidden_size
        self.encoder.config.use_cache = False
        self.decoder.config.use_cache = False
        self.decoder.transformer.config.use_cache = False
        self.latent_proj = nn.Linear(hidden_size, model_config.latent_dim)
        self.decoder_latent_proj = nn.Linear(model_config.latent_dim, self.decoder.config.n_embd)
        self.future_token_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(self.decoder.config.n_embd, self.decoder.config.vocab_size, bias=False)
                for horizon in range(2, model_config.max_token_mtp_horizon + 1)
            }
        )

    def encode(self, prefix_ids: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=prefix_ids, attention_mask=prefix_mask.long())
        hidden_states = outputs.last_hidden_state
        last_indices = prefix_mask.long().sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices]
        return self.latent_proj(pooled)

    def decode(self, latent: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.decode_multi_horizon(latent, target_tokens, [1])[1]

    def decode_multi_horizon(
        self,
        latent: torch.Tensor,
        target_tokens: torch.Tensor,
        token_horizons: list[int],
    ) -> dict[int, torch.Tensor]:
        token_inputs = target_tokens[:, :-1]
        token_embeddings = self.decoder.transformer.wte(token_inputs)
        latent_prefix = self.decoder_latent_proj(latent).unsqueeze(1)
        inputs_embeds = torch.cat([latent_prefix, token_embeddings], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
        hidden_states = self.decoder.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state[:, 1:, :]

        logits_by_horizon: dict[int, torch.Tensor] = {}
        for horizon in token_horizons:
            if horizon == 1:
                logits_by_horizon[horizon] = self.decoder.lm_head(hidden_states)
            else:
                head = self.future_token_heads[str(horizon)]
                logits_by_horizon[horizon] = head(hidden_states)
        return logits_by_horizon
