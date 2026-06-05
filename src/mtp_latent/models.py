from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Config as HFGPT2Config
from transformers import GPT2LMHeadModel, GPT2Model

from mtp_latent.config import ModelConfig, TransitionConfig


def _from_pretrained(model_cls, model_config: ModelConfig):
    kwargs = {}
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation
    try:
        return model_cls.from_pretrained(model_config.model_name_or_path, **kwargs)
    except (TypeError, ValueError) as error:
        if "attn" not in str(error).lower():
            raise
    return model_cls.from_pretrained(model_config.model_name_or_path)


def _maybe_identity_init(linear: nn.Linear) -> None:
    if linear.in_features != linear.out_features:
        return
    with torch.no_grad():
        nn.init.eye_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


def _zero_init(linear: nn.Linear) -> None:
    with torch.no_grad():
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


class ReasoningCodec(nn.Module):
    def __init__(self, model_config: ModelConfig, pad_token_id: int) -> None:
        super().__init__()
        self.model_config = model_config
        self.pad_token_id = pad_token_id

        if model_config.model_name_or_path:
            self.encoder = _from_pretrained(GPT2Model, model_config)
            self.decoder = _from_pretrained(GPT2LMHeadModel, model_config)
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
                attn_implementation=model_config.attn_implementation,
            )
            self.encoder = GPT2Model(gpt2_config)
            self.decoder = GPT2LMHeadModel(gpt2_config)

        hidden_size = self.encoder.config.hidden_size
        self.encoder.config.use_cache = False
        self.decoder.config.use_cache = False
        self.decoder.transformer.config.use_cache = False
        self.latent_proj = nn.Linear(hidden_size, model_config.latent_dim)
        self.decoder_latent_proj = nn.Linear(model_config.latent_dim, self.decoder.config.n_embd)
        _maybe_identity_init(self.latent_proj)
        _maybe_identity_init(self.decoder_latent_proj)
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

    def _decode_hidden_states(self, latent: torch.Tensor, token_inputs: torch.Tensor) -> torch.Tensor:
        latent = latent.to(self.decoder_latent_proj.weight.dtype)
        token_embeddings = self.decoder.transformer.wte(token_inputs)
        latent_prefix = self.decoder_latent_proj(latent).unsqueeze(1)
        inputs_embeds = torch.cat([latent_prefix, token_embeddings], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
        return self.decoder.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

    def decode_multi_horizon(
        self,
        latent: torch.Tensor,
        target_tokens: torch.Tensor,
        token_horizons: list[int],
    ) -> dict[int, torch.Tensor]:
        token_inputs = target_tokens[:, :-1]
        hidden_states = self._decode_hidden_states(latent, token_inputs)

        logits_by_horizon: dict[int, torch.Tensor] = {}
        for horizon in token_horizons:
            if horizon == 1:
                logits_by_horizon[horizon] = self.decoder.lm_head(hidden_states)
            else:
                head = self.future_token_heads[str(horizon)]
                logits_by_horizon[horizon] = head(hidden_states)
        return logits_by_horizon

    def next_token_logits(self, latent: torch.Tensor, decode_tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self._decode_hidden_states(latent, decode_tokens)
        return self.decoder.lm_head(hidden_states[:, -1, :])

    def generate_step(
        self,
        latent: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent.size(0)
        decode_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=latent.device)
        generated_tokens: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=latent.device)

        for _ in range(max_new_tokens):
            next_logits = self.next_token_logits(latent, decode_tokens)
            next_token = next_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated_tokens.append(next_token)
            decode_tokens = torch.cat([decode_tokens, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

        if generated_tokens:
            generated = torch.stack(generated_tokens, dim=1)
        else:
            generated = torch.empty((batch_size, 0), dtype=torch.long, device=latent.device)
        return generated, finished


class SFTLanguageModel(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        if model_config.model_name_or_path:
            self.model = _from_pretrained(GPT2LMHeadModel, model_config)
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
                attn_implementation=model_config.attn_implementation,
            )
            self.model = GPT2LMHeadModel(gpt2_config)
        self.model.config.use_cache = False
        self.model.transformer.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask.long()).logits

    def generate_continuation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated = input_ids
        generated_mask = attention_mask.long()
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        new_tokens: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            logits = self.model(input_ids=generated, attention_mask=generated_mask).logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            new_tokens.append(next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            generated_mask = torch.cat(
                [generated_mask, torch.ones((generated_mask.size(0), 1), device=generated_mask.device, dtype=generated_mask.dtype)],
                dim=1,
            )
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

        if new_tokens:
            return torch.stack(new_tokens, dim=1), finished
        return torch.empty((input_ids.size(0), 0), dtype=torch.long, device=input_ids.device), finished


class LatentTransitionModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = latent_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class ReasoningTransitionModel(nn.Module):
    def __init__(self, model_config: ModelConfig, transition_config: TransitionConfig, vocab_size: int) -> None:
        super().__init__()
        if model_config.model_name_or_path:
            self.backbone = _from_pretrained(GPT2Model, model_config)
            transition_hidden_dim = self.backbone.config.n_embd
        else:
            if transition_config.hidden_dim % transition_config.n_head != 0:
                raise ValueError("transition.hidden_dim must be divisible by transition.n_head")

            backbone_config = HFGPT2Config(
                vocab_size=vocab_size,
                n_positions=model_config.n_positions,
                n_ctx=model_config.n_positions,
                n_embd=transition_config.hidden_dim,
                n_layer=transition_config.num_layers,
                n_head=transition_config.n_head,
                resid_pdrop=transition_config.dropout,
                embd_pdrop=transition_config.dropout,
                attn_pdrop=transition_config.dropout,
                attn_implementation=model_config.attn_implementation,
            )
            self.backbone = GPT2Model(backbone_config)
            transition_hidden_dim = transition_config.hidden_dim

        self.backbone.config.use_cache = False
        self.latent_in_proj = nn.Linear(model_config.latent_dim, transition_hidden_dim)
        self.latent_out_proj = nn.Linear(transition_hidden_dim, model_config.latent_dim)
        self.next_type_head = nn.Linear(transition_hidden_dim, 2)
        _maybe_identity_init(self.latent_in_proj)
        _maybe_identity_init(self.latent_out_proj)

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        latent_inputs: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_embeddings = self.backbone.wte(question_ids)
        latent_embeddings = self.latent_in_proj(latent_inputs)
        inputs_embeds = torch.cat([question_embeddings, latent_embeddings], dim=1)
        attention_mask = torch.cat([question_mask.long(), latent_mask.long()], dim=1)
        hidden_states = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state

        batch_size, max_latents = latent_mask.size()
        question_lengths = question_mask.long().sum(dim=1)
        device = hidden_states.device
        supervision_indices = torch.zeros((batch_size, max_latents + 1), dtype=torch.long, device=device)
        supervision_indices[:, 0] = torch.clamp(question_lengths - 1, min=0)
        if max_latents > 0:
            latent_offsets = torch.arange(max_latents, device=device).unsqueeze(0).expand(batch_size, -1)
            supervision_indices[:, 1:] = question_lengths.unsqueeze(1) + latent_offsets

        gather_index = supervision_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        supervision_hidden_states = hidden_states.gather(1, gather_index)
        next_type_logits = self.next_type_head(supervision_hidden_states)
        predicted_latents = self.latent_out_proj(supervision_hidden_states)
        supervision_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                latent_mask,
            ],
            dim=1,
        )
        return next_type_logits, predicted_latents, supervision_mask
