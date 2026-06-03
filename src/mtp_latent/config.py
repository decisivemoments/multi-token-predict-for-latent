from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_path: str
    valid_path: str
    test_path: str
    train_max_records: int | None = None
    valid_max_records: int | None = None
    test_max_records: int | None = None
    max_prefix_tokens: int = 256
    max_step_tokens: int = 64
    max_horizon: int = 1
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    text_separator: str = "\n"
    drop_empty_steps: bool = True


@dataclass
class ModelConfig:
    backbone_type: str = "gpt2"
    tokenizer_name_or_path: str = "gpt2"
    model_name_or_path: str | None = None
    vocab_size: int = 50257
    embedding_dim: int = 768
    hidden_dim: int = 768
    latent_dim: int = 768
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 512
    max_token_mtp_horizon: int = 1
    dropout: float = 0.1
    init_source: str = "ntp"
    init_checkpoint: str | None = None


@dataclass
class CodecObjectiveConfig:
    name: str = "standard"
    horizon_weights: list[float] = field(default_factory=lambda: [1.0])
    token_prediction_horizons: list[int] = field(default_factory=lambda: [1])
    token_prediction_weights: list[float] = field(default_factory=lambda: [1.0])
    teacher_forcing: bool = True


@dataclass
class TransitionConfig:
    hidden_dim: int = 768
    num_layers: int = 2
    n_head: int = 4
    dropout: float = 0.1
    init_source: str = "random"
    codec_checkpoint: str | None = None
    init_checkpoint: str | None = None
    type_loss_weight: float = 1.0
    decode_loss_weight: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    precision: str = "bf16"
    allow_tf32: bool = True
    output_dir: str = "outputs/default"
    log_every: int = 20
    tensorboard_dir: str | None = None
    valid_generate_examples: int = 8
    distributed_backend: str = "nccl"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1


@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    codec_objective: CodecObjectiveConfig = field(default_factory=CodecObjectiveConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls(
            experiment_name=raw["experiment_name"],
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            codec_objective=CodecObjectiveConfig(**raw.get("codec_objective", {})),
            transition=TransitionConfig(**raw.get("transition", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.codec_objective.horizon_weights) == 0:
            raise ValueError("codec_objective.horizon_weights must not be empty.")

        if self.codec_objective.name == "decoder_token_mtp":
            horizons = self.codec_objective.token_prediction_horizons
            weights = self.codec_objective.token_prediction_weights
            if not horizons:
                raise ValueError("decoder_token_mtp requires token_prediction_horizons.")
            if horizons[0] != 1:
                raise ValueError("decoder_token_mtp requires token_prediction_horizons to start from 1.")
            if any(horizon < 1 for horizon in horizons):
                raise ValueError("token_prediction_horizons must be positive integers.")
            if len(weights) < len(horizons):
                raise ValueError("token_prediction_weights must cover every token_prediction_horizon.")
            if max(horizons) > self.model.max_token_mtp_horizon:
                raise ValueError("model.max_token_mtp_horizon is smaller than codec_objective.token_prediction_horizons.")
            if self.data.max_horizon != 1:
                raise ValueError("decoder_token_mtp is only defined for data.max_horizon == 1 in experiment 2A.")

    def dump_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "codec_objective": self.codec_objective.__dict__,
            "transition": self.transition.__dict__,
            "train": self.train.__dict__,
        }
