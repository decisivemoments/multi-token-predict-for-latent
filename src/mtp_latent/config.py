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
    max_horizon: int = 3
    batch_size: int = 8
    num_workers: int = 0
    text_separator: str = "\n"
    drop_empty_steps: bool = True


@dataclass
class ModelConfig:
    backbone_type: str = "gpt2"
    tokenizer_name_or_path: str = "gpt2"
    model_name_or_path: str | None = None
    vocab_size: int = 50257
    embedding_dim: int = 256
    hidden_dim: int = 256
    latent_dim: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 512
    dropout: float = 0.1
    init_source: str = "ntp"
    init_checkpoint: str | None = None


@dataclass
class CodecObjectiveConfig:
    name: str = "standard"
    horizon_weights: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    teacher_forcing: bool = True


@dataclass
class TransitionConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    init_source: str = "random"
    init_checkpoint: str | None = None


@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "outputs/default"
    log_every: int = 20
    tensorboard_dir: str | None = None


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
        return cls(
            experiment_name=raw["experiment_name"],
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            codec_objective=CodecObjectiveConfig(**raw.get("codec_objective", {})),
            transition=TransitionConfig(**raw.get("transition", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def dump_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "codec_objective": self.codec_objective.__dict__,
            "transition": self.transition.__dict__,
            "train": self.train.__dict__,
        }
