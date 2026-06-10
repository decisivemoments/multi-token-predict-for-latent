from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class CandidateRankingConfig:
    experiment_name: str
    experiment_config_path: str
    codec_checkpoint: str
    split: str = "valid"
    output_dir: str = "outputs/analysis/candidate_ranking"
    seed: int = 42
    max_samples: int = 256
    random_negatives: int = 8
    same_question_negatives: int = 2
    include_answer_targets: bool = True
    max_examples: int = 24

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CandidateRankingConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls(
            experiment_name=raw["experiment_name"],
            experiment_config_path=raw["experiment_config_path"],
            codec_checkpoint=raw["codec_checkpoint"],
            split=raw.get("split", "valid"),
            output_dir=raw.get("output_dir", "outputs/analysis/candidate_ranking"),
            seed=raw.get("seed", 42),
            max_samples=raw.get("max_samples", 256),
            random_negatives=raw.get("random_negatives", 8),
            same_question_negatives=raw.get("same_question_negatives", 2),
            include_answer_targets=raw.get("include_answer_targets", True),
            max_examples=raw.get("max_examples", 24),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: train, valid, test.")
        if self.max_samples < 1:
            raise ValueError("max_samples must be >= 1.")
        if self.random_negatives < 0:
            raise ValueError("random_negatives must be >= 0.")
        if self.same_question_negatives < 0:
            raise ValueError("same_question_negatives must be >= 0.")
        if self.random_negatives + self.same_question_negatives < 1:
            raise ValueError("At least one negative candidate is required.")
        if self.max_examples < 0:
            raise ValueError("max_examples must be >= 0.")


@dataclass
class LatentVerifierConfig:
    experiment_name: str
    experiment_config_path: str
    codec_checkpoint: str
    split: str = "valid"
    output_dir: str = "outputs/analysis/latent_verifier"
    seed: int = 42
    max_samples: int = 256
    random_negatives: int = 4
    same_question_negatives: int = 2
    hard_negatives: int = 4
    include_answer_targets: bool = True
    max_examples: int = 32

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LatentVerifierConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls(
            experiment_name=raw["experiment_name"],
            experiment_config_path=raw["experiment_config_path"],
            codec_checkpoint=raw["codec_checkpoint"],
            split=raw.get("split", "valid"),
            output_dir=raw.get("output_dir", "outputs/analysis/latent_verifier"),
            seed=raw.get("seed", 42),
            max_samples=raw.get("max_samples", 256),
            random_negatives=raw.get("random_negatives", 4),
            same_question_negatives=raw.get("same_question_negatives", 2),
            hard_negatives=raw.get("hard_negatives", 4),
            include_answer_targets=raw.get("include_answer_targets", True),
            max_examples=raw.get("max_examples", 32),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("split must be one of: train, valid, test.")
        if self.max_samples < 1:
            raise ValueError("max_samples must be >= 1.")
        if self.random_negatives < 0:
            raise ValueError("random_negatives must be >= 0.")
        if self.same_question_negatives < 0:
            raise ValueError("same_question_negatives must be >= 0.")
        if self.hard_negatives < 0:
            raise ValueError("hard_negatives must be >= 0.")
        if self.random_negatives + self.same_question_negatives + self.hard_negatives < 1:
            raise ValueError("At least one negative candidate is required.")
        if self.max_examples < 0:
            raise ValueError("max_examples must be >= 0.")
