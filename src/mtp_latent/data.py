from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

from mtp_latent.config import DataConfig
from mtp_latent.utils import build_tokenizer


@dataclass
class TraceRecord:
    question: str
    steps: list[str]
    answer: str


@dataclass
class ReasoningSample:
    prefix_text: str
    future_steps: list[str]
    future_kinds: list[str]
    answer: str


@dataclass
class TransitionSample:
    question_text: str
    latent_prefix_texts: list[str]
    target_texts: list[str]
    target_kinds: list[str]
    answer: str


def load_reasoning_traces(
    path: str | Path,
    max_records: int | None = None,
    drop_empty_steps: bool = True,
) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        return records

    if raw_text[0] == "[":
        raw_records = json.loads(raw_text)
    else:
        raw_records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    for raw in raw_records[:max_records]:
        steps = [step.strip() for step in raw.get("steps", []) if isinstance(step, str)]
        if drop_empty_steps:
            steps = [step for step in steps if step]
        if not steps:
            continue
        records.append(
            TraceRecord(
                question=raw["question"].strip(),
                steps=steps,
                answer=str(raw.get("answer", "")).strip(),
            )
        )
    return records


def iter_reasoning_samples(
    records: Iterable[TraceRecord],
    max_horizon: int,
    text_separator: str,
) -> Iterable[ReasoningSample]:
    for record in records:
        targets = record.steps[:]
        target_kinds = ["step"] * len(record.steps)
        if record.answer:
            targets.append(record.answer)
            target_kinds.append("answer")
        for index in range(len(targets)):
            prefix_steps = record.steps[: min(index, len(record.steps))]
            prefix_parts = [record.question]
            if prefix_steps:
                prefix_parts.append(text_separator.join(prefix_steps))
            future_steps = targets[index : index + max_horizon]
            future_kinds = target_kinds[index : index + max_horizon]
            yield ReasoningSample(
                prefix_text=text_separator.join(prefix_parts).strip(),
                future_steps=future_steps,
                future_kinds=future_kinds,
                answer=record.answer,
            )


class ReasoningDataset(Dataset):
    def __init__(self, path: str | Path, data_config: DataConfig, tokenizer, split: str) -> None:
        self.data_config = data_config
        self.tokenizer = tokenizer
        max_records = {
            "train": data_config.train_max_records,
            "valid": data_config.valid_max_records,
            "test": data_config.test_max_records,
        }[split]
        self.records = load_reasoning_traces(
            path,
            max_records=max_records,
            drop_empty_steps=data_config.drop_empty_steps,
        )
        self.samples = list(
            iter_reasoning_samples(
                self.records,
                data_config.max_horizon,
                data_config.text_separator,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReasoningSample:
        return self.samples[index]

    def collate_fn(self, batch: list[ReasoningSample]) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
        prefix_batch = self.tokenizer(
            [sample.prefix_text for sample in batch],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.data_config.max_prefix_tokens,
            return_tensors="pt",
        )

        target_steps: list[torch.Tensor] = []
        target_labels: list[torch.Tensor] = []
        horizon_mask = torch.zeros((len(batch), self.data_config.max_horizon), dtype=torch.bool)

        for horizon in range(self.data_config.max_horizon):
            step_texts: list[str] = []
            step_kinds: list[str] = []
            for row, sample in enumerate(batch):
                if horizon < len(sample.future_steps):
                    horizon_mask[row, horizon] = True
                    step_texts.append(sample.future_steps[horizon])
                    step_kinds.append(sample.future_kinds[horizon])
                else:
                    step_texts.append("")
                    step_kinds.append("")

            encoded = self.tokenizer(
                step_texts,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=max(self.data_config.max_step_tokens - 1, 0),
                return_tensors="pt",
            )
            step_tokens = torch.full(
                (len(batch), encoded["input_ids"].size(1) + 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            step_labels = torch.full(
                (len(batch), encoded["input_ids"].size(1) + 1),
                -100,
                dtype=torch.long,
            )
            for row in range(len(batch)):
                if not horizon_mask[row, horizon]:
                    continue
                step_length = int(encoded["attention_mask"][row].sum().item())
                if step_length > 0:
                    step_tokens[row, :step_length] = encoded["input_ids"][row, :step_length]
                    step_labels[row, :step_length] = encoded["input_ids"][row, :step_length]
                step_tokens[row, step_length] = self.tokenizer.eos_token_id
                step_labels[row, step_length] = self.tokenizer.eos_token_id
            target_steps.append(step_tokens)
            target_labels.append(step_labels)

        return {
            "prefix_ids": prefix_batch["input_ids"],
            "prefix_mask": prefix_batch["attention_mask"].bool(),
            "target_steps": target_steps,
            "target_labels": target_labels,
            "horizon_mask": horizon_mask,
            "prefix_texts": [sample.prefix_text for sample in batch],
            "future_texts": [sample.future_steps for sample in batch],
            "future_kinds": [sample.future_kinds for sample in batch],
            "answers": [sample.answer for sample in batch],
        }


def build_transition_samples(
    records: Iterable[TraceRecord],
    data_config: DataConfig,
) -> list[TransitionSample]:
    samples: list[TransitionSample] = []
    for record in records:
        if not record.answer:
            continue
        latent_prefix_texts: list[str] = []
        target_texts = record.steps[:] + [record.answer]
        target_kinds = ["step"] * len(record.steps) + ["answer"]
        for index in range(len(record.steps)):
            prefix_steps = record.steps[:index]
            prefix_parts = [record.question]
            if prefix_steps:
                prefix_parts.append(data_config.text_separator.join(prefix_steps))
            prefix_text = data_config.text_separator.join(prefix_parts).strip()
            latent_prefix_texts.append(prefix_text)
        samples.append(
            TransitionSample(
                question_text=record.question,
                latent_prefix_texts=latent_prefix_texts,
                target_texts=target_texts,
                target_kinds=target_kinds,
                answer=record.answer,
            )
        )
    return samples


class TransitionDataset(Dataset):
    def __init__(self, samples: list[TransitionSample], data_config: DataConfig, tokenizer) -> None:
        self.samples = samples
        self.data_config = data_config
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]

    def collate_fn(self, batch: list[TransitionSample]) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
        question_batch = self.tokenizer(
            [sample.question_text for sample in batch],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.data_config.max_prefix_tokens,
            return_tensors="pt",
        )

        max_latents = max((len(sample.latent_prefix_texts) for sample in batch), default=0)
        latent_prefix_ids = torch.full(
            (len(batch), max_latents, 1),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        latent_prefix_mask = torch.zeros((len(batch), max_latents, 1), dtype=torch.bool)
        latent_mask = torch.zeros((len(batch), max_latents), dtype=torch.bool)

        max_targets = max((len(sample.target_texts) for sample in batch), default=0)
        target_type_labels = torch.full((len(batch), max_targets), -100, dtype=torch.long)
        supervision_mask = torch.zeros((len(batch), max_targets), dtype=torch.bool)

        target_text_matrix: list[list[str]] = []
        target_kind_matrix: list[list[str]] = []
        tokenized_targets: list[list[dict[str, torch.Tensor]]] = []
        max_target_token_length = 0
        flat_latent_prefixes: list[str] = []
        flat_latent_positions: list[tuple[int, int]] = []

        for row, sample in enumerate(batch):
            for latent_index, prefix_text in enumerate(sample.latent_prefix_texts):
                latent_mask[row, latent_index] = True
                flat_latent_prefixes.append(prefix_text)
                flat_latent_positions.append((row, latent_index))
            for target_index, target_kind in enumerate(sample.target_kinds):
                supervision_mask[row, target_index] = True
                target_type_labels[row, target_index] = 0 if target_kind == "step" else 1

            target_text_matrix.append(sample.target_texts)
            target_kind_matrix.append(sample.target_kinds)

            sample_tokens: list[dict[str, torch.Tensor]] = []
            for target_text in sample.target_texts:
                encoded = self.tokenizer(
                    target_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(self.data_config.max_step_tokens - 1, 0),
                    return_tensors="pt",
                )
                step_length = encoded["input_ids"].size(1)
                max_target_token_length = max(max_target_token_length, step_length + 1)
                sample_tokens.append(encoded)
            tokenized_targets.append(sample_tokens)

        if flat_latent_prefixes:
            encoded_prefixes = self.tokenizer(
                flat_latent_prefixes,
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.data_config.max_prefix_tokens,
                return_tensors="pt",
            )
            prefix_seq_len = encoded_prefixes["input_ids"].size(1)
            latent_prefix_ids = torch.full(
                (len(batch), max_latents, prefix_seq_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            latent_prefix_mask = torch.zeros((len(batch), max_latents, prefix_seq_len), dtype=torch.bool)
            for flat_index, (row, latent_index) in enumerate(flat_latent_positions):
                latent_prefix_ids[row, latent_index] = encoded_prefixes["input_ids"][flat_index]
                latent_prefix_mask[row, latent_index] = encoded_prefixes["attention_mask"][flat_index].bool()

        target_tokens = torch.full(
            (len(batch), max_targets, max_target_token_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        target_labels = torch.full(
            (len(batch), max_targets, max_target_token_length),
            -100,
            dtype=torch.long,
        )

        for row, sample_tokens in enumerate(tokenized_targets):
            for target_index, encoded in enumerate(sample_tokens):
                step_length = encoded["input_ids"].size(1)
                if step_length > 0:
                    token_ids = encoded["input_ids"][0, :step_length]
                    target_tokens[row, target_index, :step_length] = token_ids
                    target_labels[row, target_index, :step_length] = token_ids
                target_tokens[row, target_index, step_length] = self.tokenizer.eos_token_id
                target_labels[row, target_index, step_length] = self.tokenizer.eos_token_id

        return {
            "question_ids": question_batch["input_ids"],
            "question_mask": question_batch["attention_mask"].bool(),
            "latent_prefix_ids": latent_prefix_ids,
            "latent_prefix_mask": latent_prefix_mask,
            "latent_mask": latent_mask,
            "target_tokens": target_tokens,
            "target_labels": target_labels,
            "target_type_labels": target_type_labels,
            "supervision_mask": supervision_mask,
            "question_texts": [sample.question_text for sample in batch],
            "target_texts": target_text_matrix,
            "target_kinds": target_kind_matrix,
            "answers": [sample.answer for sample in batch],
        }


def build_dataloaders(
    data_config: DataConfig,
    tokenizer_name_or_path: str,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[ReasoningDataset, dict[str, DataLoader], object]:
    tokenizer = build_tokenizer(tokenizer_name_or_path)
    train_dataset = ReasoningDataset(data_config.train_path, data_config, tokenizer=tokenizer, split="train")
    valid_dataset = ReasoningDataset(data_config.valid_path, data_config, tokenizer=tokenizer, split="valid")
    test_dataset = ReasoningDataset(data_config.test_path, data_config, tokenizer=tokenizer, split="test")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    persistent_workers = data_config.persistent_workers and data_config.num_workers > 0
    loader_kwargs = {
        "num_workers": data_config.num_workers,
        "collate_fn": train_dataset.collate_fn,
        "pin_memory": data_config.pin_memory,
        "persistent_workers": persistent_workers,
    }
    if data_config.num_workers > 0 and data_config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = data_config.prefetch_factor

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=data_config.num_workers,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=data_config.pin_memory,
            persistent_workers=persistent_workers,
            **({"prefetch_factor": data_config.prefetch_factor} if data_config.num_workers > 0 and data_config.prefetch_factor is not None else {}),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=data_config.num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=data_config.pin_memory,
            persistent_workers=persistent_workers,
            **({"prefetch_factor": data_config.prefetch_factor} if data_config.num_workers > 0 and data_config.prefetch_factor is not None else {}),
        ),
    }
    return train_dataset, loaders, tokenizer


def build_transition_dataloaders(
    data_config: DataConfig,
    tokenizer_name_or_path: str,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[TransitionDataset, dict[str, DataLoader], object]:
    tokenizer = build_tokenizer(tokenizer_name_or_path)
    split_records = {
        "train": load_reasoning_traces(data_config.train_path, data_config.train_max_records, data_config.drop_empty_steps),
        "valid": load_reasoning_traces(data_config.valid_path, data_config.valid_max_records, data_config.drop_empty_steps),
        "test": load_reasoning_traces(data_config.test_path, data_config.test_max_records, data_config.drop_empty_steps),
    }
    split_datasets = {}
    for split, records in split_records.items():
        transition_samples = build_transition_samples(records, data_config)
        split_datasets[split] = TransitionDataset(transition_samples, data_config, tokenizer)

    train_sampler = DistributedSampler(split_datasets["train"], num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    valid_sampler = DistributedSampler(split_datasets["valid"], num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(split_datasets["test"], num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    persistent_workers = data_config.persistent_workers and data_config.num_workers > 0
    loader_kwargs = {
        "num_workers": data_config.num_workers,
        "collate_fn": split_datasets["train"].collate_fn,
        "pin_memory": data_config.pin_memory,
        "persistent_workers": persistent_workers,
    }
    if data_config.num_workers > 0 and data_config.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = data_config.prefetch_factor

    loaders = {
        "train": DataLoader(
            split_datasets["train"],
            batch_size=data_config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        ),
        "valid": DataLoader(
            split_datasets["valid"],
            batch_size=data_config.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=data_config.num_workers,
            collate_fn=split_datasets["valid"].collate_fn,
            pin_memory=data_config.pin_memory,
            persistent_workers=persistent_workers,
            **({"prefetch_factor": data_config.prefetch_factor} if data_config.num_workers > 0 and data_config.prefetch_factor is not None else {}),
        ),
        "test": DataLoader(
            split_datasets["test"],
            batch_size=data_config.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=data_config.num_workers,
            collate_fn=split_datasets["test"].collate_fn,
            pin_memory=data_config.pin_memory,
            persistent_workers=persistent_workers,
            **({"prefetch_factor": data_config.prefetch_factor} if data_config.num_workers > 0 and data_config.prefetch_factor is not None else {}),
        ),
    }
    return split_datasets["train"], loaders, tokenizer
