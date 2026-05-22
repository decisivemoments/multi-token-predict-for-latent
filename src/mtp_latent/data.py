from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
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
        for index in range(len(record.steps)):
            prefix_steps = record.steps[:index]
            prefix_parts = [record.question]
            if prefix_steps:
                prefix_parts.append(text_separator.join(prefix_steps))
            future_steps = record.steps[index : index + max_horizon]
            yield ReasoningSample(
                prefix_text=text_separator.join(prefix_parts).strip(),
                future_steps=future_steps,
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
            padding=True,
            truncation=True,
            max_length=self.data_config.max_prefix_tokens,
            return_tensors="pt",
        )

        target_steps: list[torch.Tensor] = []
        horizon_mask = torch.zeros((len(batch), self.data_config.max_horizon), dtype=torch.bool)

        for horizon in range(self.data_config.max_horizon):
            step_texts: list[str] = []
            for row, sample in enumerate(batch):
                if horizon < len(sample.future_steps):
                    horizon_mask[row, horizon] = True
                    step_texts.append(sample.future_steps[horizon])
                else:
                    step_texts.append("")

            encoded = self.tokenizer(
                step_texts,
                padding=True,
                truncation=True,
                max_length=max(self.data_config.max_step_tokens - 1, 1),
                return_tensors="pt",
            )
            bos_column = torch.full(
                (len(batch), 1),
                self.tokenizer.bos_token_id,
                dtype=torch.long,
            )
            step_tokens = torch.cat([bos_column, encoded["input_ids"]], dim=1)
            for row in range(len(batch)):
                if not horizon_mask[row, horizon]:
                    step_tokens[row, 1:] = self.tokenizer.pad_token_id
            target_steps.append(step_tokens)

        return {
            "prefix_ids": prefix_batch["input_ids"],
            "prefix_mask": prefix_batch["attention_mask"].bool(),
            "target_steps": target_steps,
            "horizon_mask": horizon_mask,
            "prefix_texts": [sample.prefix_text for sample in batch],
            "future_texts": [sample.future_steps for sample in batch],
            "answers": [sample.answer for sample in batch],
        }


def build_dataloaders(data_config: DataConfig, tokenizer_name_or_path: str) -> tuple[ReasoningDataset, dict[str, DataLoader], object]:
    tokenizer = build_tokenizer(tokenizer_name_or_path)
    train_dataset = ReasoningDataset(data_config.train_path, data_config, tokenizer=tokenizer, split="train")
    valid_dataset = ReasoningDataset(data_config.valid_path, data_config, tokenizer=tokenizer, split="valid")
    test_dataset = ReasoningDataset(data_config.test_path, data_config, tokenizer=tokenizer, split="test")

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=data_config.batch_size,
            shuffle=True,
            num_workers=data_config.num_workers,
            collate_fn=train_dataset.collate_fn,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            collate_fn=valid_dataset.collate_fn,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            collate_fn=test_dataset.collate_fn,
        ),
    }
    return train_dataset, loaders, tokenizer
