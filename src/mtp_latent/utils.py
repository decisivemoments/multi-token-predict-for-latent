from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer


PAD_ID = 0
EOS_ID = 2
UNK_ID = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_tokenizer(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested via train.device={device}, but no CUDA device is available.")
    return torch.device(device)


def configure_torch_runtime(device: torch.device, allow_tf32: bool) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True


class DistributedContext:
    def __init__(self, enabled: bool, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def init_distributed(device: str, backend: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if device == "auto":
            resolved_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        elif device.startswith("cuda"):
            resolved_device = torch.device(f"cuda:{local_rank}")
        else:
            resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)
        return DistributedContext(True, rank, local_rank, world_size, resolved_device)

    return DistributedContext(False, 0, 0, 1, resolve_device(device))


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_mean(value: float, device: torch.device) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()


def distributed_sum_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_available() or not dist.is_initialized():
        return metrics
    reduced: dict[str, float] = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor.item()
    return reduced
