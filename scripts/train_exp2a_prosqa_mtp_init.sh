#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
PYTHONPATH=src torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m mtp_latent.cli train-codec --config configs/exp2a_prosqa_mtp_init.yaml

