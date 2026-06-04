#!/usr/bin/env bash
set -euo pipefail
NPROC_PER_NODE="${NPROC_PER_NODE:-1}" PYTHONPATH=src torchrun --nproc_per_node="${NPROC_PER_NODE}" -m mtp_latent.cli train-sft --config configs/sft_prosqa_next_step_ntp.yaml
