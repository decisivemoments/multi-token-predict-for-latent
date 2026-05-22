#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CODEC_CKPT="${1:-outputs/prosqa_a1_quick/codec_best.pt}"
PYTHONPATH=src python -m mtp_latent.cli train-transition \
  --config configs/prosqa_a1_quick.yaml \
  --codec-checkpoint "$CODEC_CKPT"

