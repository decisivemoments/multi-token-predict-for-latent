#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH=src python -m mtp_latent.cli analyze-latent-verifier --config configs/analysis_verifier_exp1_gsm_ntp.yaml
