#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/analyze_ranking_exp1_gsm_ntp.sh
bash scripts/analyze_ranking_exp1_gsm_mtp.sh
