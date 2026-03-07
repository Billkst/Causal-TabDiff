#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET_METADATA_PATH="src/data/dataset_metadata_noleak.json"
export STASY_SAMPLE_CONT_JITTER="0.08"
echo "[NOLEAK] DATASET_METADATA_PATH=$DATASET_METADATA_PATH"
echo "[NOLEAK] STASY_SAMPLE_CONT_JITTER=$STASY_SAMPLE_CONT_JITTER"

bash scripts/run_stasy_with_audit.sh
