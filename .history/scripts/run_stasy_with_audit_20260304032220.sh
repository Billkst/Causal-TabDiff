#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

cd "$ROOT_DIR"

mkdir -p logs/evaluation

echo "[PIPELINE] $(date '+%F %T') Start STaSy pre-run gate check"
"$PYTHON_BIN" scripts/check_stasy_leakage.py \
	--fit_epochs 2 \
	--max_batches 4 \
	--batch_size 32 \
	--report_path logs/evaluation/stasy_precheck_report.md \
	--calibrate_fake_y \
	--fail_on_risk

echo "[PIPELINE] $(date '+%F %T') Gate passed, start STaSy full rerun"
"$PYTHON_BIN" run_baselines.py --model "STaSy (ICLR 23)"

echo "[PIPELINE] $(date '+%F %T') Start STaSy leakage quick check"
"$PYTHON_BIN" scripts/check_stasy_leakage.py \
	--fit_epochs 2 \
	--max_batches 4 \
	--batch_size 32 \
	--calibrate_fake_y \
	--report_path logs/evaluation/stasy_postcheck_report.md

echo "[PIPELINE] $(date '+%F %T') Start global baseline anomaly audit"
"$PYTHON_BIN" scripts/audit_baseline_report.py

echo "[PIPELINE] $(date '+%F %T') Completed"
