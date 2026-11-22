#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
py="${CTD_PYTHON:-python3}"
workers="${CTD_WORKERS:-5}"
"$py" checks.py
"$py" run.py screen --workers "$workers"
"$py" run.py lock
"$py" run.py final --workers "$workers"
"$py" evidence.py
