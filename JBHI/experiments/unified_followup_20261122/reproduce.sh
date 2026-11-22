#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
study_python="${CTD_PYTHON:-/tmp/ctd-conditional-clean-20261121/bin/python}"
"$study_python" registry_data.py
"$study_python" preflight.py
"$study_python" baseline.py --smoke
"$study_python" run.py --workers "${CTD_WORKERS:-6}"
"$study_python" baseline.py --workers "${CTD_WORKERS:-5}"
"$study_python" matched.py --workers "${CTD_WORKERS:-3}"
"$study_python" verify.py --workers "${CTD_WORKERS:-3}"
"$study_python" measurements.py
"$study_python" analyze.py
"$study_python" secondary.py
