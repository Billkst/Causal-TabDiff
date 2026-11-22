#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
py="${CTD_PYTHON:-python3}"
mkdir -p logs private statistics
run() {
    local label="$1"
    shift
    /usr/bin/time -f 'elapsed_seconds=%e user_cpu_seconds=%U sys_cpu_seconds=%S max_rss_kb=%M exit=%x' -o "logs/${label}-time.txt" "$py" -Bu "$@" 2>&1 | tee "logs/${label}.log"
}
run preflight sensitivity.py preflight
if [[ -s logs/fits-time.txt ]] && [[ "$("$py" -B -c "from pathlib import Path; print(len(list(Path('private/fits').glob('dev*/*/*/evaluation.json'))))")" == 54 ]]; then
    "$py" -Bu sensitivity.py train --workers "${CTD_WORKERS:-7}"
else
    run fits sensitivity.py train --workers "${CTD_WORKERS:-7}"
fi
run pairs sensitivity.py summarize
run cases cases.py select
run local-render cases.py render-local
run benchmark benchmark.py
run verification verify_audit.py
run report report.py
