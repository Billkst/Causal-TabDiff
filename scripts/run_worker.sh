#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WORKER_ID="${1:?Usage: run_worker.sh <WORKER_ID> <GPU_ID> <RUN_LIST>}"
GPU_ID="${2:?Usage: run_worker.sh <WORKER_ID> <GPU_ID> <RUN_LIST>}"
shift 2
RUN_LIST=("$@")

OUTPUT_ROOT="outputs/submission_v3_5seeds/param_discussion_layer2"
LOG_ROOT="logs/5seeds_run"
CHECKPOINT_ROOT="checkpoints/5seeds_run"
SCRIPT="scripts/run_param_discussion_layer2_official.py"
BATCH_SIZE=94739
NUM_WORKERS=0

WORKER_LOG="$LOG_ROOT/worker_${WORKER_ID}.log"
PROGRESS_FILE="$LOG_ROOT/progress/worker_${WORKER_ID}_PROGRESS.json"

mkdir -p "$LOG_ROOT/progress"
mkdir -p "$CHECKPOINT_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

TOTAL=${#RUN_LIST[@]}
COMPLETED=()
FAILED=()

update_progress() {
    local run_id="$1"
    local status="$2"
    local extra="${3:-}"
    python3 -c "
import json, datetime
completed = $(python3 -c "import json; print(json.dumps([$(printf '\"w%s\",' "${COMPLETED[@]}" 2>/dev/null | sed 's/,$//' || echo '')]))")
failed = $(python3 -c "import json; print(json.dumps([$(printf '\"w%s\",' "${FAILED[@]}" 2>/dev/null | sed 's/,$//' || echo '')]))")
d = {
    'worker_id': '$WORKER_ID',
    'gpu_id': '$GPU_ID',
    'current_run': '$run_id',
    'status': '$status',
    'completed': completed,
    'failed': failed,
    'total': $TOTAL,
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
}
if '$extra':
    d['error'] = '$extra'
with open('$PROGRESS_FILE', 'w') as f:
    json.dump(d, f, indent=2)
"
}

is_completed() {
    local param="$1"
    local value="$2"
    local seed="$3"
    local npz="$OUTPUT_ROOT/$param/${param}_${value}_seed${seed}.npz"
    local eff="$OUTPUT_ROOT/$param/${param}_${value}_seed${seed}_efficiency.json"
    [[ -f "$npz" && -f "$eff" ]]
}

echo "====================================================" | tee -a "$WORKER_LOG"
echo "Worker $WORKER_ID | GPU $GPU_ID | $(date -u)" | tee -a "$WORKER_LOG"
echo "Runs: $TOTAL | bs=$BATCH_SIZE nw=$NUM_WORKERS" | tee -a "$WORKER_LOG"
echo "====================================================" | tee -a "$WORKER_LOG"

for entry in "${RUN_LIST[@]}"; do
    IFS='|' read -r PARAM VALUE SEED <<< "$entry"
    RUN_TAG="${PARAM}_${VALUE}_seed${SEED}"

    if is_completed "$PARAM" "$VALUE" "$SEED"; then
        echo "[W${WORKER_ID}] SKIP (already done): $RUN_TAG" | tee -a "$WORKER_LOG"
        COMPLETED+=("$RUN_TAG")
        continue
    fi

    echo "" | tee -a "$WORKER_LOG"
    echo "[W${WORKER_ID}] START: $RUN_TAG (${#COMPLETED[@]}/$TOTAL done)" | tee -a "$WORKER_LOG"
    echo "[W${WORKER_ID}] Time: $(date -u)" | tee -a "$WORKER_LOG"
    update_progress "$RUN_TAG" "running"

    CMD="python -u $SCRIPT --seed $SEED --param $PARAM --values $VALUE --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --output_root $OUTPUT_ROOT --log_root $LOG_ROOT --checkpoint_root $CHECKPOINT_ROOT --use_cached_dataset"
    echo "[W${WORKER_ID}] CMD: $CMD" | tee -a "$WORKER_LOG"

    EXIT_CODE=0
    $CMD 2>&1 | tee -a "$WORKER_LOG" || EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[W${WORKER_ID}] FAILED: $RUN_TAG exit_code=$EXIT_CODE" | tee -a "$WORKER_LOG"
        FAILED+=("$RUN_TAG")
        update_progress "$RUN_TAG" "FAILED" "exit_code=$EXIT_CODE"
        continue
    fi

    NPZ="$OUTPUT_ROOT/$PARAM/${PARAM}_${VALUE}_seed${SEED}.npz"
    EFF="$OUTPUT_ROOT/$PARAM/${PARAM}_${VALUE}_seed${SEED}_efficiency.json"

    if [[ ! -f "$NPZ" ]]; then
        echo "[W${WORKER_ID}] FAILED: $RUN_TAG npz missing: $NPZ" | tee -a "$WORKER_LOG"
        FAILED+=("$RUN_TAG")
        update_progress "$RUN_TAG" "FAILED" "npz_missing"
        continue
    fi
    if [[ ! -f "$EFF" ]]; then
        echo "[W${WORKER_ID}] FAILED: $RUN_TAG efficiency.json missing: $EFF" | tee -a "$WORKER_LOG"
        FAILED+=("$RUN_TAG")
        update_progress "$RUN_TAG" "FAILED" "eff_missing"
        continue
    fi

    COMPLETED+=("$RUN_TAG")
    update_progress "$RUN_TAG" "completed"
    echo "[W${WORKER_ID}] DONE: $RUN_TAG (${#COMPLETED[@]}/$TOTAL)" | tee -a "$WORKER_LOG"
done

echo "" | tee -a "$WORKER_LOG"
echo "====================================================" | tee -a "$WORKER_LOG"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Worker $WORKER_ID FINISHED WITH FAILURES: ${FAILED[*]}" | tee -a "$WORKER_LOG"
    update_progress "DONE_WITH_FAILURES" "partial" "failed_runs=${FAILED[*]}"
    exit 1
else
    echo "Worker $WORKER_ID ALL $TOTAL RUNS COMPLETED" | tee -a "$WORKER_LOG"
    update_progress "ALL" "ALL_COMPLETED"
fi
