#!/usr/bin/env bash
# =============================================================================
# Param Discussion v3 Clean Launcher
# 14 real training runs, fail-fast, PROGRESS.json, --resume support
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="outputs/submission_v3/param_discussion_layer2"
LOG_ROOT="logs/param_discussion_v3_clean"
CHECKPOINT_ROOT="checkpoints/param_discussion_v3_clean"
PROGRESS_FILE="$LOG_ROOT/PROGRESS.json"
SCRIPT="scripts/run_param_discussion_layer2_official.py"
SEED=42
RESUME=false

if [[ "${1:-}" == "--resume" ]]; then
    RESUME=true
    echo "[LAUNCHER] Resume mode enabled" | tee -a "$LOG_ROOT/launcher.log"
fi

# --- Run definitions: P0-P13 ---
# Format: "RUN_ID|PARAM|VALUE"
RUNS=(
    "P0|focal_gamma|2.0"
    "P1|focal_gamma|0.0"
    "P2|focal_gamma|1.0"
    "P3|focal_gamma|3.0"
    "P4|focal_gamma|4.0"
    "P5|oversample_ratio|1.0"
    "P6|oversample_ratio|2.5"
    "P7|oversample_ratio|7.5"
    "P8|oversample_ratio|10.0"
    "P9|disc_weight|0.0"
    "P10|disc_weight|0.01"
    "P11|disc_weight|0.05"
    "P12|disc_weight|0.1"
    "P13|disc_weight|0.2"
)

TOTAL=${#RUNS[@]}
COMPLETED=()

# --- Helper: update PROGRESS.json ---
update_progress() {
    local run_id="$1"
    local status="$2"
    local extra="${3:-}"
    local completed_str=""
    if [[ ${#COMPLETED[@]} -gt 0 ]]; then
        completed_str=$(printf '"%s",' "${COMPLETED[@]}" | sed 's/,$//')
    fi
    python3 -c "
import json, datetime
d = {'current_run': '$run_id', 'status': '$status', 'completed': [${completed_str}], 'total': $TOTAL, 'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'}
if '$extra': d['error'] = '$extra'
with open('$PROGRESS_FILE', 'w') as f: json.dump(d, f, indent=2)
"
}

# --- Helper: check if run already completed (for --resume) ---
is_completed() {
    local param="$1"
    local value="$2"
    local npz="$OUTPUT_ROOT/$param/${param}_${value}_seed${SEED}.npz"
    local eff="$OUTPUT_ROOT/$param/${param}_${value}_seed${SEED}_efficiency.json"
    [[ -f "$npz" && -f "$eff" ]]
}

# --- Pre-flight ---
echo "============================================" | tee -a "$LOG_ROOT/launcher.log"
echo "Param Discussion v3 Clean Launcher" | tee -a "$LOG_ROOT/launcher.log"
echo "Time: $(date -u)" | tee -a "$LOG_ROOT/launcher.log"
echo "Total runs: $TOTAL" | tee -a "$LOG_ROOT/launcher.log"
echo "Resume: $RESUME" | tee -a "$LOG_ROOT/launcher.log"
echo "Output: $OUTPUT_ROOT" | tee -a "$LOG_ROOT/launcher.log"
echo "Logs: $LOG_ROOT" | tee -a "$LOG_ROOT/launcher.log"
echo "============================================" | tee -a "$LOG_ROOT/launcher.log"

# --- Main loop ---
for entry in "${RUNS[@]}"; do
    IFS='|' read -r RUN_ID PARAM VALUE <<< "$entry"

    # Resume: skip completed runs
    if $RESUME && is_completed "$PARAM" "$VALUE"; then
        echo "[LAUNCHER] $RUN_ID ($PARAM=$VALUE) already completed, skipping" | tee -a "$LOG_ROOT/launcher.log"
        COMPLETED+=("$RUN_ID")
        continue
    fi

    echo "" | tee -a "$LOG_ROOT/launcher.log"
    echo "[LAUNCHER] Starting $RUN_ID: $PARAM=$VALUE (${#COMPLETED[@]}/$TOTAL done)" | tee -a "$LOG_ROOT/launcher.log"
    echo "[LAUNCHER] Time: $(date -u)" | tee -a "$LOG_ROOT/launcher.log"

    update_progress "$RUN_ID" "started"

    # Execute training run
    CMD="python -u $SCRIPT --seed $SEED --param $PARAM --values $VALUE --output_root $OUTPUT_ROOT --log_root $LOG_ROOT --checkpoint_root $CHECKPOINT_ROOT --use_cached_dataset"
    echo "[LAUNCHER] CMD: $CMD" | tee -a "$LOG_ROOT/launcher.log"

    EXIT_CODE=0
    $CMD || EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[LAUNCHER] FATAL: $RUN_ID FAILED with exit code $EXIT_CODE" | tee -a "$LOG_ROOT/launcher.log"
        update_progress "$RUN_ID" "FAILED" "exit_code=$EXIT_CODE"
        echo "[LAUNCHER] Stopping. Fix the issue and re-run with --resume" | tee -a "$LOG_ROOT/launcher.log"
        exit 1
    fi

    # Verify outputs exist
    NPZ="$OUTPUT_ROOT/$PARAM/${PARAM}_${VALUE}_seed${SEED}.npz"
    EFF="$OUTPUT_ROOT/$PARAM/${PARAM}_${VALUE}_seed${SEED}_efficiency.json"

    if [[ ! -f "$NPZ" ]]; then
        echo "[LAUNCHER] FATAL: $RUN_ID completed but npz missing: $NPZ" | tee -a "$LOG_ROOT/launcher.log"
        update_progress "$RUN_ID" "FAILED" "npz_missing=$NPZ"
        exit 1
    fi
    if [[ ! -f "$EFF" ]]; then
        echo "[LAUNCHER] FATAL: $RUN_ID completed but efficiency.json missing: $EFF" | tee -a "$LOG_ROOT/launcher.log"
        update_progress "$RUN_ID" "FAILED" "efficiency_missing=$EFF"
        exit 1
    fi

    COMPLETED+=("$RUN_ID")
    update_progress "$RUN_ID" "completed"
    echo "[LAUNCHER] $RUN_ID completed successfully (${#COMPLETED[@]}/$TOTAL)" | tee -a "$LOG_ROOT/launcher.log"
done

# --- Final summary ---
echo "" | tee -a "$LOG_ROOT/launcher.log"
echo "============================================" | tee -a "$LOG_ROOT/launcher.log"
echo "ALL $TOTAL RUNS COMPLETED SUCCESSFULLY" | tee -a "$LOG_ROOT/launcher.log"
echo "Time: $(date -u)" | tee -a "$LOG_ROOT/launcher.log"
echo "============================================" | tee -a "$LOG_ROOT/launcher.log"

update_progress "ALL" "ALL_COMPLETED"
