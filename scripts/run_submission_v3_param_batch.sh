#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PROJECT_ROOT="/home/UserData/ljx/Project_2/Causal-TabDiff"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="outputs/submission_v3/param_discussion_layer2"
LOG_ROOT="logs/param_discussion_layer2_v3"
CHECKPOINT_ROOT="checkpoints/param_discussion_layer2_v3"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate causal_tabdiff

echo "========================================"
echo "Submission V3 Parameter Discussion Batch"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Output: $OUTPUT_ROOT"
echo "Log: $LOG_ROOT"
echo "========================================"

RUNS=(
    "focal_gamma 4.0"
    "oversample_ratio 2.5"
    "oversample_ratio 7.5"
    "disc_weight 0.01"
    "disc_weight 0.05"
    "disc_weight 0.1"
    "disc_weight 0.2"
)

TOTAL=${#RUNS[@]}
COMPLETED=0
FAILED=0

for run_spec in "${RUNS[@]}"; do
    read -r param value <<< "$run_spec"
    COMPLETED=$((COMPLETED + 1))
    
    echo ""
    echo "========================================"
    echo "[$COMPLETED/$TOTAL] Starting: ${param}=${value} seed=42"
    echo "Time: $(date)"
    echo "========================================"
    
    RUN_LOG="${LOG_ROOT}/${param}/${param}_${value}_seed42.log"
    mkdir -p "$(dirname "$RUN_LOG")"
    
    if python -u scripts/run_param_discussion_layer2_official.py \
        --seed 42 \
        --param "$param" \
        --values "$value" \
        --output_root "$OUTPUT_ROOT" \
        --log_root "$LOG_ROOT" \
        --checkpoint_root "$CHECKPOINT_ROOT" \
        --use_cached_dataset \
        2>&1 | tee "$RUN_LOG"; then
        
        BASE="${param}_${value}_seed42"
        PDIR="${OUTPUT_ROOT}/${param}"
        if [ -f "${PDIR}/${BASE}.npz" ]; then
            mv "${PDIR}/${BASE}.npz" "${PDIR}/${BASE}_official.npz"
            mv "${PDIR}/${BASE}_efficiency.json" "${PDIR}/${BASE}_official_efficiency.json"
            echo "[OK] ${param}=${value} completed + renamed to _official"
        else
            echo "[OK] ${param}=${value} completed (no rename needed)"
        fi
    else
        echo "[FAIL] ${param}=${value} FAILED with exit code $?"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Batch Complete: $(date)"
echo "Total: $TOTAL | Completed: $COMPLETED | Failed: $FAILED"
echo "========================================"
