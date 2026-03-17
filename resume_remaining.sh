#!/bin/bash
set -uo pipefail

SEEDS=(42 52 62 72 82)
OUTPUT_BASE="outputs/fulldata_baselines"
LOG_BASE="logs/fulldata_baselines"
PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

skip_if_done() {
    local pred="$1" fail="$2" label="$3"
    if [[ -f "${pred}" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP ${label} (已完成)"
        return 0
    fi
    if [[ -f "${fail}" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP ${label} (之前 FAILED)"
        return 0
    fi
    return 1
}

echo "=========================================="
echo "恢复剩余任务: STaSy TSTR + TSDiff TSTR + Layer2 全部 + 评估"
echo "时间: $(timestamp)"
echo "=========================================="

echo ""
echo "=== TSTR: STaSy ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/stasy_seed${seed}_predictions.npz"
    fail="${OUTPUT_BASE}/tstr/stasy_seed${seed}_FAILED.txt"
    skip_if_done "$pred" "$fail" "stasy_seed${seed}" && continue
    echo "[$(date +%H:%M:%S)] Starting stasy TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model stasy --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/stasy_seed${seed}.log" 2>&1 || echo "[WARN] stasy seed=${seed} exited with code $?"
done

echo ""
echo "=== TSTR: TSDiff ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/tsdiff_seed${seed}_predictions.npz"
    fail="${OUTPUT_BASE}/tstr/tsdiff_seed${seed}_FAILED.txt"
    skip_if_done "$pred" "$fail" "tsdiff_seed${seed}" && continue
    echo "[$(date +%H:%M:%S)] Starting tsdiff TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model tsdiff --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/tsdiff_seed${seed}.log" 2>&1 || echo "[WARN] tsdiff seed=${seed} exited with code $?"
done

echo "TSTR 完成: $(timestamp)"

echo ""
echo "=== Phase 3: Layer2 Trajectory ==="

for model in itransformer timexer; do
    for seed in "${SEEDS[@]}"; do
        pred="${OUTPUT_BASE}/layer2/${model}_seed${seed}_layer2.npz"
        [[ -f "$pred" ]] && echo "[$(date +%H:%M:%S)] SKIP ${model} L2 seed=${seed}" && continue
        echo "[$(date +%H:%M:%S)] Starting ${model} Layer2 seed=${seed}"
        "${PYTHON}" -u train_tslib_layer2.py \
            --model "${model}" --seed "${seed}" --epochs 30 \
            --output_dir "${OUTPUT_BASE}/layer2" \
            > "${LOG_BASE}/layer2/${model}_seed${seed}.log" 2>&1 || echo "[WARN] ${model} L2 seed=${seed} exited with code $?"
    done
done

for model in sssd survtraj; do
    for seed in "${SEEDS[@]}"; do
        pred="${OUTPUT_BASE}/layer2/${model}_seed${seed}_layer2.npz"
        [[ -f "$pred" ]] && echo "[$(date +%H:%M:%S)] SKIP ${model} L2 seed=${seed}" && continue
        echo "[$(date +%H:%M:%S)] Starting ${model} Layer2 seed=${seed}"
        "${PYTHON}" -u train_generative_layer2.py \
            --model "${model}" --seed "${seed}" --epochs 30 \
            --output_dir "${OUTPUT_BASE}/layer2" \
            > "${LOG_BASE}/layer2/${model}_seed${seed}.log" 2>&1 || echo "[WARN] ${model} L2 seed=${seed} exited with code $?"
    done
done

echo "Phase 3 完成: $(timestamp)"

echo ""
echo "=== Phase 4: 统一评估 ==="
"${PYTHON}" -u scripts/eval_fulldata_baselines.py \
    --input_dir "${OUTPUT_BASE}" \
    --output_dir "${OUTPUT_BASE}/formal_runs" \
    > "${LOG_BASE}/eval/evaluation.log" 2>&1 || echo "[WARN] Evaluation exited with code $?"

echo ""
echo "=========================================="
echo "全部完成: $(timestamp)"
echo "=========================================="
