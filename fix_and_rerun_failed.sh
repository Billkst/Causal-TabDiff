#!/bin/bash
set -uo pipefail

SEEDS=(42 52 62 72 82)
OUTPUT_BASE="outputs/fulldata_baselines"
LOG_BASE="logs/fulldata_baselines"
PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================="
echo "补跑失败的 TSTR 任务: STaSy + TSDiff"
echo "时间: $(timestamp)"
echo "=========================================="

echo ""
echo "=== STaSy TSTR (采样 bug 已修复: config.sampling 补全) ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/stasy_seed${seed}_predictions.npz"
    if [[ -f "${pred}" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP stasy seed=${seed} (已完成)"
        continue
    fi
    echo "[$(date +%H:%M:%S)] Starting stasy TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model stasy --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/stasy_seed${seed}.log" 2>&1 || echo "[WARN] stasy seed=${seed} exited with code $?"
done

echo ""
echo "=== TSDiff TSTR (Y_syn 二值化 bug 已修复) ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/tsdiff_seed${seed}_predictions.npz"
    if [[ -f "${pred}" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP tsdiff seed=${seed} (已完成)"
        continue
    fi
    echo "[$(date +%H:%M:%S)] Starting tsdiff TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model tsdiff --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/tsdiff_seed${seed}.log" 2>&1 || echo "[WARN] tsdiff seed=${seed} exited with code $?"
done

echo ""
echo "=== 重新运行统一评估 ==="
"${PYTHON}" -u scripts/eval_fulldata_baselines.py \
    --input_dir "${OUTPUT_BASE}" \
    --output_dir "${OUTPUT_BASE}/formal_runs" \
    > "${LOG_BASE}/eval/evaluation_final.log" 2>&1 || echo "[WARN] Evaluation exited with code $?"

echo ""
echo "=========================================="
echo "补跑完成: $(timestamp)"
echo "=========================================="
