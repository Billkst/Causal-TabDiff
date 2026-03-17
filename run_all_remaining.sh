#!/bin/bash
set -uo pipefail

SEEDS=(42 52 62 72 82)
OUTPUT_BASE="outputs/fulldata_baselines"
LOG_BASE="logs/fulldata_baselines"
PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=========================================="
echo "全量 Baseline 实验 — 跑所有未完成任务"
echo "时间: $(timestamp)"
echo "=========================================="

# === TSTR: STaSy (bug 已修: config.sampling 补全) ===
echo ""
echo "=== TSTR: STaSy ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/stasy_seed${seed}_predictions.npz"
    [[ -f "${pred}" ]] && echo "[$(date +%H:%M:%S)] SKIP stasy seed=${seed}" && continue
    echo "[$(date +%H:%M:%S)] Starting stasy TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model stasy --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/stasy_seed${seed}.log" 2>&1 || echo "[WARN] stasy seed=${seed} exit $?"
done

# === TSTR: TSDiff (bug 已修: Y_syn 二值化) ===
echo ""
echo "=== TSTR: TSDiff ==="
for seed in "${SEEDS[@]}"; do
    pred="${OUTPUT_BASE}/tstr/tsdiff_seed${seed}_predictions.npz"
    [[ -f "${pred}" ]] && echo "[$(date +%H:%M:%S)] SKIP tsdiff seed=${seed}" && continue
    echo "[$(date +%H:%M:%S)] Starting tsdiff TSTR seed=${seed}"
    "${PYTHON}" -u train_generative_strict.py \
        --model tsdiff --seed "${seed}" --epochs 30 --n_synthetic -1 \
        --output_dir "${OUTPUT_BASE}/tstr" \
        > "${LOG_BASE}/tstr/tsdiff_seed${seed}.log" 2>&1 || echo "[WARN] tsdiff seed=${seed} exit $?"
done

echo "TSTR 完成: $(timestamp)"

# === Layer2: TSLib (iTransformer + TimeXer) ===
echo ""
echo "=== Layer2: TSLib ==="
for model in itransformer timexer; do
    for seed in "${SEEDS[@]}"; do
        pred="${OUTPUT_BASE}/layer2/${model}_seed${seed}_layer2.npz"
        [[ -f "$pred" ]] && echo "[$(date +%H:%M:%S)] SKIP ${model} L2 seed=${seed}" && continue
        echo "[$(date +%H:%M:%S)] Starting ${model} Layer2 seed=${seed}"
        "${PYTHON}" -u train_tslib_layer2.py \
            --model "${model}" --seed "${seed}" --epochs 30 \
            --output_dir "${OUTPUT_BASE}/layer2" \
            > "${LOG_BASE}/layer2/${model}_seed${seed}.log" 2>&1 || echo "[WARN] ${model} L2 seed=${seed} exit $?"
    done
done

# === Layer2: Generative (SSSD + SurvTraj) ===
echo ""
echo "=== Layer2: Generative ==="
for model in sssd survtraj; do
    for seed in "${SEEDS[@]}"; do
        pred="${OUTPUT_BASE}/layer2/${model}_seed${seed}_layer2.npz"
        [[ -f "$pred" ]] && echo "[$(date +%H:%M:%S)] SKIP ${model} L2 seed=${seed}" && continue
        echo "[$(date +%H:%M:%S)] Starting ${model} Layer2 seed=${seed}"
        "${PYTHON}" -u train_generative_layer2.py \
            --model "${model}" --seed "${seed}" --epochs 30 \
            --output_dir "${OUTPUT_BASE}/layer2" \
            > "${LOG_BASE}/layer2/${model}_seed${seed}.log" 2>&1 || echo "[WARN] ${model} L2 seed=${seed} exit $?"
    done
done

echo "Layer2 完成: $(timestamp)"

# === 统一评估 ===
echo ""
echo "=== 统一评估 ==="
"${PYTHON}" -u scripts/eval_fulldata_baselines.py \
    --input_dir "${OUTPUT_BASE}" \
    --output_dir "${OUTPUT_BASE}/formal_runs" \
    > "${LOG_BASE}/eval/evaluation_final.log" 2>&1 || echo "[WARN] Evaluation exit $?"

echo ""
echo "=========================================="
echo "全部完成: $(timestamp)"
echo "=========================================="
