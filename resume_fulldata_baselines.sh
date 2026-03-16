#!/bin/bash
set -euo pipefail

SEEDS=(42 52 62 72 82)
OUTPUT_BASE="outputs/fulldata_baselines"
LOG_BASE="logs/fulldata_baselines"
PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "=========================================="
echo "全量数据 Baseline 对比实验 — 恢复运行"
echo "跳过已完成: Phase1 全部, Phase2 tabsyn"
echo "从 tabdiff TSTR 继续"
echo "恢复时间: $(timestamp)"
echo "=========================================="

echo ""
echo "=== Phase 2 (续): Layer1 TSTR ==="

for model in tabdiff survtraj sssd stasy tsdiff; do
    for seed in "${SEEDS[@]}"; do
        pred_file="${OUTPUT_BASE}/tstr/${model}_seed${seed}_predictions.npz"
        fail_file="${OUTPUT_BASE}/tstr/${model}_seed${seed}_FAILED.txt"
        if [[ -f "${pred_file}" ]]; then
            echo "[$(date +%H:%M:%S)] SKIP ${model} seed=${seed} (已完成)"
            continue
        fi
        if [[ -f "${fail_file}" ]]; then
            echo "[$(date +%H:%M:%S)] RETRY ${model} seed=${seed} (之前失败，重试)"
        fi
        echo "[$(date +%H:%M:%S)] Starting ${model} TSTR seed=${seed}"
        nohup "${PYTHON}" -u train_generative_strict.py \
            --model "${model}" \
            --seed "${seed}" \
            --epochs 30 \
            --n_synthetic -1 \
            --output_dir "${OUTPUT_BASE}/tstr" \
            > "${LOG_BASE}/tstr/${model}_seed${seed}.log" 2>&1 &
        echo $! > "${LOG_BASE}/tstr/${model}_seed${seed}.pid"
        wait $!
    done
done

echo "Phase 2 完成: $(timestamp)"

echo ""
echo "=== Phase 3: Layer2 Trajectory ==="

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting iTransformer Layer2 seed=${seed}"
    nohup "${PYTHON}" -u train_tslib_layer2.py \
        --model itransformer \
        --seed "${seed}" \
        --epochs 30 \
        --output_dir "${OUTPUT_BASE}/layer2" \
        > "${LOG_BASE}/layer2/itransformer_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer2/itransformer_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting TimeXer Layer2 seed=${seed}"
    nohup "${PYTHON}" -u train_tslib_layer2.py \
        --model timexer \
        --seed "${seed}" \
        --epochs 30 \
        --output_dir "${OUTPUT_BASE}/layer2" \
        > "${LOG_BASE}/layer2/timexer_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer2/timexer_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting SSSD Layer2 seed=${seed}"
    nohup "${PYTHON}" -u train_generative_layer2.py \
        --model sssd \
        --seed "${seed}" \
        --epochs 30 \
        --output_dir "${OUTPUT_BASE}/layer2" \
        > "${LOG_BASE}/layer2/sssd_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer2/sssd_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting SurvTraj Layer2 seed=${seed}"
    nohup "${PYTHON}" -u train_generative_layer2.py \
        --model survtraj \
        --seed "${seed}" \
        --epochs 30 \
        --output_dir "${OUTPUT_BASE}/layer2" \
        > "${LOG_BASE}/layer2/survtraj_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer2/survtraj_seed${seed}.pid"
    wait $!
done

echo "Phase 3 完成: $(timestamp)"

echo ""
echo "=== Phase 4: 统一评估 ==="
"${PYTHON}" -u scripts/eval_fulldata_baselines.py \
    --input_dir "${OUTPUT_BASE}" \
    --output_dir "${OUTPUT_BASE}/formal_runs" \
    > "${LOG_BASE}/eval/evaluation.log" 2>&1

echo ""
echo "=========================================="
echo "全量数据 Baseline 对比实验全部完成"
echo "完成时间: $(timestamp)"
echo "结果目录: ${OUTPUT_BASE}/"
echo "=========================================="
