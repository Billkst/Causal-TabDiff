#!/bin/bash
set -euo pipefail

SEEDS=(42 52 62 72 82)
OUTPUT_BASE="outputs/fulldata_baselines"
LOG_BASE="logs/fulldata_baselines"
PYTHON="/home/UserData/miniconda/envs/causal_tabdiff/bin/python"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

mkdir -p "${LOG_BASE}"/{layer1,layer2,tstr,eval}
mkdir -p "${OUTPUT_BASE}"/{layer1,layer2,tstr,formal_runs/{layer1,layer2,tstr},summaries}

echo "=========================================="
echo "全量数据 Baseline 对比实验"
echo "数据: data/landmark_tables/unified_person_landmark_table.pkl"
echo "Seeds: ${SEEDS[*]}"
echo "启动时间: $(timestamp)"
echo "=========================================="

echo ""
echo "=== Phase 1: Layer1 Direct Baselines ==="

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting CausalForest seed=${seed}"
    nohup "${PYTHON}" -u train_causal_forest_b2.py \
        --seed "${seed}" \
        --output_dir "${OUTPUT_BASE}/layer1" \
        > "${LOG_BASE}/layer1/causal_forest_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer1/causal_forest_seed${seed}.pid"
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting iTransformer Layer1 seed=${seed}"
    nohup "${PYTHON}" -u train_tslib_models.py \
        --model itransformer \
        --seed "${seed}" \
        --epochs 50 \
        --output_dir "${OUTPUT_BASE}/layer1" \
        > "${LOG_BASE}/layer1/itransformer_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer1/itransformer_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting TSDiff Layer1 seed=${seed}"
    nohup "${PYTHON}" -u train_diffusion_layer1_formal.py \
        --model tsdiff \
        --seed "${seed}" \
        --epochs 5 \
        --output_dir "${OUTPUT_BASE}/layer1" \
        > "${LOG_BASE}/layer1/tsdiff_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer1/tsdiff_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    echo "[$(date +%H:%M:%S)] Starting STaSy Layer1 seed=${seed}"
    nohup "${PYTHON}" -u train_diffusion_layer1_formal.py \
        --model stasy \
        --seed "${seed}" \
        --epochs 5 \
        --output_dir "${OUTPUT_BASE}/layer1" \
        > "${LOG_BASE}/layer1/stasy_seed${seed}.log" 2>&1 &
    echo $! > "${LOG_BASE}/layer1/stasy_seed${seed}.pid"
    wait $!
done

for seed in "${SEEDS[@]}"; do
    pid_file="${LOG_BASE}/layer1/causal_forest_seed${seed}.pid"
    if [[ -f "${pid_file}" ]]; then
        wait "$(cat "${pid_file}")"
    fi
done

echo "Phase 1 完成: $(timestamp)"

echo ""
echo "=== Phase 2: Layer1 TSTR ==="

for model in tabsyn tabdiff survtraj sssd stasy tsdiff; do
    for seed in "${SEEDS[@]}"; do
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
nohup "${PYTHON}" -u scripts/eval_fulldata_baselines.py \
    --input_dir "${OUTPUT_BASE}" \
    --output_dir "${OUTPUT_BASE}/formal_runs" \
    > "${LOG_BASE}/eval/evaluation.log" 2>&1 &
echo $! > "${LOG_BASE}/eval/evaluation.pid"
wait $!

echo ""
echo "=== Phase 5: 生成汇总表 ==="
nohup "${PYTHON}" -u scripts/finalize_b2_official_5seed.py \
    > "${LOG_BASE}/eval/finalize.log" 2>&1 &
echo $! > "${LOG_BASE}/eval/finalize.pid"
wait $!

echo ""
echo "=========================================="
echo "全量数据 Baseline 对比实验全部完成"
echo "完成时间: $(timestamp)"
echo "结果目录: ${OUTPUT_BASE}/"
echo "=========================================="
