#!/bin/bash

SEEDS=(42 52 62 72 82)
OUTPUT_DIR="outputs/retained_baselines_b2"
LOG_DIR="logs/retained_baselines_b2"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "B2-2: Retained Baselines Training"
echo "=========================================="
echo "Part A: CausalForest (Direct)"
echo "Part B: STaSy/TabSyn/TabDiff/TSDiff (TSTR - TODO)"
echo "Seeds: ${SEEDS[@]}"
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

echo "=========================================="
echo "Part A: Training CausalForest"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo "  Seed ${seed}..."
    
    log_file="${LOG_DIR}/causal_forest_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
    
    nohup python3 train_causal_forest_b2.py \
        --seed ${seed} \
        --output_dir ${OUTPUT_DIR} \
        > ${log_file} 2>&1 &
    
    pid=$!
    echo "    PID: ${pid}, Log: ${log_file}"
    
    wait ${pid}
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Training completed"
    else
        echo "    ✗ Training failed (check log)"
    fi
    
    echo "  Evaluating..."
    eval_log="${LOG_DIR}/causal_forest_seed${seed}_eval_$(date +%Y%m%d_%H%M%S).log"
    
    nohup python3 evaluate_real_anchors.py \
        --predictions_file ${OUTPUT_DIR}/causal_forest_seed${seed}_predictions.npz \
        --output_dir ${OUTPUT_DIR} \
        --model_name causal_forest_seed${seed} \
        > ${eval_log} 2>&1 &
    
    pid=$!
    wait ${pid}
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Evaluation completed"
    else
        echo "    ✗ Evaluation failed (check log)"
    fi
    
    echo ""
done

echo ""
echo "=========================================="
echo "Part B: TSTR Baselines (TODO)"
echo "=========================================="
echo "⚠️  STaSy/TabSyn/TabDiff/TSDiff require TSTR implementation"
echo "   These will be implemented in next phase"
echo ""

echo "=========================================="
echo "CausalForest training completed!"
echo "=========================================="
echo "Results: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
