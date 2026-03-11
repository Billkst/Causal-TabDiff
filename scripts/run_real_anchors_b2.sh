#!/bin/bash

SEEDS=(42 52 62 72 82)
MODELS=("lr" "xgb" "brf")
OUTPUT_DIR="outputs/real_anchors_b2"
LOG_DIR="logs/real_anchors_b2"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "B2-1: Real-data Anchors Training"
echo "=========================================="
echo "Models: Logistic Regression, XGBoost, Balanced Random Forest"
echo "Seeds: ${SEEDS[@]}"
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Training ${model^^}"
    echo "=========================================="
    
    for seed in "${SEEDS[@]}"; do
        echo "  Seed ${seed}..."
        
        log_file="${LOG_DIR}/${model}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"
        
        nohup python3 train_real_anchors.py \
            --model ${model} \
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
        eval_log="${LOG_DIR}/${model}_seed${seed}_eval_$(date +%Y%m%d_%H%M%S).log"
        
        nohup python3 evaluate_real_anchors.py \
            --predictions_file ${OUTPUT_DIR}/${model}_seed${seed}_predictions.npz \
            --output_dir ${OUTPUT_DIR} \
            --model_name ${model}_seed${seed} \
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
done

echo "=========================================="
echo "All training and evaluation completed!"
echo "=========================================="
echo "Results: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
