#!/bin/bash
# Baseline 最终封版实验 - 统一 9 seeds 批量训练脚本

set -e

SEEDS=(42 52 62 72 82 1024 2024 2025 9999)
OUTPUT_BASE="outputs/b2_baseline"

echo "=========================================="
echo "Baseline 最终封版实验"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Layer1 Direct Baselines
echo ""
echo "=== Layer1 Direct Baselines ==="

# CausalForest
for seed in "${SEEDS[@]}"; do
    echo "Training CausalForest seed=$seed"
    python train_causal_forest_b2.py --seed $seed --output_dir $OUTPUT_BASE/layer1
done

# iTransformer
for seed in "${SEEDS[@]}"; do
    echo "Training iTransformer seed=$seed"
    python train_tslib_models.py --model itransformer --seed $seed --epochs 50
done

# TimeXer
for seed in "${SEEDS[@]}"; do
    echo "Training TimeXer seed=$seed"
    python train_tslib_models.py --model timexer --seed $seed --epochs 50
done

echo ""
echo "=== 所有训练完成 ==="
