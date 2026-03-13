#!/bin/bash
SEEDS=(42 52 62 72 82)
mkdir -p logs/b2_baseline/layer1 outputs/b2_baseline/layer1

for seed in "${SEEDS[@]}"; do
    echo "启动 CausalForest seed=${seed}"
    nohup python -u train_causal_forest_b2.py --seed ${seed} --output_dir outputs/b2_baseline/layer1 > logs/b2_baseline/layer1/causal_forest_seed${seed}.log 2>&1 &
done

for seed in "${SEEDS[@]}"; do
    echo "启动 iTransformer seed=${seed}"
    nohup python -u train_tslib_models.py --model itransformer --seed ${seed} --epochs 50 > logs/b2_baseline/layer1/itransformer_seed${seed}.log 2>&1 &
done

echo "所有 Layer1 任务已启动"
