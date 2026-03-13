#!/bin/bash
SEEDS=(42 52 62 72 82)
mkdir -p logs/b2_baseline/layer2 outputs/b2_baseline/layer2

for seed in "${SEEDS[@]}"; do
    echo "启动 TimeXer Layer2 seed=${seed}"
    nohup python -u train_tslib_layer2.py --model timexer --seed ${seed} --epochs 30 > logs/b2_baseline/layer2/timexer_seed${seed}.log 2>&1 &
done

echo "所有 Layer2 任务已启动"
