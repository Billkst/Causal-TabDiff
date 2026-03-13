#!/bin/bash
SEEDS=(42 52 62 72 82)
mkdir -p logs/b2_baseline/tstr outputs/b2_baseline/tstr

for model in tabsyn tabdiff survtraj sssd; do
    for seed in "${SEEDS[@]}"; do
        echo "启动 ${model}_strict seed=${seed}"
        nohup python -u train_generative_strict.py \
            --model ${model} \
            --seed ${seed} \
            --epochs 30 \
            --output_dir outputs/b2_baseline/tstr \
            > logs/b2_baseline/tstr/${model}_seed${seed}.log 2>&1 &
    done
done

echo "所有 generative baseline 已启动"
