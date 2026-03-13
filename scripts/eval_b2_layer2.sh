#!/bin/bash
echo "=== 自动评估 Layer2 结果 ==="
SEEDS=(42 52 62 72 82)

for seed in "${SEEDS[@]}"; do
    pred_file="outputs/tslib_layer2/timexer_seed${seed}_layer2.npz"
    if [ -f "$pred_file" ]; then
        echo "评估 TimeXer Layer2 seed=${seed}"
        python evaluate_layer2.py \
            --predictions_file $pred_file \
            --output_dir outputs/b2_baseline/layer2 \
            --model_name TimeXer_seed${seed} \
            > logs/b2_baseline/layer2/timexer_seed${seed}_eval.log 2>&1
    fi
done

echo "✓ Layer2 评估完成"
