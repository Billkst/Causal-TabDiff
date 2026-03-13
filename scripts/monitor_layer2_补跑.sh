#!/bin/bash
# B2 Baseline Layer2 补跑监控脚本

echo "=== B2 Baseline Layer2 补跑监控 ==="
echo "等待所有训练任务完成..."

while true; do
    running=$(ps aux | grep -E "(train_tslib_layer2|train_generative_layer2)" | grep -v grep | wc -l)
    
    if [ $running -eq 0 ]; then
        echo "✓ 所有训练任务已完成"
        break
    fi
    
    echo "$(date '+%H:%M:%S') - 运行中: $running 个任务"
    sleep 30
done

echo ""
echo "=== 开始评估 Layer2 结果 ==="

for model in itransformer timexer sssd survtraj; do
    for seed in 42 52 62 72 82; do
        pred_file="outputs/tslib_layer2/${model}_seed${seed}_layer2.npz"
        if [ ! -f "$pred_file" ]; then
            pred_file="outputs/b2_baseline/layer2/${model}_seed${seed}_layer2.npz"
        fi
        
        if [ -f "$pred_file" ]; then
            echo "评估 ${model} seed=${seed}"
            python evaluate_layer2.py \
                --predictions_file $pred_file \
                --output_dir outputs/b2_baseline/layer2 \
                --model_name ${model}_seed${seed} \
                > logs/b2_baseline/layer2_补跑/${model}_seed${seed}_eval.log 2>&1
        fi
    done
done

echo ""
echo "=== 生成汇总表格 ==="
python scripts/generate_b2_tables.py

echo ""
echo "=== B2 Baseline Layer2 补跑完成 ==="
