#!/bin/bash
# B2 Baseline 自动化监控和汇总脚本

echo "=== B2 Baseline 实验监控 ==="
echo "等待所有训练任务完成..."

while true; do
    running=$(ps aux | grep -E "(train_causal_forest|train_tslib)" | grep -v grep | wc -l)
    
    if [ $running -eq 0 ]; then
        echo "✓ 所有训练任务已完成"
        break
    fi
    
    echo "$(date '+%H:%M:%S') - 运行中: $running 个任务"
    sleep 30
done

echo ""
echo "=== 开始自动评估 ==="
bash scripts/eval_b2_layer1.sh
bash scripts/eval_b2_layer2.sh

echo ""
echo "=== 生成汇总表格 ==="
python scripts/generate_b2_tables.py

echo ""
echo "=== B2 Baseline 实验完成 ==="
