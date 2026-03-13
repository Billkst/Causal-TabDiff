#!/bin/bash
# B2 Baseline 最终静态验收脚本

echo "=== B2 Baseline 最终静态验收 ==="
echo "等待所有补跑任务完成..."

# 等待 TSDiff/STaSy 补跑完成
while true; do
    running=$(ps aux | grep "train_tstr_pipeline" | grep -v grep | wc -l)
    if [ $running -eq 0 ]; then
        echo "✓ TSDiff/STaSy 补跑完成"
        break
    fi
    echo "$(date '+%H:%M:%S') - TSDiff/STaSy 运行中: $running 个任务"
    sleep 30
done

# 等待 Layer2 补跑完成
while true; do
    running=$(ps aux | grep -E "(train_tslib_layer2|train_generative_layer2)" | grep -v grep | wc -l)
    if [ $running -eq 0 ]; then
        echo "✓ Layer2 补跑完成"
        break
    fi
    echo "$(date '+%H:%M:%S') - Layer2 运行中: $running 个任务"
    sleep 30
done

echo ""
echo "=== 评估所有结果 ==="

# 评估 TSDiff/STaSy
for model in tsdiff stasy; do
    for seed in 42 52 62 72 82; do
        pred_file="outputs/tstr_baselines/${model}_seed${seed}_predictions.npz"
        if [ -f "$pred_file" ]; then
            echo "评估 ${model} seed=${seed}"
            python evaluate_model.py \
                --predictions_file $pred_file \
                --output_dir outputs/b2_baseline/tsdiff_stasy \
                --model_name ${model}_seed${seed} \
                > logs/b2_baseline/tsdiff_stasy_正式/${model}_seed${seed}_eval.log 2>&1
        fi
    done
done

# 评估 Layer2
for model in itransformer timexer sssd survtraj; do
    for seed in 42 52 62 72 82; do
        pred_file="outputs/tslib_layer2/${model}_seed${seed}_layer2.npz"
        if [ ! -f "$pred_file" ]; then
            pred_file="outputs/b2_baseline/layer2/${model}_seed${seed}_layer2.npz"
        fi
        
        if [ -f "$pred_file" ]; then
            echo "评估 ${model} layer2 seed=${seed}"
            python evaluate_layer2.py \
                --predictions_file $pred_file \
                --output_dir outputs/b2_baseline/layer2 \
                --model_name ${model}_seed${seed} \
                > logs/b2_baseline/layer2_补跑/${model}_seed${seed}_eval.log 2>&1
        fi
    done
done

echo ""
echo "=== 生成最终汇总表格 ==="
python scripts/generate_b2_tables.py

echo ""
echo "=== 生成效率表 ==="
python scripts/generate_efficiency_table.py

echo ""
echo "=== 生成对账清单 ==="
python scripts/generate_reconciliation_report.py

echo ""
echo "=== B2 Baseline 最终静态验收完成 ==="
