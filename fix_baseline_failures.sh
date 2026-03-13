#!/bin/bash
# B2 Baseline 失败修复脚本

set -e

echo "================================================================================"
echo "B2 Baseline 失败修复脚本"
echo "================================================================================"

# 激活环境
echo "[1/5] 激活 conda 环境..."
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate causal_tabdiff

# 修复 XGBoost 依赖
echo "[2/5] 修复 TSTR XGBoost 依赖..."
python << 'PYTHON_FIX'
import sys
sys.path.insert(0, 'src')

# 读取文件
with open('src/baselines/tstr_pipeline.py', 'r') as f:
    content = f.read()

# 替换 XGBoost 配置
old_code = """            self.classifier = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                tree_method='hist',
                device='cpu'
            )"""

new_code = """            self.classifier = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                tree_method='hist'
            )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('src/baselines/tstr_pipeline.py', 'w') as f:
        f.write(content)
    print("✓ XGBoost 配置已修复 (移除 device='cpu')")
else:
    print("⚠️ 未找到预期的 XGBoost 配置，请手动检查")
PYTHON_FIX

# 重新训练 TSTR 模型
echo "[3/5] 重新训练 TSDiff 和 STaSy..."
for seed in 42 52 62 72 82; do
    echo "  - TSDiff seed=$seed"
    python train_tstr_pipeline.py --model tsdiff --seed $seed 2>&1 | tee logs/b2_baseline/tsdiff_stasy_正式/tsdiff_seed${seed}_rerun.log
    
    echo "  - STaSy seed=$seed"
    python train_tstr_pipeline.py --model stasy --seed $seed 2>&1 | tee logs/b2_baseline/tsdiff_stasy_正式/stasy_seed${seed}_rerun.log
done

# 重新计算 Layer2 指标
echo "[4/5] 重新计算 Layer2 指标..."
for model in itransformer sssd survtraj; do
    echo "  - $model Layer2 指标"
    python train_tslib_layer2.py --model $model --compute-metrics-only 2>&1 | tee logs/b2_baseline/layer2/${model}_metrics_rerun.log
done

# 生成 CausalForest 预测
echo "[5/5] 生成 CausalForest 预测..."
for seed in 42 52 62 72 82; do
    echo "  - CausalForest seed=$seed"
    python train_causal_forest_b2.py --generate-predictions --seed $seed 2>&1 | tee logs/b2_baseline/layer1/causal_forest_seed${seed}_pred_rerun.log
done

echo "================================================================================"
echo "✓ 所有修复任务完成"
echo "================================================================================"
