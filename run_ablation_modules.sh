#!/bin/bash
# 模块消融实验启动脚本
# 用法: bash run_ablation_modules.sh
# 后台运行: mkdir -p logs/ablation_modules && nohup bash run_ablation_modules.sh > logs/ablation_modules/nohup.log 2>&1 &

set -e

# 切换到项目根目录（无论从哪里启动都能找到正确路径）
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate causal_tabdiff

mkdir -p logs/ablation_modules

echo "=========================================="
echo "Causal-TabDiff 模块消融实验"
echo "=========================================="
echo "工作目录: $(pwd)"
echo "开始时间: $(date)"
echo "配置: 5 variants × 5 seeds × 30 epochs"
echo ""

python -u run_ablation_modules.py

echo ""
echo "结束时间: $(date)"
echo "=========================================="
