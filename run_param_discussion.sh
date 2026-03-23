#!/bin/bash
# ============================================================
# Causal-TabDiff 参数讨论实验启动脚本
# 4 参数 × 5 值 × 5 seeds = 100 组
# ============================================================

set -e

CONDA_ENV="causal_tabdiff"
SCRIPT="run_param_discussion.py"
LOG_DIR="logs/param_discussion"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Causal-TabDiff 参数讨论实验"
echo "启动时间: $(date)"
echo "Conda 环境: $CONDA_ENV"
echo "============================================================"

# 激活 conda
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

echo "Python: $(which python)"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 运行所有参数实验 (串行)
MASTER_LOG="$LOG_DIR/param_discussion_${TIMESTAMP}.log"

echo "主日志: $MASTER_LOG"
echo "启动全部参数讨论实验..."
echo ""

python -u $SCRIPT --param all 2>&1 | tee "$MASTER_LOG"

echo ""
echo "============================================================"
echo "所有参数讨论实验完成!"
echo "完成时间: $(date)"
echo "结果目录: outputs/param_discussion/"
echo "日志目录: $LOG_DIR/"
echo "============================================================"
