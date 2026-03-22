#!/bin/bash
# 后台运行所有6个TSTR生成模型的重训+采样延迟测量
# 每个模型顺序执行（共享GPU），日志写入 logs/

set -e
MODELS="sssd tabsyn stasy tsdiff survtraj tabdiff"

for model in $MODELS; do
    echo "========================================" 
    echo "[$(date)] Starting $model (30 epochs, seed=42)"
    echo "========================================"
    python -u scripts/retrain_and_measure_generative.py --model $model --seed 42 --epochs 30
    echo "[$(date)] Finished $model"
    echo ""
done

echo "[$(date)] ALL MODELS COMPLETE"
