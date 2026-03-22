#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff
conda activate causal_tabdiff

nohup python -u run_ablation_disc_weight.py > logs/ablation_disc_weight_run.log 2>&1 &

echo "消融实验已启动"
echo "实时监控: tail -f logs/ablation_disc_weight.log"
echo "运行日志: tail -f logs/ablation_disc_weight_run.log"
