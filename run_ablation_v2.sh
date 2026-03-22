#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

echo "=== 消融实验 v2 串行运行 ==="
echo "开始时间: $(date)"

echo "--- 消融1: guidance_scale ---"
python -u ablation_v2.py guidance_scale

echo "--- 消融2: diffusion_steps ---"
python -u ablation_v2.py diffusion_steps

echo "--- 消融3: traj_weight ---"
python -u ablation_v2.py traj_weight

echo "=== 全部完成 ==="
echo "结束时间: $(date)"
