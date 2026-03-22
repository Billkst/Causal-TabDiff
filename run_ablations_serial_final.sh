#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

echo "串行运行3个消融实验"
python -u ablation_minimal.py disc_weight
python -u ablation_minimal.py diffusion_steps  
python -u ablation_minimal.py traj_weight
echo "完成"
