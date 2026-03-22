#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff
conda activate causal_tabdiff

echo "启动 4 个消融实验..."

nohup python -u run_ablation_unified.py --ablation_type disc_weight > logs/ablations/disc_weight_run.log 2>&1 &
echo "消融1: 判别器损失权重 已启动"

nohup python -u run_ablation_unified.py --ablation_type diffusion_steps > logs/ablations/diffusion_steps_run.log 2>&1 &
echo "消融2: 扩散步数 已启动"

nohup python -u run_ablation_unified.py --ablation_type heads > logs/ablations/heads_run.log 2>&1 &
echo "消融3: 注意力头数 已启动"

nohup python -u run_ablation_unified.py --ablation_type traj_weight > logs/ablations/traj_weight_run.log 2>&1 &
echo "消融4: 轨迹损失权重 已启动"

echo ""
echo "所有消融实验已启动"
echo "监控命令:"
echo "  tail -f logs/ablations/disc_weight.log"
echo "  tail -f logs/ablations/diffusion_steps.log"
echo "  tail -f logs/ablations/heads.log"
echo "  tail -f logs/ablations/traj_weight.log"
