#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

echo "启动完整消融实验 (5个模块)"

nohup python -u run_ablation_complete.py --ablation_type disc_weight > logs/ablations/disc_weight_run.log 2>&1 &
echo "消融1: 因果判别器梯度引导权重 [0.0, 0.25, 0.5, 0.75, 1.0]"

nohup python -u run_ablation_complete.py --ablation_type diffusion_steps > logs/ablations/diffusion_steps_run.log 2>&1 &
echo "消融2: 扩散步数 [25, 50, 100, 150, 200]"

nohup python -u run_ablation_complete.py --ablation_type heads > logs/ablations/heads_run.log 2>&1 &
echo "消融3: 正交双重注意力头数 [1, 2, 4, 6, 8]"

nohup python -u run_ablation_complete.py --ablation_type traj_weight > logs/ablations/traj_weight_run.log 2>&1 &
echo "消融4: 轨迹损失权重(共病模式) [0.0, 0.5, 1.0, 1.5, 2.0]"

echo ""
echo "所有消融实验已启动 (每个5个测试值 × 5 seeds = 25次训练)"
echo ""
echo "监控: tail -f logs/ablations/disc_weight.log"
