#!/bin/bash
cd /home/UserData/ljx/Project_2/Causal-TabDiff

echo "=== 优化版消融实验 (串行执行) ==="
echo "优化: Epochs 30, Batch 1024, 串行运行"
echo ""

echo "[1/4] 因果判别器梯度引导权重..."
python -u run_ablation_complete.py --ablation_type disc_weight

echo "[2/4] 扩散步数..."
python -u run_ablation_complete.py --ablation_type diffusion_steps

echo "[3/4] 正交双重注意力头数..."
python -u run_ablation_complete.py --ablation_type heads

echo "[4/4] 轨迹损失权重..."
python -u run_ablation_complete.py --ablation_type traj_weight

echo ""
echo "所有消融实验完成！"
