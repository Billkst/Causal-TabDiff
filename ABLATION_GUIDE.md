# 消融实验设计文档

## 实验概述

**目标**: 验证 CausalTabDiff 模型各核心组件对性能的贡献

**统一协议**:
- Seeds: [42, 52, 62, 72, 82] (与对比实验一致)
- Epochs: 50
- 评估指标: 验证集最佳 AUPRC
- 数据集: 157,934 样本，正类率 1.08%

## 消融模块

### 1. 判别器损失权重 (disc_weight)
**测试值**: [0.0, 0.25, 0.5, 1.0]
**默认值**: 0.5
**目的**: 验证 LSTM 判别器的因果约束引导作用
- 0.0 = 完全移除判别器
- 0.5 = 当前主模型配置
- 1.0 = 强判别器约束

### 2. 扩散步数 (diffusion_steps)
**测试值**: [50, 100, 200]
**默认值**: 100
**目的**: 验证扩散过程的充分性
- 50 = 快速扩散
- 100 = 当前主模型配置
- 200 = 精细扩散

### 3. 注意力头数 (heads)
**测试值**: [1, 4, 8]
**默认值**: 4
**目的**: 验证多头注意力的表征能力
- 1 = 单头注意力
- 4 = 当前主模型配置
- 8 = 多头注意力

### 4. 轨迹损失权重 (traj_weight)
**测试值**: [0.0, 0.5, 1.0, 2.0]
**默认值**: 1.0
**目的**: 验证轨迹预测对最终风险预测的影响
- 0.0 = 移除轨迹约束
- 1.0 = 当前主模型配置
- 2.0 = 强轨迹约束

## 启动方式

### 方式1: 启动所有消融实验
```bash
bash run_all_ablations.sh
```

### 方式2: 单独启动某个消融
```bash
python run_ablation_unified.py --ablation_type disc_weight
python run_ablation_unified.py --ablation_type diffusion_steps
python run_ablation_unified.py --ablation_type heads
python run_ablation_unified.py --ablation_type traj_weight
```

## 监控命令

```bash
tail -f logs/ablations/disc_weight.log
tail -f logs/ablations/diffusion_steps.log
tail -f logs/ablations/heads.log
tail -f logs/ablations/traj_weight.log
```

## 结果文件

- `logs/ablations/{ablation_type}.log` - 训练日志
- `logs/ablations/{ablation_type}_results.json` - 结果JSON
- `logs/ablations/{ablation_type}_run.log` - 运行日志（含错误）

## 预计完成时间

- 单个消融: 约 8-10 小时 (4个配置 × 5 seeds × 50 epochs)
- 全部4个消融: 约 32-40 小时

## 注意事项

1. 4个消融实验会并行运行，确保GPU内存充足
2. 每个消融独立运行，互不影响
3. 结果自动保存，可随时中断和恢复
