# B1-3 最终纠偏报告

## 修正的两个点

### 1. evaluate_causal_and_distribution() 真正输出三个指标

**修正前**: 只返回 ate_bias
**修正后**: 返回 ate_estimate, wasserstein_distance, cmd

**当前实现**:
```python
def evaluate_causal_and_distribution(model, dataloader_real, dataloader_generated, device, output_dir=None):
    ate_estimate = compute_ate_estimate(model, dataloader_real, device)
    
    # 收集真实和生成数据
    real_data = ...
    gen_data = ...
    
    wasserstein = compute_wasserstein(real_data, gen_data)
    cmd = compute_cmd(real_data, gen_data)
    
    return {
        'ate_estimate': float(ate_estimate),
        'wasserstein_distance': float(wasserstein),
        'cmd': float(cmd)
    }
```

### 2. ATE_Bias 改名为 ATE_Estimate

**修正前**: 名为 `ATE_Bias`，但无 ground-truth 参考
**修正后**: 改名为 `ATE_Estimate`

**原因**: 
- 当前没有 ground-truth ATE 可比较
- 只是模型在两个 alpha 条件下的预测风险差
- 不是严格意义的 "bias"

**定义**:
```
ATE_Estimate = E[Risk(X, alpha=1) - Risk(X, alpha=0)]
```

这是模型估计的平均处理效应，不是与真实值的偏差。

## 三个指标的计算输入

1. **ATE_Estimate**: 
   - 输入: model, dataloader_real, alpha_low=0.0, alpha_high=1.0
   - 计算: 模型在两个 alpha 条件下预测风险的平均差

2. **Wasserstein Distance**:
   - 输入: real_data (真实数据), generated_data (生成数据)
   - 计算: 两个分布之间的 Wasserstein 距离

3. **CMD**:
   - 输入: real_data, generated_data, n_moments=5
   - 计算: 前 5 阶矩的绝对差之和

## 当前状态

✅ 所有阻塞点已清除，可进入 B2。
