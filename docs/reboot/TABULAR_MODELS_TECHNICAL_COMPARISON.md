# 表格生成模型技术对比表 (2022-2024)

## 快速对比矩阵

| 模型 | 年份 | 类型 | 条件生成 | 时序支持 | 完整Pipeline | 适配难度 | 推荐度 |
|------|------|------|----------|----------|--------------|----------|--------|
| **TSDiff** | 2023 | Diffusion | ✅ | ✅ | ✅ | ⭐ 低 | ⭐⭐⭐⭐⭐ |
| **TabSyn** | 2024 | Diffusion | ✅ | ❌ | ✅ | ⭐⭐⭐ 高 | ⭐⭐⭐⭐ |
| **STaSy** | 2023 | GAN | ✅ | ❌ | ✅ | ⭐⭐ 中 | ⭐⭐⭐ |
| **TabDiff** | 2025 | Diffusion | ✅ | ❌ | ⚠️ 未开源 | ❓ 未知 | ⭐⭐⭐⭐ |
| **CTAB-GAN+** | 2022 | GAN | ✅ | ❌ | ✅ | ⭐⭐ 中 | ⭐⭐ |

---

## 详细技术分析

### 1. TSDiff - 时序扩散模型

**论文**: "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models"

**技术架构**:
```
输入: X_history (历史特征)
编码器: Transformer/S4 (结构化状态空间模型)
扩散过程: DDPM (去噪扩散概率模型)
条件机制: Cross-attention 或 Concatenation
输出: Y_trajectory (时序轨迹)
```

**关键优势**:
1. 原生时序建模，无需改造核心架构
2. 支持变长序列输入
3. 扩散模型训练稳定，生成质量高
4. 已有条件生成接口

**潜在问题**:
1. 需要验证是否支持多变量时间序列
2. 可能需要调整噪声调度适配医疗数据
3. 训练时间可能较长

**改造路径**:
```python
# 伪代码示例
model = TSDiff(
    input_dim=X_history.shape[-1],
    output_dim=risk_trajectory_dim,
    condition_encoder="transformer",
    diffusion_steps=1000
)

# 训练
model.fit(
    condition=X_history,  # (B, T_history, D)
    target=Y_trajectory   # (B, T_future, 1)
)

# 生成
risk_pred = model.sample(
    condition=X_history_test,
    num_samples=100
)
```

---

### 2. TabSyn - 表格扩散模型

**论文**: "TabSyn: Tabular Data Synthesis via Diffusion"

**技术架构**:
```
输入: X_static (静态表格)
编码器: MLP + Embedding
扩散过程: DDPM
条件机制: Class-conditional
输出: X_synthetic (合成表格)
```

**关键优势**:
1. 最新技术(2024)，生成质量SOTA
2. 支持混合数据类型(连续+离散)
3. 条件生成机制成熟
4. 代码质量高，易于理解

**潜在问题**:
1. **不支持时序**，需要大幅改造
2. 需要重新设计时序扩散过程
3. 可能需要引入时序注意力机制

**改造路径**:
```python
# 需要扩展的部分
class TemporalTabSyn(TabSyn):
    def __init__(self):
        super().__init__()
        # 添加时序编码器
        self.temporal_encoder = TransformerEncoder(...)
        # 修改扩散过程支持3D张量
        self.diffusion = TemporalDiffusion(...)
    
    def forward(self, x_history, timestep):
        # x_history: (B, T, D)
        # 需要在时间维度上应用扩散
        ...
```

**预期工作量**: 3-4周（需要深入理解扩散机制）

---

### 3. STaSy - 条件表格GAN

**论文**: "STaSy: Score-based Tabular data Synthesis"

**技术架构**:
```
输入: X_condition (条件特征)
Generator: MLP + Noise
Discriminator: MLP
条件机制: Concatenation
输出: X_synthetic (合成样本)
```

**关键优势**:
1. Van der Schaar实验室，医疗数据经验丰富
2. 条件生成机制清晰
3. 训练相对快速
4. 代码结构清晰

**潜在问题**:
1. GAN训练不稳定，可能模式崩溃
2. 需要精心设计时序判别器
3. 可能需要多次调参

**改造路径**:
```python
# 添加时序生成器
class TemporalGenerator(nn.Module):
    def __init__(self):
        self.condition_encoder = LSTM(...)  # 编码历史
        self.trajectory_decoder = LSTM(...)  # 生成轨迹
    
    def forward(self, x_history, noise):
        h = self.condition_encoder(x_history)
        trajectory = self.trajectory_decoder(h, noise)
        return trajectory

# 添加时序判别器
class TemporalDiscriminator(nn.Module):
    def __init__(self):
        self.temporal_conv = Conv1D(...)
        self.classifier = MLP(...)
    
    def forward(self, trajectory):
        features = self.temporal_conv(trajectory)
        return self.classifier(features)
```

**预期工作量**: 2-3周（需要仔细调试GAN训练）

---

## 技术选型建议

### 场景1: 快速原型验证（1-2周）
**推荐**: TSDiff
- 最小改造成本
- 直接适配时序任务
- 训练稳定

### 场景2: 追求最高生成质量（3-4周）
**推荐**: TabSyn改造
- 最新技术
- 生成质量可能最优
- 需要投入时间研究

### 场景3: 医疗数据特化（2-3周）
**推荐**: STaSy改造
- 医疗领域经验
- 代码易理解
- 需要处理GAN训练问题

---

## 风险评估

### TSDiff风险
- **低风险**: 技术成熟，改造路径清晰
- **中风险**: 可能需要调整超参数
- **缓解**: 先在小数据集上验证

### TabSyn风险
- **高风险**: 时序扩展是重大架构改动
- **中风险**: 可能引入新的训练不稳定性
- **缓解**: 参考视频扩散模型的设计

### STaSy风险
- **高风险**: GAN训练可能崩溃
- **中风险**: 时序判别器设计困难
- **缓解**: 使用Wasserstein GAN或Spectral Normalization

