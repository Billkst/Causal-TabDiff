# 表格生成模型调研总结 (2022-2024)

**调研日期**: 2026-03-11  
**调研目标**: 寻找支持条件生成和时序建模的表格生成模型，用于轨迹生成任务

---

## 核心发现

### ✅ 已识别的高优先级模型

| 模型 | 年份 | 类型 | 条件生成 | 时序支持 | 推荐度 | 仓库 |
|------|------|------|----------|----------|--------|------|
| **TSDiff** | 2023 | Diffusion | ✅ | ✅ | ⭐⭐⭐⭐⭐ | [amazon-science/unconditional-time-series-diffusion](https://github.com/amazon-science/unconditional-time-series-diffusion) |
| **TabSyn** | 2024 | Diffusion | ✅ | ❌ | ⭐⭐⭐⭐ | [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn) |
| **STaSy** | 2023 | GAN | ✅ | ❌ | ⭐⭐⭐ | [vanderschaarlab/STaSy](https://github.com/vanderschaarlab/STaSy) |

---

## 推荐方案

### 🥇 方案A: 直接适配TSDiff（强烈推荐）

**理由**:
- 原生支持时序+条件生成
- 改造成本最低（预计1-2周）
- 扩散模型训练稳定

**改造路径**:
```
输入: X_history (T0-T2的历史特征)
条件编码: Transformer/LSTM
扩散生成: DDPM
输出: risk_trajectory (未来2年风险轨迹)
```

**下一步**: 克隆仓库验证可行性

---

### 🥈 方案B: 改造TabSyn支持时序（备选）

**理由**:
- 最新技术(2024)
- 生成质量可能最优
- 需要较大改造（预计3-4周）

**改造路径**:
- 扩展扩散过程到时序维度
- 添加时序注意力机制
- 重新设计噪声调度

**下一步**: 评估架构复杂度

---

### 🥉 方案C: 改造STaSy支持时序（保守）

**理由**:
- 医疗数据经验丰富
- 代码相对简单
- GAN训练可能不稳定（预计2-3周）

**改造路径**:
- 添加LSTM/Transformer编码器
- 设计时序判别器
- 使用WGAN或Spectral Norm稳定训练

---

## 立即行动项

### 第1步: 验证TSDiff（1-2天）
```bash
cd /tmp
git clone https://github.com/amazon-science/unconditional-time-series-diffusion
cd unconditional-time-series-diffusion
# 阅读README和代码
# 运行示例
# 输出可行性报告
```

### 第2步: 评估TabSyn（1天）
```bash
cd /tmp
git clone https://github.com/amazon-science/tabsyn
cd tabsyn
# 理解扩散机制
# 评估时序扩展难度
```

### 第3步: 搜索最新论文（半天）
- arXiv: `longitudinal tabular generation 2024`
- Papers with Code: `temporal tabular synthesis`
- 确保没有遗漏最新模型

---

## 技术对比

### 训练稳定性
- **Diffusion (TSDiff/TabSyn)**: ⭐⭐⭐⭐⭐ 非常稳定
- **GAN (STaSy)**: ⭐⭐⭐ 需要仔细调参

### 生成质量
- **TSDiff**: ⭐⭐⭐⭐ 时序生成质量高
- **TabSyn**: ⭐⭐⭐⭐⭐ 表格生成SOTA
- **STaSy**: ⭐⭐⭐ 中等

### 改造难度
- **TSDiff**: ⭐ 低（直接适配）
- **TabSyn**: ⭐⭐⭐ 高（需要重新设计）
- **STaSy**: ⭐⭐ 中（添加时序模块）

---

## 风险评估

### TSDiff风险
- **技术风险**: 低
- **时间风险**: 低
- **缓解措施**: 先在小数据集验证

### TabSyn风险
- **技术风险**: 中高（时序扩展是重大改动）
- **时间风险**: 中
- **缓解措施**: 参考视频扩散模型

### STaSy风险
- **技术风险**: 中（GAN训练不稳定）
- **时间风险**: 中
- **缓解措施**: 使用WGAN-GP

---

## 决策建议

**建议优先尝试TSDiff**，理由如下：
1. 技术路径最清晰
2. 改造成本最低
3. 训练最稳定
4. 如果1周内验证可行，立即开始实施
5. 如果发现不适配，立即转向TabSyn

**决策时间点**: 2026-03-18（1周后）

---

## 相关文档

- [详细调研报告](./TABULAR_MODELS_SURVEY_2024.md)
- [技术对比分析](./TABULAR_MODELS_TECHNICAL_COMPARISON.md)
- [下一步行动计划](./NEXT_STEPS_MODEL_EVALUATION.md)
- [基线兼容性报告](./BASELINE_COMPATIBILITY_REPORT.md)

