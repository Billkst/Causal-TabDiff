# TSLib 模型选择与接入方案

## 最终选择：2 个模型

### 1. iTransformer ⭐ 首选
**选择理由**:
- SOTA 性能（反转 Transformer 架构）
- 原生支持多变量时序
- 支持长短期预测
- 代码成熟，易于适配

**适配层级**: 层1 + 层2

### 2. TimeXer ⭐ 次选
**选择理由**:
- 专门处理外生变量（exogenous variables）
- 适合 landmark-conditioned 任务
- 支持多步预测
- 医疗时序数据友好

**适配层级**: 层1 + 层2

## 为什么不选其他模型

### PatchTST
- 优点：高效
- 缺点：主要优化推理速度，性能不如 iTransformer
- 结论：性能优先，不选

### TimeMixer
- 优点：多尺度混合
- 缺点：复杂度高，接入成本大
- 结论：时间有限，不选

## 接入计划

### Phase 1: 克隆与环境准备
```bash
git clone https://github.com/thuml/Time-Series-Library.git external/TSLib
```

### Phase 2: 数据适配
- 输入格式: `[Batch, Seq, Features]`
- 输出格式: `[Batch, Pred_len, 1]`
- 对齐 landmark 数据

### Phase 3: 模型封装
- 创建 `TSLibWrapper` 基类
- 实现 `iTransformerWrapper`
- 实现 `TimeXerWrapper`

### Phase 4: Smoke Test
- 数据加载测试
- 前向传播测试
- 层1输出测试
- 层2输出测试
