# Official Risk Enrichment Results

## 状态说明

### ✅ 已完成 (Official)
- **seed42**: predictions/landmark/phase2_seed42.npz
  - 文件修改时间: 2026-03-31 14:44
  - 分析完成，结果可信

### ⏳ 待完成
- seed52, seed62, seed72, seed82 正在训练中
- 完成后需手动验证文件来源再聚合

---

## Seed42 Official Results

### 总体性能
- AUROC: 0.6931
- AUPRC: 0.0310
- Random Baseline: 0.0111
- **AUPRC 提升: 2.79x**

### 数据集统计
- 总样本数: 31,607
- 阳性样本数: 351
- 总体事件率: 1.11%

### 风险分层结果

| 风险分位 | 样本数 | 事件数 | 事件率 | 捕获率 | 富集倍数 |
|---------|--------|--------|--------|--------|----------|
| Top 1%  | 317    | 26     | 8.20%  | 7.41%  | **7.39x** |
| Top 5%  | 1,581  | 59     | 3.73%  | 16.81% | **3.36x** |
| Top 10% | 3,161  | 91     | 2.88%  | 25.93% | **2.59x** |

### 关键发现

1. **强富集效应**: Top 1% 高风险人群的事件率是总体的 7.4 倍
2. **显著优于随机**: AUPRC 比随机排序提升 2.8 倍
3. **临床实用性**: 监测 top 5% 人群可捕获 16.8% 的真实病例

### 反驳"瞎猜"的证据

- 随机排序 AUPRC = 0.0111
- 模型 AUPRC = 0.0310 (提升 2.79x)
- Top 1% 富集倍数 = 7.39x
- **结论**: 模型具有显著的风险分层能力

---

## 待办事项

等剩余 4 个 seeds 完成后：

1. 确认文件来源和参数一致性
2. 运行聚合脚本:
```bash
python scripts/aggregate_risk_enrichment.py \
  predictions/landmark/phase2_seed42.npz \
  predictions/landmark/phase2_seed52.npz \
  predictions/landmark/phase2_seed62.npz \
  predictions/landmark/phase2_seed72.npz \
  predictions/landmark/phase2_seed82.npz
```
3. 更新本文档的 5-seed 聚合结果
