# Risk Enrichment Analysis

## 目的

为 rare-event 风险预测提供"不是瞎猜"的量化证据，通过分析高风险人群中的事件富集程度来证明模型的实际价值。

## 核心指标

### 1. Random Baseline AUPRC
- **定义**: 随机排序器的 AUPRC = 数据集事件率（prevalence）
- **作用**: 提供最低性能基线
- **本项目**: ~0.011 (1.1% 事件率)

### 2. Enrichment (富集倍数)
- **定义**: Top X% 高风险人群的事件率 / 总体事件率
- **解释**: 
  - 1.0x = 和随机选择一样
  - 5.0x = 高风险人群的事件率是总体的 5 倍
  - 越高越好，说明模型能有效识别高风险个体

### 3. Capture Rate (捕获率)
- **定义**: Top X% 捕获的真实阳性数 / 总阳性数
- **解释**: 如果只关注 top 1% 高风险人群，能捕获多少真实病例

## 当前结果 (seed42 only - Official)

| 风险分位 | 样本数 | 事件数 | 事件率 | 捕获率 | 富集倍数 |
|---------|--------|--------|--------|--------|----------|
| Top 1%  | 317 | 26 | 8.20% | 7.41% | **7.39x** |
| Top 5%  | 1,581 | 59 | 3.73% | 16.81% | **3.36x** |
| Top 10% | 3,161 | 91 | 2.88% | 25.93% | **2.59x** |

**关键证据** (seed42):
- AUPRC: 0.0310 vs Random 0.0111 → **2.79x 提升**
- Top 1% 富集倍数: **7.39x** → 高风险人群事件率是总体的 7.4 倍
- 结论: **模型具有显著的风险分层能力**

**注**: 5-seed 聚合结果需等待剩余 4 个 seeds 正式完成后手动确认文件来源再生成。

## 使用方法

### 单个 seed 分析
```bash
conda activate causal_tabdiff
python scripts/analyze_risk_enrichment.py
```

输出:
- `reports/risk_enrichment/phase2_seed42_enrichment.csv`
- `reports/risk_enrichment/phase2_seed42_enrichment.txt`

### 多 seed 聚合分析

**必须显式传入文件路径**（防止误用历史文件）：

```bash
conda activate causal_tabdiff
python scripts/aggregate_risk_enrichment.py \
  predictions/landmark/phase2_seed42.npz \
  predictions/landmark/phase2_seed52.npz \
  predictions/landmark/phase2_seed62.npz \
  predictions/landmark/phase2_seed72.npz \
  predictions/landmark/phase2_seed82.npz
```

脚本会：
1. 显示每个文件的 mtime、seed、best_epoch、oversample_ratio、focal_gamma
2. 要求用户确认这些文件都是当前 official run 的结果
3. 确认后才执行聚合

输出:
- `reports/risk_enrichment/phase2_aggregated_enrichment.csv`
- `reports/risk_enrichment/phase2_aggregated_enrichment.txt`

### 自定义分析

修改脚本中的参数：

```python
# 分析不同的风险分位
percentiles = [1, 2, 5, 10, 20]

# 分析不同的 seed
seeds = [42, 52, 62, 72, 82]

# 分析不同阶段的结果
npz_paths = [f"predictions/landmark/phase1_seed{s}.npz" for s in seeds]
```

## 如何反驳"和瞎猜没区别"

### 证据 1: AUPRC 提升
- 随机排序: AUPRC = 0.011
- 我们的模型: AUPRC = 0.026
- **提升 2.4 倍**

### 证据 2: 风险分层能力
- Top 1% 高风险人群的事件率: 5.88%
- 总体事件率: 1.09%
- **富集 5.4 倍**

### 证据 3: 临床实用性
- 如果只监测 top 5% 高风险人群（1,581 人）
- 可以捕获 17.2% 的真实病例（59 例）
- 事件率提升到 3.73%（总体 1.09%）
- **监测效率提升 3.4 倍**

## 文件说明

- `analyze_risk_enrichment.py`: 单 seed 分析脚本
- `aggregate_risk_enrichment.py`: 多 seed 聚合脚本
- `phase2_seed42_enrichment.csv`: seed42 详细结果
- `phase2_aggregated_enrichment.csv`: 5-seed 聚合结果
- `*.txt`: 完整文本报告
