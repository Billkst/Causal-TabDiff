# 指标参考指南执行总结

**生成日期**: 2026-03-14  
**完整文档**: `METRICS_REFERENCE_GUIDE.md` (679 行)

---

## 📋 任务完成情况

✅ **已完成所有要求的内容**：

1. ✅ 收集了 AUROC, AUPRC, F1, Precision, Recall 的权威定义
2. ✅ 解释了阈值选择策略（验证集优化 + 测试集固定评估）
3. ✅ 说明了轨迹保真度和判别能力为何是独立维度
4. ✅ 提供了效率指标（参数量、推理时间）的报告规范
5. ✅ 强调了不平衡医疗数据的特殊考量
6. ✅ 提供了论文写作的完整示例

---

## 🎯 核心要点速览

### 1. 不平衡医疗数据的指标优先级

**推荐顺序**（患病率 <5% 时）：
1. **AUPRC** (首要指标) - 对少数正类敏感
2. **AUROC** (辅助指标) - 提供整体排序能力
3. **F1 @ 验证集最优阈值** - 平衡精确率和召回率

**关键论点**：
> "在高度不平衡数据（正类比例 <5%）中，AUROC 可能因大量负样本而虚高（如 0.9），但 AUPRC 可能仅 0.3。AUPRC 更真实反映模型对罕见正类的识别能力。"

**权威支持**：
- Saito & Rehmsmeier (2015). *PLoS ONE*
- scikit-learn 官方文档

---

### 2. 阈值选择的黄金法则

**标准流程**：
```
训练集 → 训练模型
验证集 → 搜索最优阈值（最大化 F1）
测试集 → 应用固定阈值（不再调整）
```

**为什么不能在测试集上调阈值？**
- ❌ 导致性能估计过于乐观（数据泄漏）
- ❌ 无法模拟真实部署（临床中阈值需预先确定）

**论文表述模板**：
> "We selected the classification threshold by maximizing F1 score on the validation set. This threshold was then fixed and applied to the test set to obtain unbiased performance estimates."

---

### 3. 生成模型的双维度评估

**维度 1：轨迹保真度（Trajectory Fidelity）**
- **测量什么**：生成数据的统计分布与真实数据的相似度
- **常用指标**：Wasserstein Distance, CMD
- **回答问题**："生成的患者轨迹看起来像真实数据吗？"

**维度 2：判别能力（Discriminative Performance）**
- **测量什么**：在生成数据上训练的模型在真实数据上的性能
- **常用指标**：TSTR AUROC/AUPRC
- **回答问题**："生成数据包含足够的预测信号吗？"

**为什么是独立维度？**
- 高保真 ≠ 高判别：可能完美复制边缘分布，但丢失因果关联
- 低保真 ≠ 低判别：可能某些特征有偏差，但保留关键预测特征

**论文表述模板**：
> "Trajectory fidelity and discriminative performance measure distinct aspects of generation quality. High fidelity ensures statistical realism, while high discriminative performance confirms that generated data retains predictive signals necessary for downstream tasks."

---

### 4. 效率指标报告规范

**参数量（Parameter Count）**：
```python
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {params/1e6:.2f}M")
```

**推理时间（Inference Time）**：
```python
# 1. 预热 20 次
# 2. 测量 1000 次
# 3. 报告均值 ± 标准差（单位：ms/sample）
```

**论文表述**：
> "We report average inference time per sample (in milliseconds) measured on [GPU型号], which is critical for assessing real-time deployment feasibility."

---

## 📚 权威参考文献汇总

### 核心方法论

1. **AUPRC vs AUROC**：
   - Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*, 10(3), e0118432.

2. **阈值选择**：
   - Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating*. Springer.

3. **生成模型评估**：
   - Yoon, J., Jarrett, D., & van der Schaar, M. (2019). "Time-series Generative Adversarial Networks." *NeurIPS*.
   - Esteban, C., et al. (2017). "Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs." *arXiv:1706.02633*.

4. **医疗 AI 指南**：
   - de Hond, A. A. H., et al. (2022). "Guidelines and quality criteria for artificial intelligence-based prediction models in healthcare: a scoping review." *npj Digital Medicine*, 5(1), 2.

### 官方文档

- scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html
- imbalanced-learn: https://imbalanced-learn.org/stable/

---

## 🚨 常见错误与避免方法

### ❌ 错误 1：在测试集上调整阈值
**后果**：性能估计过于乐观，无法反映真实部署效果  
**正确做法**：在验证集上选定阈值，测试集仅应用

### ❌ 错误 2：仅报告 AUROC 而忽略 AUPRC
**后果**：在不平衡数据上给出误导性的高性能  
**正确做法**：同时报告 AUROC 和 AUPRC，以 AUPRC 为主要指标

### ❌ 错误 3：不说明阈值选择方法
**后果**：审稿人无法判断结果的可靠性  
**正确做法**：明确说明"在验证集上最大化 F1"

### ❌ 错误 4：混淆轨迹保真度和判别能力
**后果**：无法全面评估生成模型质量  
**正确做法**：分别报告两个维度，说明它们的独立性

---

## 📊 论文写作快速模板

### 方法部分（Metrics）

```markdown
### Evaluation Metrics

**Threshold-independent metrics:**
- AUROC: Measures ranking ability across all thresholds
- AUPRC: Measures precision-recall trade-off (primary metric for imbalanced data)

**Threshold-dependent metrics:**
We select the threshold by maximizing F1 on validation set, then apply it to test set:
- F1 Score: Harmonic mean of precision and recall
- Precision: Proportion of predicted positives that are true positives
- Recall: Proportion of true positives correctly identified

**Generative model metrics:**
- Trajectory Fidelity: Wasserstein distance and CMD
- Discriminative Performance: TSTR AUROC/AUPRC

**Efficiency metrics:**
- Parameter count (millions)
- Inference time (ms/sample)
```

### 结果表格标题

```markdown
Table X: Performance comparison on [Dataset] (positive rate: X.X%). 
Threshold-independent metrics (AUROC, AUPRC) and threshold-dependent 
metrics (F1, Precision, Recall) at validation-optimized threshold. 
Mean ± std over N seeds. Best results in bold.
```

---

## 🔍 项目特定建议

基于您的项目代码（`src/evaluation/metrics.py`），当前实现已包含：

✅ `compute_ranking_metrics()` - AUROC, AUPRC  
✅ `find_optimal_threshold()` - 验证集阈值优化  
✅ `compute_threshold_metrics()` - F1, Precision, Recall  
✅ `compute_calibration_metrics()` - Brier Score, 校准斜率

**建议补充**：
1. 在 `find_optimal_threshold()` 中添加注释说明"仅用于验证集"
2. 在评估脚本中明确区分验证集和测试集的阈值使用
3. 在结果报告中添加正类比例（prevalence）

---

## 📖 完整文档位置

详细的定义、公式、代码示例和引用请参考：
**`/home/UserData/ljx/Project_2/Causal-TabDiff/docs/METRICS_REFERENCE_GUIDE.md`**

包含内容：
- 10 个章节，679 行
- 每个指标的数学定义、适用场景、局限性
- 完整的论文写作示例
- 权威参考文献列表
- 快速查询表

---

**生成工具**: Claude (Kiro)  
**数据来源**: scikit-learn 官方文档, Nature Digital Medicine, NeurIPS, PLoS ONE
