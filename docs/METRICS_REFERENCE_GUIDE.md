# 医疗风险预测与生成评估指标权威参考指南

**版本**: v1.0  
**日期**: 2026-03-14  
**目标**: 为论文写作提供权威、简洁的指标定义和应用场景说明

---

## 1. 核心分类指标（Classification Metrics）

### 1.1 AUROC (Area Under the Receiver Operating Characteristic Curve)

**定义**：
AUROC 衡量分类器在所有可能阈值下区分正负样本的能力。ROC 曲线以假阳性率（FPR）为横轴、真阳性率（TPR/Recall）为纵轴，AUROC 为曲线下面积。

**数学表达**：
$$\text{AUROC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

其中 $\text{TPR} = \frac{TP}{TP + FN}$，$\text{FPR} = \frac{FP}{FP + TN}$

**取值范围**: [0.5, 1.0]，0.5 表示随机猜测，1.0 表示完美分类

**适用场景**：
- **阈值无关评估**：当不确定最优决策阈值时，AUROC 提供整体排序能力的度量
- **类别平衡或轻度不平衡**：AUROC 对类别分布相对稳健
- **医疗筛查初筛**：评估模型是否能有效将高风险患者排在前列

**局限性**：
- **高度不平衡数据**：在正类极少（如患病率 <5%）的场景下，AUROC 可能过于乐观，因为大量负样本会主导指标
- **不反映校准**：AUROC 仅评估排序，不评估预测概率的准确性

**权威参考**：
- scikit-learn 官方文档: [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- Fawcett, T. (2006). "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861-874.

**论文写作建议**：
> "We report AUROC to assess the model's threshold-independent ranking ability across all patients. AUROC measures the probability that a randomly chosen positive case is ranked higher than a randomly chosen negative case."

---

### 1.2 AUPRC (Area Under the Precision-Recall Curve)

**定义**：
AUPRC 衡量在不同阈值下精确率（Precision）和召回率（Recall）的权衡。PR 曲线以召回率为横轴、精确率为纵轴，AUPRC 为曲线下面积。

**数学表达**：
$$\text{AUPRC} = \int_0^1 \text{Precision}(r) \, dr$$

其中 $\text{Precision} = \frac{TP}{TP + FP}$，$\text{Recall} = \frac{TP}{TP + FN}$

**取值范围**: [prevalence, 1.0]，基线为正类比例（prevalence）

**适用场景**：
- **高度不平衡数据**：AUPRC 对少数正类更敏感，是不平衡医疗数据的首选指标
- **医疗风险预测**：当漏诊（FN）和误诊（FP）成本都很高时
- **罕见疾病检测**：患病率 <10% 的场景

**优势**：
- **聚焦正类**：不受大量负样本影响，直接反映模型对少数正类的识别能力
- **临床相关性**：Precision 和 Recall 都是临床决策的关键指标

**权威参考**：
- scikit-learn 官方文档: [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves." *ICML*.
- Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*, 10(3), e0118432.

**论文写作建议**：
> "Given the class imbalance (positive rate: X%), we prioritize AUPRC over AUROC. AUPRC directly measures the trade-off between precision (positive predictive value) and recall (sensitivity), which are critical for clinical decision-making in rare disease prediction."

---

### 1.3 F1 Score

**定义**：
F1 分数是精确率和召回率的调和平均数，在固定阈值下综合评估模型性能。

**数学表达**：
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**取值范围**: [0, 1]，1 表示完美分类

**适用场景**：
- **需要单一指标**：当需要在精确率和召回率之间平衡时
- **阈值已确定**：在验证集上选定阈值后，在测试集上评估
- **不平衡数据**：F1 对正类更敏感，适合少数类评估

**关键注意事项**：
- **阈值依赖**：F1 分数依赖于分类阈值的选择，必须明确说明阈值如何确定
- **不对称性**：F1 对假阴性（FN）和假阳性（FP）的惩罚不对称

**权威参考**：
- scikit-learn 官方文档: [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
- Lipton, Z. C., et al. (2014). "Optimal thresholding of classifiers to maximize F1 measure." *ECML PKDD*.

**论文写作建议**：
> "We report F1 score at a threshold optimized on the validation set to balance precision and recall. The threshold was selected to maximize F1 on validation data and then applied to the test set without further tuning."

---

### 1.4 Precision (Positive Predictive Value, PPV)

**定义**：
精确率衡量模型预测为正类的样本中，真正为正类的比例。

**数学表达**：
$$\text{Precision} = \frac{TP}{TP + FP}$$

**临床意义**：
- **回答问题**："如果模型预测患者为高风险，患者真正高风险的概率是多少？"
- **成本考量**：高精确率减少不必要的干预和患者焦虑

**权威参考**：
- scikit-learn 官方文档: [sklearn.metrics.precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

**论文写作建议**：
> "Precision (positive predictive value) indicates the proportion of patients flagged as high-risk who truly are at high risk, directly informing the efficiency of clinical interventions."

---

### 1.5 Recall (Sensitivity, True Positive Rate)

**定义**：
召回率衡量所有真正为正类的样本中，被模型正确识别的比例。

**数学表达**：
$$\text{Recall} = \frac{TP}{TP + FN}$$

**临床意义**：
- **回答问题**："在所有真正高风险的患者中，模型能识别出多少？"
- **成本考量**：高召回率减少漏诊，对于严重疾病至关重要

**权威参考**：
- scikit-learn 官方文档: [sklearn.metrics.recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

**论文写作建议**：
> "Recall (sensitivity) measures the proportion of true high-risk patients correctly identified by the model, which is critical for minimizing missed diagnoses in life-threatening conditions."

---

## 2. 阈值选择策略（Threshold Selection）

### 2.1 验证集阈值优化 + 测试集固定评估

**标准流程**：
1. **训练阶段**：在训练集上训练模型，输出概率预测
2. **验证阶段**：在验证集上搜索最优阈值（如最大化 F1 或平衡准确率）
3. **测试阶段**：将验证集确定的阈值应用于测试集，**不再调整**

**理论依据**：
- **避免过拟合**：在测试集上调整阈值会导致性能估计过于乐观
- **模拟真实部署**：临床部署时阈值需预先确定，不能根据测试数据调整

**权威参考**：
- Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

**论文写作建议**：
> "We selected the classification threshold by maximizing F1 score on the validation set. This threshold was then fixed and applied to the test set to obtain unbiased performance estimates, simulating real-world deployment where thresholds must be predetermined."

---

### 2.2 Prevalence-Aware 阈值

**定义**：
根据数据集的正类比例（prevalence）设置阈值，使预测的正类比例与真实比例一致。

**计算方法**：
$$\text{threshold} = \text{quantile}(y_{\text{pred\_proba}}, 1 - \text{prevalence})$$

**适用场景**：
- **生成模型评估**：评估生成数据是否保持了真实数据的类别分布
- **公平性约束**：确保不同子群体的预测正类比例一致

**论文写作建议**：
> "To assess whether the generated data preserves the outcome distribution, we compute F1 score at a prevalence-aware threshold, where the threshold is set such that the predicted positive rate matches the true prevalence in the real data."

---

## 3. 生成模型特定指标

### 3.1 轨迹保真度（Trajectory Fidelity）

**定义**：
评估生成的时间序列轨迹在统计分布和时间动态上与真实数据的相似度。

**常用指标**：
- **Wasserstein Distance**：衡量两个分布之间的"搬运成本"
  $$W(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$
  
- **Central Moment Discrepancy (CMD)**：比较高阶矩差异
  $$\text{CMD} = \sum_{k=1}^K |\mathbb{E}[X_{\text{real}}^k] - \mathbb{E}[X_{\text{gen}}^k]|$$

**临床意义**：
- **纵向一致性**：生成的患者轨迹是否符合疾病进展的自然规律
- **时间依赖性**：是否保留了真实数据中的时间相关性

**权威参考**：
- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). "Time-series Generative Adversarial Networks." *NeurIPS*.
- Esteban, C., et al. (2017). "Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs." *arXiv:1706.02633*.

**论文写作建议**：
> "We assess trajectory fidelity using Wasserstein distance and Central Moment Discrepancy to ensure that generated longitudinal patient data preserves both distributional properties and temporal dynamics of real clinical trajectories."

---

### 3.2 判别能力（Discriminative Performance）

**定义**：
评估在生成数据上训练的下游预测模型在真实数据上的性能（TSTR: Train on Synthetic, Test on Real）。

**评估协议**：
1. 在生成数据上训练预测模型（如 XGBoost）
2. 在真实测试集上评估 AUROC、AUPRC、F1
3. 与在真实训练数据上训练的模型对比

**临床意义**：
- **实用性验证**：生成数据是否能用于训练有效的临床预测模型
- **隐私保护**：评估合成数据能否替代真实数据用于模型开发

**权威参考**：
- Xu, L., et al. (2019). "Modeling Tabular data using Conditional GAN." *NeurIPS*.
- Choi, E., et al. (2017). "Generating Multi-label Discrete Patient Records using Generative Adversarial Networks." *MLHC*.

**论文写作建议**：
> "We evaluate discriminative performance via the TSTR (Train on Synthetic, Test on Real) protocol: a downstream risk prediction model is trained on generated data and evaluated on real test data. This assesses whether synthetic trajectories retain sufficient predictive signal for clinical applications."

---

### 3.3 为什么轨迹保真度和判别能力是独立维度

**核心论点**：
- **轨迹保真度**：评估生成数据的**统计真实性**（是否像真实数据）
- **判别能力**：评估生成数据的**任务相关性**（是否对下游任务有用）

**可能的解耦场景**：
1. **高保真但低判别**：生成数据完美复制了真实分布，但丢失了与结局相关的微妙模式
2. **低保真但高判别**：生成数据分布有偏差，但保留了预测关键特征

**论文写作建议**：
> "Trajectory fidelity and discriminative performance measure distinct aspects of generation quality. High fidelity ensures statistical realism of longitudinal patterns, while strong discriminative performance confirms that outcome-relevant predictive signals are preserved. A model may achieve one without the other: for instance, a generator might produce realistic-looking trajectories that nonetheless fail to capture subtle risk markers, or conversely, preserve predictive features while distorting marginal distributions."

---

## 4. 效率指标（Efficiency Metrics）

### 4.1 参数量（Parameter Count）

**定义**：
模型中可训练参数的总数，通常以百万（M）为单位报告。

**计算方法**（PyTorch）：
```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**临床意义**：
- **部署可行性**：参数量影响模型的存储和计算需求
- **过拟合风险**：参数量与样本量的比例影响泛化能力

**论文写作建议**：
> "We report the number of trainable parameters (in millions) to characterize model complexity and computational requirements for clinical deployment."

---

### 4.2 推理时间（Inference Time）

**定义**：
模型对单个样本进行预测所需的平均时间，通常以毫秒（ms）为单位。

**测量协议**：
1. 在目标硬件（如 CPU/GPU）上运行
2. 预热阶段（warm-up）：运行 10-50 次以稳定缓存
3. 测量阶段：对 N 个样本（如 N=1000）计时，取平均值
4. 报告均值 ± 标准差

**临床意义**：
- **实时性要求**：急诊场景需要秒级响应
- **批量处理**：大规模筛查可接受分钟级延迟

**论文写作建议**：
> "We measure average inference time per sample on [hardware specification] after a warm-up phase of 50 iterations. Inference time is critical for real-time clinical decision support systems."

---

## 5. 不平衡医疗数据的特殊考量

### 5.1 为什么 AUPRC 优于 AUROC

**理论依据**：
- **基线差异**：AUROC 的随机基线为 0.5，AUPRC 的基线为正类比例（prevalence）
- **敏感性**：在正类极少时，AUROC 对模型改进不敏感，而 AUPRC 能显著区分模型优劣

**实证证据**：
- Saito & Rehmsmeier (2015) 证明：在不平衡数据上，AUPRC 比 AUROC 更能反映模型的实际临床价值

**论文写作建议**：
> "In highly imbalanced medical datasets (positive rate <5%), AUPRC is more informative than AUROC because it focuses on the minority positive class and is not inflated by the large number of true negatives. We therefore prioritize AUPRC as our primary metric."

---

### 5.2 报告完整的性能画像

**推荐报告内容**：
1. **阈值无关指标**：AUROC, AUPRC
2. **固定阈值指标**：F1, Precision, Recall, Specificity（明确说明阈值选择方法）
3. **校准指标**：Brier Score, Calibration Slope/Intercept
4. **混淆矩阵**：TP, FP, TN, FN 的绝对数量

**论文写作建议**：
> "We report a comprehensive performance profile including threshold-independent metrics (AUROC, AUPRC), threshold-dependent metrics at a validation-optimized threshold (F1, Precision, Recall), and calibration metrics (Brier score). This multi-faceted evaluation ensures transparency about model behavior across different operating points."

---

## 6. 引用建议

### 6.1 官方文档引用

**scikit-learn**:
> Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.

**imbalanced-learn**:
> Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). "Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning." *Journal of Machine Learning Research*, 18(17), 1-5.

### 6.2 方法学引用

**不平衡数据评估**:
- Saito, T., & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*, 10(3), e0118432.

**临床预测模型**:
- Steyerberg, E. W., et al. (2010). "Assessing the performance of prediction models: a framework for traditional and novel measures." *Epidemiology*, 21(1), 128-138.

**医疗 AI 指南**:
- de Hond, A. A. H., et al. (2022). "Guidelines and quality criteria for artificial intelligence-based prediction models in healthcare: a scoping review." *npj Digital Medicine*, 5(1), 2.

---

## 7. 快速查询表

| 指标 | 适用场景 | 阈值依赖 | 不平衡敏感性 | 临床解释 |
|------|---------|---------|------------|---------|
| **AUROC** | 整体排序能力 | 否 | 低 | 随机选两个患者，高风险者得分更高的概率 |
| **AUPRC** | 不平衡数据 | 否 | 高 | 在不同召回率下的平均精确率 |
| **F1** | 平衡精确率和召回率 | 是 | 中 | 精确率和召回率的调和平均 |
| **Precision** | 减少假阳性 | 是 | 中 | 预测为阳性中真阳性的比例 |
| **Recall** | 减少假阴性 | 是 | 中 | 真阳性中被识别出的比例 |
| **Wasserstein** | 分布相似度 | N/A | N/A | 两分布间的"搬运距离" |
| **TSTR AUROC** | 生成数据实用性 | 否 | 低 | 合成数据训练模型的判别能力 |

---

## 8. 常见错误与避免方法

### 错误 1：在测试集上调整阈值
**问题**：导致性能估计过于乐观  
**正确做法**：仅在验证集上选择阈值，测试集使用固定阈值

### 错误 2：仅报告 AUROC 而忽略 AUPRC
**问题**：在不平衡数据上 AUROC 可能误导  
**正确做法**：同时报告 AUROC 和 AUPRC，优先解释 AUPRC

### 错误 3：不说明阈值选择方法
**问题**：F1/Precision/Recall 无法复现  
**正确做法**：明确说明阈值是如何确定的（如"验证集 F1 最大化"）

### 错误 4：混淆轨迹保真度和判别能力
**问题**：认为分布相似就意味着预测有用  
**正确做法**：分别评估并报告两个维度

---

## 参考文献

1. **scikit-learn Documentation**: https://scikit-learn.org/stable/modules/model_evaluation.html
2. **Saito & Rehmsmeier (2015)**: "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*.
3. **Steyerberg (2019)**: *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating*. Springer.
4. **de Hond et al. (2022)**: "Guidelines and quality criteria for artificial intelligence-based prediction models in healthcare: a scoping review." *npj Digital Medicine*.
5. **Yoon et al. (2019)**: "Time-series Generative Adversarial Networks." *NeurIPS*.
6. **Davis & Goadrich (2006)**: "The relationship between Precision-Recall and ROC curves." *ICML*.

---

**文档维护**：本指南应随项目进展和新文献发布定期更新。

### 3.1 轨迹保真度（Trajectory Fidelity）

**定义**：
评估生成的时间序列轨迹在统计分布和时间动态上与真实轨迹的相似度。

**常用指标**：
- **Wasserstein Distance**：衡量两个分布之间的"搬运成本"
  $$W(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$
  
- **Central Moment Discrepancy (CMD)**：比较高阶矩差异
  $$\text{CMD} = \sum_{k=1}^{K} |\mathbb{E}[X_{\text{real}}^k] - \mathbb{E}[X_{\text{gen}}^k]|$$

**适用场景**：
- **时间序列生成**：评估生成的患者轨迹是否保持了真实的时间演化模式
- **分布匹配**：验证生成数据的边缘分布和联合分布

**权威参考**：
- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). "Time-series generative adversarial networks." *NeurIPS*.
- Zhu, J., et al. (2017). "Maximum mean discrepancy for generative models." *arXiv:1705.08584*.

**论文写作建议**：
> "We assess trajectory fidelity using Wasserstein distance and Central Moment Discrepancy (CMD) to ensure that generated patient trajectories preserve both the marginal distributions and temporal dynamics of real data."

---

### 3.2 判别器性能（Discriminative Performance）

**定义**：
评估在生成数据上训练的预测模型在真实数据上的性能，验证生成数据的下游任务有效性。

**评估协议**：
- **TSTR (Train on Synthetic, Test on Real)**：在生成数据上训练分类器，在真实测试集上评估
- **TRTS (Train on Real, Test on Synthetic)**：在真实数据上训练，在生成数据上测试（较少使用）

**关键指标**：
- TSTR AUROC/AUPRC：判别器在真实数据上的性能
- 与 TRTR (Train on Real, Test on Real) 的性能差距

**适用场景**：
- **数据增强验证**：评估生成数据是否能有效扩充训练集
- **隐私保护数据共享**：验证合成数据的实用性

**权威参考**：
- Esteban, C., et al. (2017). "Real-valued (medical) time series generation with recurrent conditional GANs." *arXiv:1706.02633*.
- Xu, L., et al. (2019). "Modeling tabular data using conditional GAN." *NeurIPS*.

**论文写作建议**：
> "We evaluate discriminative performance via the TSTR protocol: training a risk prediction model on generated data and testing on real held-out data. High TSTR AUROC indicates that the generated data captures the predictive patterns necessary for downstream clinical tasks."

---

### 3.3 为什么轨迹保真度和判别性能是独立维度

**核心论点**：
1. **轨迹保真度**关注**数据分布的统计相似性**，回答"生成的数据看起来像真实数据吗？"
2. **判别性能**关注**任务相关的预测信号**，回答"生成的数据包含足够的预测信息吗？"

**可能的解耦场景**：
- **高保真但低判别**：生成数据完美复制了边缘分布，但丢失了结果变量的因果关联
- **低保真但高判别**：生成数据在某些特征上有偏差，但保留了关键的预测特征

**论文写作建议**：
> "Trajectory fidelity and discriminative performance measure distinct aspects of generation quality. High fidelity ensures statistical realism of patient trajectories, while high discriminative performance confirms that the generated data retains the predictive signals necessary for risk modeling. Both dimensions are essential: fidelity for clinical plausibility and discriminative performance for downstream utility."

---

## 4. 效率指标（Efficiency Metrics）

### 4.1 参数量（Parameter Count）

**定义**：
模型中可训练参数的总数，通常以百万（M）为单位报告。

**计算方法**（PyTorch）：
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**临床意义**：
- **部署成本**：参数量影响模型的存储和内存需求
- **可解释性**：参数量与模型复杂度相关

**论文写作建议**：
> "We report the number of trainable parameters (in millions) to characterize model complexity and deployment feasibility in resource-constrained clinical settings."

---

### 4.2 推理时间（Inference Time）

**定义**：
模型对单个样本进行预测所需的平均时间，通常以毫秒（ms）为单位报告。

**测量协议**：
1. **预热**：运行 10-20 次推理以稳定 GPU/CPU 状态
2. **批量测试**：对 N 个样本（如 N=1000）进行推理
3. **平均计算**：总时间 / N，报告均值和标准差

**代码示例**（PyTorch）：
```python
import time
import torch

model.eval()
times = []
with torch.no_grad():
    # 预热
    for _ in range(20):
        _ = model(sample_input)
    
    # 测量
    for _ in range(1000):
        start = time.time()
        _ = model(sample_input)
        torch.cuda.synchronize()  # 如果使用 GPU
        times.append((time.time() - start) * 1000)  # 转换为 ms

avg_time = np.mean(times)
std_time = np.std(times)
```

**临床意义**：
- **实时性要求**：急诊场景需要秒级响应
- **吞吐量**：影响系统能同时服务的患者数量

**论文写作建议**：
> "We report average inference time per sample (in milliseconds) measured on [hardware specification], which is critical for assessing real-time deployment feasibility in clinical workflows."

---

## 5. 不平衡医疗数据的特殊考量

### 5.1 为什么 AUPRC 优于 AUROC

**理论依据**：
在高度不平衡数据（如患病率 <5%）中：
- **AUROC 的乐观偏差**：大量负样本使得 FPR 分母很大，即使 FP 较多，FPR 仍然很小，导致 AUROC 虚高
- **AUPRC 的敏感性**：Precision 的分母是 TP+FP，直接受 FP 影响，对少数正类的预测质量更敏感

**实证证据**：
- Saito & Rehmsmeier (2015) 通过实验表明，在不平衡数据上，AUROC 可能高达 0.9，但 AUPRC 仅 0.3，后者更真实反映模型性能

**论文写作建议**：
> "In highly imbalanced medical datasets (positive rate: X%), AUROC can be misleadingly optimistic due to the dominance of negative samples. We therefore prioritize AUPRC, which directly measures precision-recall trade-offs and is more sensitive to the model's ability to identify rare positive cases."

---

### 5.2 类别不平衡的报告规范

**必须报告的信息**：
1. **正类比例**：训练集、验证集、测试集的患病率
2. **基线性能**：随机猜测的 AUPRC（等于正类比例）
3. **指标选择理由**：为什么选择 AUPRC 而非 AUROC

**示例表述**：
> "Our dataset exhibits severe class imbalance with a positive rate of 3.2% (training), 3.1% (validation), and 3.3% (test). The baseline AUPRC for a random classifier is 0.032. We report both AUROC and AUPRC, with AUPRC as the primary metric due to its sensitivity to minority class performance."

---

## 6. 论文写作的完整示例

### 6.1 方法部分（Evaluation Metrics）

```markdown
### Evaluation Metrics

We evaluate model performance using both threshold-independent and threshold-dependent metrics:

**Threshold-independent metrics:**
- **AUROC** (Area Under the ROC Curve): Measures the model's ability to rank positive cases higher than negative cases across all thresholds.
- **AUPRC** (Area Under the Precision-Recall Curve): Measures the precision-recall trade-off, which is more informative than AUROC for imbalanced datasets [Saito & Rehmsmeier, 2015]. Given our positive rate of 3.2%, we prioritize AUPRC as the primary metric.

**Threshold-dependent metrics:**
We select the classification threshold by maximizing F1 score on the validation set. This threshold is then fixed and applied to the test set to obtain unbiased estimates of:
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of predicted positives that are true positives
- **Recall**: Proportion of true positives correctly identified

**Generative model evaluation:**
- **Trajectory Fidelity**: Wasserstein distance and CMD to assess distributional similarity
- **Discriminative Performance**: TSTR (Train on Synthetic, Test on Real) AUROC/AUPRC to evaluate downstream task utility

**Efficiency metrics:**
- **Parameter Count**: Number of trainable parameters (in millions)
- **Inference Time**: Average time per sample (in milliseconds) on [hardware]
```

---

### 6.2 结果部分（Results Table Caption）

```markdown
**Table X. Performance Comparison on Medical Risk Prediction**

Models are evaluated on held-out test data. AUROC and AUPRC are threshold-independent ranking metrics. F1, Precision, and Recall are computed at a threshold optimized on the validation set (threshold = 0.XX). AUPRC is the primary metric due to class imbalance (positive rate: 3.2%). Results are averaged over 9 random seeds with standard deviations in parentheses. Statistical significance is assessed via paired t-test (p < 0.05 marked with *).
```

---

## 7. 常见错误和注意事项

### 7.1 阈值泄漏（Threshold Leakage）

**错误做法**：
- 在测试集上搜索最优阈值
- 报告测试集上的"最佳 F1"而不说明阈值如何确定

**正确做法**：
- 在验证集上确定阈值，在测试集上固定使用
- 明确说明阈值选择策略

---

### 7.2 忽略类别不平衡

**错误做法**：
- 仅报告 AUROC 和 Accuracy
- 不报告正类比例

**正确做法**：
- 同时报告 AUROC 和 AUPRC
- 明确说明数据集的类别分布
- 解释为什么 AUPRC 更重要

---

### 7.3 混淆保真度和判别性能

**错误做法**：
- 认为 Wasserstein 距离低就意味着生成数据有用
- 仅报告 TSTR 性能而不评估分布相似性

**正确做法**：
- 同时报告轨迹保真度和判别性能
- 明确说明两者衡量不同维度

---

## 8. 权威参考文献汇总

### 核心方法论文献
1. **Saito, T., & Rehmsmeier, M. (2015).** "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLoS ONE*, 10(3), e0118432.

2. **Davis, J., & Goadrich, M. (2006).** "The relationship between Precision-Recall and ROC curves." *ICML*.

3. **Steyerberg, E. W. (2019).** *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating*. Springer.

4. **Fawcett, T. (2006).** "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861-874.

### 生成模型评估
5. **Yoon, J., Jarrett, D., & van der Schaar, M. (2019).** "Time-series generative adversarial networks." *NeurIPS*.

6. **Esteban, C., et al. (2017).** "Real-valued (medical) time series generation with recurrent conditional GANs." *arXiv:1706.02633*.

7. **Xu, L., et al. (2019).** "Modeling tabular data using conditional GAN." *NeurIPS*.

### 医疗 AI 指南
8. **de Hond, A. A. H., et al. (2022).** "Guidelines and quality criteria for artificial intelligence-based prediction models in healthcare: a scoping review." *npj Digital Medicine*, 5(1), 2.

9. **Park, S. H., & Han, K. (2018).** "Methodologic guide for evaluating clinical performance and effect of artificial intelligence technology for medical diagnosis and prediction." *Radiology*, 286(3), 800-809.

### 官方文档
10. **scikit-learn Documentation**: https://scikit-learn.org/stable/modules/model_evaluation.html
11. **imbalanced-learn Documentation**: https://imbalanced-learn.org/stable/

---

## 9. 快速查询表

| 指标 | 适用场景 | 阈值依赖 | 不平衡敏感 | 主要用途 |
|------|---------|---------|-----------|---------|
| AUROC | 类别平衡/轻度不平衡 | 否 | 低 | 整体排序能力 |
| AUPRC | 高度不平衡 | 否 | 高 | 少数类识别 |
| F1 | 需要单一指标 | 是 | 中 | 精确率-召回率平衡 |
| Precision | 关注误诊成本 | 是 | 中 | 预测为正的准确性 |
| Recall | 关注漏诊成本 | 是 | 中 | 正类覆盖率 |
| Wasserstein | 分布匹配 | 否 | N/A | 轨迹保真度 |
| TSTR AUROC | 下游任务 | 否 | 低 | 判别性能 |

---

## 10. 项目特定建议

基于 Causal-TabDiff 项目的特点：

1. **主指标优先级**：AUPRC > AUROC > F1
2. **阈值策略**：验证集优化 + 测试集固定
3. **生成评估**：同时报告 Wasserstein/CMD（保真度）和 TSTR（判别性能）
4. **效率报告**：参数量 + 推理时间（单样本 ms）
5. **不平衡处理**：明确报告正类比例，解释 AUPRC 优先的理由

---

**文档版本**: v1.0  
**最后更新**: 2026-03-14  
**维护者**: Causal-TabDiff 项目组
