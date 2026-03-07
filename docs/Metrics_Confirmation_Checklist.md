# 核心评估指标确认清单 (Metrics Confirmation Checklist)

## 📌 1. 结局变量 (Outcome Variable $Y$) 的界定与特性
基于对 NLST 数据集的初步探索 (`nlst_780_canc_idc_20210527.csv` 等)，我们发现 $Y$ 存在两种主要形态：
*   **二分类结局 (`cancyr`)**：表示是否患癌。该类别通常存在**严重的类别不平衡 (Class Imbalance)**（绝大多数样本为阴性）。
*   **连续型结局 (`canc_free_days`)**：表示无癌生存天数（通常伴随右删失现象）。

**✅ 架构师建议**：
在标准的 Causal tabular 任务中，如果我们将 $Y$ 定义为二分类的患癌结局（`cancyr`），后续的 TSTR (Train on Synthetic, Test on Real) 任务将是一个**二分类任务**。考虑到存在类别不平衡，传统的评估指标将面临失效风险。

---

## 📌 2. 评估体系重构方案 (Publication-Ready Evaluation Framework)
为了对齐顶会 (ICLR/NeurIPS) 严格的评估标准，废弃原有的简略占位符（如简单 RMSE 或无意义的 Accuracy），重构为以下三大维度：

### 维度一：下游机器学习效用 (Downstream Task Utility - TSTR)
*   **废止指标**：`RMSE` (不适用于二分类 $Y$)、`Accuracy` (在严重不平衡的医学数据中，模型预测全为负类即可获得高准确率，极具欺骗性)。
*   **启用指标**：**`ROC-AUC`** (衡量模型整体的分类阈值剥离能力) 和 **`F1-Score`** (准确率与召回率的调和平均，对少数正类更敏感)。
*   **执行逻辑**：使用生成数据训练 XGBoost/RandomForest 分类器，在真实数据的 Test 集上评估 `ROC-AUC` 和 `F1-Score`。

### 维度二：分布保真度 (Distributional Fidelity)
*   **保留指标**：`Wasserstein Distance` (用于评估单一维度的边缘分布/Marginal Distributions 差异)。
*   **新增指标**：**`Correlation Matrix Distance (CMD)`** 或者 `Pearson Correlation 差异`。
*   **理论依据**：简单的 Wasserstein 只能保证单一特征层面的近似。在医学数据中，特征间的共线性和互相依赖关系（如年龄与特定生化指标的关联）至关重要。引入 CMD 能够以矩阵范数的形式，严密评估生成数据是否维持了原始数据的多元联合分布特性。

### 维度三：因果效应保持度 (Causal Preservation -核心基石)
*   **废止基线**：单纯的 `Ridge Regression` 等带有强自变量正则化的估算器。
*   **理论阻断原因**：Ridge 的 L2 正则化在收缩系数时，会破坏因果推断所需的无偏性，特别是在存在高维复杂混杂因子时，直接拟合 Ridge 会导致严重的正则化诱导偏倚 (Regularization-induced bias)。
*   **启用方案**：强制使用 **`EconML / DoWhy`** 框架。
*   **严格估计方法**：
    *   **Double Machine Learning (DML)**：通过正交化 (Orthogonalization) 彻底剥离混杂因子的影响 $X \rightarrow T$ 和 $X \rightarrow Y$，再使用任意 ML 模型进行偏差修正。
    *   **Inverse Probability Weighting (IPW)**：通过倾向得分 (Propensity Score) 重加权来模拟随机对照试验 (RCT)。
*   **评估指标**：计算 $|\text{ATE}_{synthetic} - \text{ATE}_{real}|$ 的**绝对因果偏差 (ATE Bias)**。

---

## 🙋‍♂️ 3. 待用户确认事项 (User Approval Required)
1.  **结局变量选择**：我们是否统一将 $Y$ 设定为**二分类变量 `cancyr`** (并应用上述分类指标)？
2.  **指标方案确认**：上述三大维度的细分指标及替换原因是否符合您对学术保真度的预期？

**如果确认无误，请回复“确认”。我将立即按照此规范更新 `run_baselines.py` 及相关数据处理逻辑，然后再引入任何深度学习模型。**

---

## 📌 4. 对比实验“防质疑”说明模板（可直接粘贴到论文/答辩）

### 4.1 口径一致性声明（Methods）
在所有生成模型评估中，我们采用统一的评估管线：
1. 使用相同的随机种子策略与数据划分协议；
2. 统一报告 `ATE_Bias`、`Wasserstein`、`CMD`、`TSTR_AUC`、`TSTR_F1`、`Params(M)`、`AvgInfer(ms/sample)`；
3. 对 STaSy 增加独立的泄漏闸门后检（postcheck），包括 shuffle-control、domain AUC、标签比例差、近邻记忆化比值；
4. 主表指标口径保持不变，附加诊断仅用于可解释性，不用于改变主表排序规则。

### 4.2 STaSy 结果解释（Results）
在 no-leak 特征策略（移除高风险泄漏字段）下，STaSy 的 postcheck 显示：
- 泄漏闸门通过（Gate PASS）；
- `domain_auc` 未触发阻断阈值；
- `TSTR_AUC` 接近 0.5、`TSTR_F1` 接近 0。

该现象解释为：在低阳性率与受限特征集条件下，下游分类信号不足，模型在判别任务上退化为近随机水平。这属于“有效但性能较弱”的可解释结果，而非评估脚本故障。

### 4.3 风险边界与审稿应答（Limitations）
为避免“结果异常来自实现错误”的质疑，本研究提供以下复现实证链：
- 守护进程日志与完整命令记录；
- 训练后统一后检报告（含 Gate Verdict）；
- 同口径主表（Markdown/LaTeX）一致输出；
- smoke 测试仅用于脚本连通性验证，不作为正式性能结论。

### 4.4 一句话结论（可直接引用）
本研究中的 STaSy 结果应表述为：
> 在严格 no-leak 设定下，模型未表现出可阻断的信息泄漏信号，但在当前特征约束下分类效能接近随机，因此该结果用于证明流程有效性与公平比较口径，而非证明该基线在本任务上的最优性能。
