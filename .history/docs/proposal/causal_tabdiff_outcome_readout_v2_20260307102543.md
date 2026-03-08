# CausalTabDiff Outcome/Readout V2 设计草案

## 1. 设计结论

当前 `CausalTabDiff` 的主干设计与开题报告在以下部分是对齐的：
- 异构表型进入统一连续空间
- 正交时序/特征骨干
- 因果约束梯度引导采样

但当前 `Y/readout` 路径与开题报告的“纵向风险轨迹 + 反事实推演”目标并不充分一致。

当前问题不在于扩散骨干完全错误，而在于**最终风险读出过于晚绑定、过于二值化、过于依赖 sample-time 阈值**。

因此提出 V2：

**把单一二分类 `outcome_head` 改为“轨迹级风险读出头（Risk Trajectory Readout Head）”。**

## 2. 为什么当前版本与开题报告不完全一致

参考：[docs/proposal/方案二：开题报告.md](docs/proposal/方案二：开题报告.md)

开题报告在高危人群识别部分的真正目标是：
1. 生成完整纵向轨迹
2. 在不同干预条件下做反事实风险演变比较
3. 从轨迹中读出风险，而不是只从最后一步做静态二分类

而当前实现：
- 在 [src/models/causal_tabdiff.py](src/models/causal_tabdiff.py) 中，`predict_outcome_logits()` 本质还是“轨迹压缩后的一次性二分类”
- 在 [src/baselines/wrappers.py](src/baselines/wrappers.py) 中，最终 `Y` 还受到 prevalence calibration 与 sample-time readout 混合策略影响
- 这更像“生成后补一个标签头”，而不是“生成风险演化轨迹并做因果读出”

## 3. V2 的核心思想

### 3.1 从单点标签预测改为离散时间风险建模

对每个时间步 $t$ 输出一个风险对数几率：

$$
\ell_t = g_\theta(x_t, \alpha, c)
$$

对应 hazard：

$$
h_t = \sigma(\ell_t)
$$

再把整条轨迹的累计风险定义为：

$$
R = 1 - \prod_{t=1}^{T}(1 - h_t)
$$

含义：
- `h_t` 表示该时间步的局部风险强度
- `R` 表示整条轨迹上的累计发病风险
- 最终监督不再只盯住最后一个 pooled feature，而是直接让模型学习“风险如何随时间累积”

### 3.2 用“轨迹风险”替代 wrapper 末端胶水标签

V2 原则：
- wrapper 可以保留外部 glue 作为诊断对照
- 但正式 readout 以模型内部 `trajectory risk score` 为主
- 二值 `Y` 仅作为评估阶段阈值化结果，不再作为训练时唯一核心目标

### 3.3 加入反事实一致性读出

对于同一个潜在轨迹表示，分别计算：
- factual risk: $R(\alpha)$
- counterfactual risk: $R(\alpha')$

定义个体化反事实风险差：

$$
\Delta R = R(\alpha = 1) - R(\alpha = 0)
$$

在 no-leak 当前口径下，`cigsmok` 是允许的真实 treatment，因此可以把 $\Delta R$ 作为更接近论文目标的读出对象：
- 不只是“会不会阳性”
- 而是“吸烟干预对风险轨迹造成了多少净变化”

## 4. V2 结构设计

### 模块 A：`RiskTrajectoryHead`

建议在 [src/models/causal_tabdiff.py](src/models/causal_tabdiff.py) 中新增独立头：

- 输入：整条隐轨迹 `x` 与 `alpha_target`
- 输出：
  - `hazard_logits`，形状 `[B, T, 1]`
  - `risk_score`，形状 `[B, 1]`
  - `cf_risk_gap`，形状 `[B, 1]`（可选）

建议形式：

1. 对每个时间步做共享 MLP：
   $$z_t = \phi([x_t, e(\alpha)])$$
2. 输出 hazard logits：
   $$\ell_t = w^T z_t + b$$
3. 聚合成累计风险：
   $$R = 1 - \prod_t (1 - \sigma(\ell_t))$$

### 模块 B：`CounterfactualReadout`

在同一条生成轨迹上，对 `alpha=0` 和 `alpha=1` 分别做风险读出：

$$
R_0 = f_{risk}(x, \alpha=0), \quad R_1 = f_{risk}(x, \alpha=1)
$$

再定义：

$$
\Delta R = R_1 - R_0
$$

这样更符合开题报告“反事实风险演变”的表述。

### 模块 C：正式评估读出策略

对正式运行，建议分三层输出：
1. `risk_score`：连续风险分数
2. `risk_binary_real_prev`：按真实患病率阈值化
3. `cf_risk_gap`：反事实风险差

其中：
- `risk_score` 负责 `AUC/PR_AUC`
- `risk_binary_real_prev` 负责 prevalence-aware `F1`
- `cf_risk_gap` 负责因果方向 sanity check

## 5. V2 损失函数

### 5.1 基础风险损失

仍可保留 BCE/Focal 风格：

$$
\mathcal{L}_{risk} = \operatorname{BCE}(R, y)
$$

或对极不平衡更稳的 focal：

$$
\mathcal{L}_{focal} = -\alpha y (1-p)^\gamma \log p - (1-\alpha)(1-y)p^\gamma \log(1-p)
$$

### 5.2 离散时间 hazard 平滑约束

为了避免时间步风险剧烈抖动：

$$
\mathcal{L}_{smooth} = \frac{1}{T-1} \sum_{t=2}^{T} (h_t - h_{t-1})^2
$$

### 5.3 反事实一致性约束

在群体均值层面，要求高暴露的风险不应系统性低于低暴露：

$$
\mathcal{L}_{cf} = \max(0, m - (\bar R_1 - \bar R_0))
$$

其中 $m \ge 0$ 为一个很小的安全边际。

注意：
- 这里只建议做弱约束
- 不建议把单调性写得过硬，否则会把真实弱信号压坏

### 5.4 总损失

$$
\mathcal{L} = \mathcal{L}_{diff} + 0.5\mathcal{L}_{disc} + \lambda_1 \mathcal{L}_{risk} + \lambda_2 \mathcal{L}_{smooth} + \lambda_3 \mathcal{L}_{cf}
$$

## 6. 与当前版本的最大区别

V2 与当前版本的根本区别不是“再换一个 pooling”，而是：

- 当前：`trajectory -> pooled feature -> binary Y`
- V2：`trajectory -> timewise hazard -> cumulative risk -> counterfactual gap`

这条路径更接近开题报告中的：
- 风险演变轨迹
- 反事实比较
- 因果一致性约束

## 7. 工程落地建议

### 第一步
在 [src/models/causal_tabdiff.py](src/models/causal_tabdiff.py) 中新增：
- `RiskTrajectoryHead`
- `predict_risk_trajectory()`
- `predict_cumulative_risk()`
- `predict_counterfactual_risk_gap()`

### 第二步
在 [src/baselines/wrappers.py](src/baselines/wrappers.py) 中：
- 保留 glue 作为调试参考
- 但正式 `sample()` 改为优先输出模型内 `risk_score`
- `Y` 的二值化只留在评估阶段

### 第三步
在 gate 评估中增加：
- `risk_score` 的 `AUC`
- `risk_score` 的 `PR_AUC`
- `real-prev` 阈值化 `F1`
- `cf_risk_gap` 的方向性检查

## 8. 当前建议

**建议不要再沿“简单 BCE + 额外 ranking loss”这条线继续微调。**

更值得投入的是：
- 把正式 outcome contract 改成“轨迹风险读出”
- 让 readout 与开题报告真正对齐

## 9. 最终一句话

当前 CausalTabDiff 的扩散骨干并不一定错，真正偏离开题报告的更可能是 `Y/readout` 契约本身。

V2 应该从“二分类标签头”转向“轨迹级风险读出头 + 反事实风险差读出”。
