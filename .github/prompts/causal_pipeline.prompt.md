## 元信息
- 版本: v1.0 (L2-Skill-Template)
- 标签: [L2-Coding, Causal-Pipeline, DoWhy-EconML, Standard-SOP]

## 目的
基于临床医学数据集，严格按照因果推断四步法（建模、识别、估算、证伪），生成高鲁棒性、可复用的标准因果分析数据管道代码。

## 输入/输出
- 必需: [临床研究假说]、[核心干预变量 (Treatment)]、[结果变量 (Outcome)]、[已知混杂因子列表 (Confounders)]
- 可选: [特定的证伪方法偏好 (如 Placebo Test)]、[缺失值处理策略]
- 预期输出格式: 
  1. 变量映射与理论对齐说明。
  2.  (DAG 结构定义代码)。
  3. 基于 DoWhy / CausalML 框架的完整 Python 管道代码（包含详尽的医学上下文注释）。
  4. 鲁棒性检验（Refutation）的输出解析指南。

## 提示词正文
作为执行层，你必须严格遵循 L1“资深医学因果算法架构师”的系统准则。在接收到临床研究数据信息后，请严格按以下标准作业程序（SOP）生成代码：

### Step 1: 理论建模与 DAG 定义 (Model)
- 使用 `dowhy.CausalModel` 定义因果图。
- 必须在代码注释中明确声明干预变量 ($T$)、结果变量 ($Y$) 以及观察到的混杂因子 ($X$)。
- 严禁在未声明因果关系的情况下直接将特征扔进模型。

### Step 2: 效应识别 (Identify)
- 调用 `model.identify_effect()`。
- 如果存在未观测混杂因子（Unobserved Confounders）的风险，必须在输出日志中打印警告，并指明后门准则（Backdoor Criterion）是否完全满足。

### Step 3: 开源胶水层估算 (Estimate)
- 强制使用成熟的统计学/因果机器学习方法估算平均因果效应 ($ATE$):
  $$ATE = E[Y^{(1)} - Y^{(0)}]$$
- 优先选用倾向得分匹配（PSM）、逆方差加权（IPW）或 Meta-Learners（如 T-Learner）。
- 代码必须模块化，核心估算器需独立实例化，避免硬编码。

### Step 4: 严苛证伪 (Refute)
- 必须在代码末尾追加至少两种证伪测试（如添加随机共同原因 `add_unobserved_common_cause` 或安慰剂干预 `placebo_treatment_refuter`）。
- 若证伪未通过，必须抛出异常（Exception），禁止返回误导性的估算结果。

## 使用示例
**输入**:
- 假说：评估新型降压药对老年患者收缩压的真实影响。
- 干预变量：`new_drug` (0或1)
- 结果变量：`systolic_bp`
- 混杂因子：`age`, `baseline_bmi`, `comorbidities`

**输出**:
> **理论映射**：$T$ = new_drug, $Y$ = systolic_bp, $X$ = {age, baseline_bmi, comorbidities}。
> **代码实现**：
> ```python
> import dowhy
> import pandas as pd
> 
> # Step 1: Causal Graph Modeling
> model = dowhy.CausalModel(
>     data=df,
>     treatment='new_drug',
>     outcome='systolic_bp',
>     common_causes=['age', 'baseline_bmi', 'comorbidities']
> )
> # ...后续识别、估算与证伪代码...
> ```

## 注意事项
- **系统越权拦截**：如果在分析中发现现有数据根本无法满足因果识别条件（例如核心混杂因子完全缺失），必须直接中断生成，并输出 L1 规定的“阻断报告”，严禁强行拟合输出垃圾代码。