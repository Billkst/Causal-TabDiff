# Causal-TabDiff 统一随访监督实验 · 独立技术审查

本分支保存 2026-11-22 统一模型在 10,594 名开发患者上的研究代码、配置和汇总证据。A/B/C/D 使用同一份当前历史表征进行风险预测，B/C/D 在训练时接收后续真实 CT 测量监督。完整模型 C 的 AUPRC 为 0.03228，最强原定基线 iTransformer 为 0.03718。分支独立于旧 main 的提交历史。已有仓库读权限仍可浏览旧 main，审查分支不会改变该权限。

## 审查入口

| 研究问题 | 实现与证据 |
|---|---|
| 原始与统一模型怎样衔接 | [原始条件扩散网络](server_repos/20260724/source/Causal-TabDiff/src/models/causal_tabdiff.py)、[原始诊断双头](server_repos/20260724/source/Causal-TabDiff/src/models/causal_tabdiff_trajectory.py)、[统一模型](JBHI/experiments/unified_followup_20261122/model.py)、[四组协议](JBHI/experiments/unified_followup_20261122/protocol.json) |
| 患者划分、信息时点、梯度和预算如何核对 | [注册表构建](JBHI/experiments/unified_followup_20261122/registry_data.py)、[训练入口](JBHI/experiments/unified_followup_20261122/train.py)、[预检结果](JBHI/experiments/unified_followup_20261122/preflight.json)、[配对预算定义](JBHI/experiments/unified_followup_20261122/fixed_budget.json) |
| 比较的结论如何得到 | [完整技术报告](JBHI/experiments/unified_followup_20261122/Causal-TabDiff统一方法与关键实验报告.md)、[三项科学问题](JBHI/experiments/unified_followup_20261122/三项核心科学问题验证结论.md)、[中文论文后续重构依据](JBHI/experiments/unified_followup_20261122/中文论文后续重构依据.md)、[图 1 说明](JBHI/experiments/unified_followup_20261122/图1新增辅助监督分支修改说明.md) |
| 如何复核结果的粒度与统计 | [每个模型、折和种子的 117 行指标](JBHI/experiments/unified_followup_20261122/statistics/metrics.csv)、[患者配对 AUPRC 差及区间](JBHI/experiments/unified_followup_20261122/statistics/paired.json)、[次要指标配对区间](JBHI/experiments/unified_followup_20261122/statistics/secondary_paired.csv)、[历史权重回放核验汇总](JBHI/experiments/unified_followup_20261122/verification.json) |

## 不读取临床数据的检查

在与实际运行记录相符的 Python 环境中，可从实验目录导入模型、评估代码和基线适配器。汇总表可独立验证 13 个模型各有 3 折 × 3 种子的 AUPRC 均值。患者配对区间的逐次重算需要受控环境内的逐例分数。[实际环境](JBHI/experiments/unified_followup_20261122/environment.json)

```bash
cd JBHI/experiments/unified_followup_20261122
python -B -c "import model, train, baseline, analyze, secondary, verify"
python -B - <<'PY'
import pandas as pd
rows = pd.read_csv("statistics/metrics.csv")
reported = pd.read_csv("statistics/means.csv").set_index("model")
grouped = rows.groupby("model")["ap"]
assert len(rows) == 117 and (grouped.size() == 9).all()
assert (grouped.mean() - reported["ap"]).abs().max() < 1e-12
PY
```

## 源码与外部依赖

研究程序使用原项目的相对目录。原模型和 TSLib 中实际导入的模块放在 `server_repos/20260724/source/Causal-TabDiff/`。九基线还导入同级实验目录中的历史比较器、参数网格和统计函数。TSDiff、SSSD 使用研究时的适配源码及上游授权文件。SurvTraj 使用未经改动的外部 Python 包，上游版本固定为 `f5dbc2593091341b0f78bee098972d83e5f521f6`。需要实际实例化该基线时，将该版本的 `surv_traj/` 包放在 `JBHI/experiments/marked_progression/assets/upstream/survtraj/surv_traj/`，或者把包含该包的上游检出目录加入 PYTHONPATH。此次对照已逐文件核对本地该包与固定版本。[SurvTraj 固定源码](https://github.com/NTAILab/SurvTraj/tree/f5dbc2593091341b0f78bee098972d83e5f521f6/surv_traj)

<a id="local-assets"></a>
## 受控环境中的依赖

| 原项目位置 | 使用目的 |
|---|---|
| `_audit_sources/20260724/Causal-TabDiff/data/` | 已获授权的 NLST 交付表和 `landmark_tables/unified_person_landmark_table.pkl` 提供原始观测、历史及结局。 |
| `JBHI/experiments/patient_diffusion/private/` | 已冻结患者映射与角色文件 cv0、person、ids、lm 用于建立开发患者名单。 |
| `JBHI/experiments/clinical_paths/baseline_development_inputs/` | folds、y、lead、native 是私有的开发分折、目标和输入数组。字段名称位于分支所附 cohort_reference.json。 |
| 实验目录下的 `private/`、`runs/`、`baselines/`、`matched/` | 本地注册表、权重、逐例预测、逐例诊断及训练中间资产，由受控环境保管。 |
| 实验目录下的 `replay/` 和 `statistics/patient_bootstrap_ap.npz` | 逐拟合回放凭据和重采样数组留在本地。分支提供回放总表、配对差及置信区间。 |

具有相应数据访问权限的研究者可结合上述输入使用原版 reproduce.sh。CTD_PYTHON 可指向本地已配置的解释器，environment.json 中的 /tmp 路径记录原实验环境。审查副本将完整报告里六处本地资产链接指向本节，报告的实验数值及正文保留原样。[复现入口](JBHI/experiments/unified_followup_20261122/reproduce.sh)
