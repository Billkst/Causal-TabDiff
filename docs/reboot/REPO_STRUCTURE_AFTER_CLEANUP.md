# Repository Structure (After Cleanup 2026-03-09)

## 1. Directory Tree
```text
.
├── archive/
│   └── legacy_20260309/      # 历史审计归档
│       ├── root_patches/     # 根目录修复脚本与Shell
│       ├── scripts_probes/   # 实验探针与消融脚本
│       └── legacy_nohup/     # 历史nohup日志与markdown记录
├── configs/                  # 待填充：集中管理模型配置文件
├── checkpoints/              # 运行时：模型权重 (Gitignored)
├── data/                     # 核心：数据集 (NLST等)
├── docs/                     # 核心：项目理论与审计记忆
│   ├── dataset/              # 数据字典
│   ├── proposal/             # 开题报告
│   └── reboot/               # 重构审计与清理报告 (当前)
├── logs/                     # 运行时：实验日志 (Gitignored)
│   ├── evaluation/
│   ├── training/
│   └── testing/
├── outputs/                  # 运行时：生成样本与中间产物 (Gitignored)
├── reports/                  # 成果：汇总后的 MD/Latex 报告
├── scripts/                  # 实用：保留的核心工具脚本
│   ├── run_stasy_with_audit.sh
│   └── ...
├── src/                      # 核心源码：算法与数据逻辑
│   ├── baselines/            # 各基线模型实现
│   ├── data/                 # 数据处理与 DataLoader
│   └── models/               # 主模型 Causal-TabDiff
├── run_experiment.py         # 训练入口
├── run_baselines.py          # 评估入口
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 2. 核心文件职责说明

### 实验入口 (Entrypoints)
- **`run_experiment.py`**: 主模型 `Causal-TabDiff` 的训练总入口。需关注其如何调用 `CausalTabDiff` 类及处理 `alpha_target` 指导。
- **`run_baselines.py`**: 批量评估基线的脚本。目前集成了 ATE_Bias, Wasserstein, CMD 等关键指标的计算。

### 源码 (src/)
- **`src/data/data_module.py`**: 处理 NLST 数据的 DataLoader。它是未来重构 Landmark 任务、解决“伪时间”问题的核心。
- **`src/models/causal_tabdiff.py`**: 主模型的实现。包含双重注意力骨干、因果解耦及梯度引导逻辑。
- **`src/baselines/`**: 包含所有对比算法（STaSy, TabSyn 等）的核心实现，这些是证明 Causal-TabDiff 优越性的参照。

### 因果评估逻辑 (Causal & Generation Logic)
- 仓库明确保留了所有包含 `ATE_Bias` (利用 EconML 计算)、`Wasserstein`、`CMD` 以及 `alpha_target` / `guidance_scale` 的代码。这些逻辑对于 Causal-TabDiff 捕捉因果效应至关重要，即便在任务重构后也将继续作为分布一致性的重要诊断指标。
