# Repository File Manifest & Functional Analysis

## 1. 核心目录与文件职责

- **`docs/proposal/` & `docs/dataset/`**
  - **职责**: 存放学术基础文档。包括开题报告（确定了基于因果表征的高危人群识别方向）以及医学数据的变量字典（如 idc_ctabc, idc_screen）。
- **`src/data/`**
  - **职责**: 数据加载与预处理核心。包含 `data_module.py` 和元数据描述 `dataset_metadata.json` / `dataset_metadata_noleak.json`。目前内部很可能包含为了解决时间序列不足而采用的“伪时间”复制逻辑，这是下一步重构的重点。
- **`src/baselines/`**
  - **职责**: 对比算法库。目前集成了 STaSy (ICLR 23), TabSyn (ICLR 24), TabDiff (ICLR 25), TSDiff (ICLR 23) 以及 CausalForest。通过 `wrappers.py` 提供了统一的调用接口。
- **`run_experiment.py`**
  - **职责**: 当前主模型的训练入口脚本，目前默认执行完整的扩散轨迹拟合（与新任务不符）。
- **`run_baselines.py`**
  - **职责**: 基线模型的批量评估入口。当前包含的指标 (Wasserstein, CMD, ATE_Bias, TSTR) 面向的是表格生成任务，对于即将重构的风险预测任务，这套评测体系不再适用。

## 2. 混乱与重叠区域 (The "Mess")

- **大量的修补脚本 (Patch Scripts)**
  根目录下散落了大量的针对单一问题的修复脚本，例如 `fix2.py`, `fix_baselines.py`, `fix_eval.py` 到 `fix_eval6.py`。这说明项目前期缺乏良好的模块化设计，遇到问题就用单一脚本打补丁。
- **探索性脚本泛滥 (Script Sprawl)**
  `scripts/` 目录下多达数十个脚本（如 `smoke_causal_tabdiff_*.py`），它们功能重叠严重，都是针对模型的不同超参、消融策略或者小批次进行探索。这些应该通过统一入口 + 配置文件来替代。
- **日志与临时产物**
  `logs/` 下既有有效的日志，也充斥着各种 `nohup_*.pid` 和临时运行的输出；根目录下出现了 `temp_stasy/` 这样的无规律缓存目录，进一步降低了项目的可维护性。