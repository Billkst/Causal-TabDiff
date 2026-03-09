# Project Memory & Context

## 1. 核心任务变更 (Core Task Shift)
- **旧任务 (Deprecated)**：未来完整表格轨迹监督生成 (Full longitudinal tabular trajectory generation)。因为公开数据的真实观测特征主要只覆盖 T0, T1, T2，而癌症结局追踪到 T7，之前采用了不合理的“伪时间复制 (Pseudo-time copying)”来拉长序列，这违背了真实的临床因果观测。
- **新任务 (Current)**：基于地标的2年内首发肺癌风险预测 (Landmark-conditioned 2-year first-lung-cancer risk prediction)。
  - **输入**：在时间点 $t \in \{T0, T1, T2\}$ 的地标 (landmark) 上，只使用**截至 $t$ 之前**已真实观测到的所有历史和当前信息。
  - **输出**：预测未来 2 年内（例如从 $t$ 到 $t+2$ 年）是否会**首次确诊**肺癌。

## 2. 阶段性开发计划
本项目处于重构与清理的过渡期。开发应当严格按照以下三个阶段进行：
1. **清理仓库 (Audit & Cleanup)**：处理积累的技术债，归档大量废弃的、重叠的 `fix_*.py` 和 `smoke_*.py` 脚本，统一代码结构。
2. **重构任务 (Task Refactoring)**：
   - 重写 `src/data/data_module.py`：移除伪时间逻辑，提取 landmark 特征。
   - 重构训练与评估入口 (`run_experiment.py`, `run_baselines.py`)：将评估指标从 TSTR / Wasserstein 转向真实的临床预测指标 (AUC, AUPRC, Calibration, Brier Score)。
3. **正式实验 (Formal Experimentation)**：在清理和重构完成后，开展受控的基线对比与消融实验。

## 3. 后续会话的硬约束 (Strict Rules for Future Sessions)
- **默认归档，不直接删除**：不确定的历史脚本移入 `archive/ legacy_YYYYMMDD/`，不要直接 `rm`。
- **严禁使用伪时间复制**：任何特征填补必须基于严谨的因果推断或临床常识（如 LOCF），绝不允许简单地把 T0 的特征复制成 T1-T7 的假时间序列。
- **实验规范 (5-seeds / nohup / logs)**：
  - 所有评估必须强制运行 5-seeds 求平均。
  - 所有长时实验必须使用 `nohup` 后台运行，不能依赖终端连接。
  - 日志必须统一输出到 `logs/` 下对应的分类目录（如 `logs/training`, `logs/evaluation`），不可散落在根目录。
- **统一评估口径**：模型评估必须使用统一的接口和统一的数据划分，防止数据泄露（如之前出现的 `noleak` 补丁）。