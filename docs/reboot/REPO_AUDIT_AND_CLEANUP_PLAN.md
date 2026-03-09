# Repository Audit and Cleanup Plan

本清单为自动化审计生成的清理建议。在执行下一步代码修改前，应参考此计划对仓库进行治理。

## 1. KEEP（必须保留）
- `data/**`：存放数据集的原始/处理结果，绝对不能动。
- `docs/proposal/**`：包括开题报告，是项目的核心理论基础。
- `docs/dataset/**`：数据字典与说明，解析数据必须。
- `src/data/data_module.py`：即便要重构，原逻辑也是理解数据结构的参考。
- `src/baselines/**`：现有的基线模型核心代码，需要保留并适配新任务。
- 最新有效实验日志与报告（如 `markdown_report.md`, `latex_report.txt`, `logs/evaluation/baselines.log`）：作为上一阶段的基准参考。
- `requirements.txt` / `.agents/**`：环境与系统级配置。

## 2. ARCHIVE_CANDIDATE（疑似废弃，先归档）
建议创建 `archive/legacy_scripts/` 和 `archive/legacy_roots/` 进行归档：
- 大量根目录修补脚本：`fix2.py`, `fix_baselines.py`, `fix_eval*.py`, `fix_sample.py`, `fix_script.py`, `fix_wrapper.py`, `rewrite.py`, `test_focal.py`, `explore_schema.py`, `update_wrapper_early_stop.py` 等。
- `scripts/` 下的大量重复探测脚本：`smoke_causal_tabdiff_*.py`, `search_causal_tabdiff_*.py`, `ablate_*.py`。这些脚本职责高度重合，名字混乱，应归档后在 `src/` 中统一提供参数化的入口。
- 根目录的旧 bash 脚本：`run_all_noleak_baselines.sh`, `run_server.sh`。

## 3. SAFE_DELETE（可安全删除或加入 .gitignore）
- 根目录下的各类调试输出文本：`debug_output*.txt`, `col_check.txt`, `cols.txt`, `trace.txt`, `proof_out.txt`, `test.log`, `true_error.log`。
- 无用的历史备份文件：`.history/` 目录中的自动备份（除非有特殊版本回溯需求，否则作为代码仓库可以直接交由 Git 管理）。
- 缓存与临时编译：所有的 `__pycache__`（应确保在 `.gitignore` 中）。

## 4. NEED_HUMAN_ATTENTION（需要人工确认）
- `temp_stasy/`, `temp_tabdiff/`, `temp_tabsyn/`：如果存放的是临时模型权重，建议移至 `checkpoints/` 下并用 `.gitignore` 忽略；如果是评估生成的样本，建议移至 `outputs/`。
- `dataset_metadata_noleak.json` 与 `dataset_metadata.json`：疑似数据泄露修复过程留下的两个版本，需要确认后续统一使用哪个。

## 5. 建议的新目录结构草案
未来重构后，仓库应遵循以下结构：
```text
.
├── archive/              # 归档所有旧的烟雾测试、探针脚本与根目录fix代码
├── configs/              # 集中管理各模型的 yaml/json 配置文件
├── data/                 # 数据文件集 (Gitignore)
├── docs/                 # 项目文档、开题报告、数据字典、重构记忆
├── checkpoints/          # 模型权重保存目录 (Gitignore)
├── logs/                 # 日志统一存放 (Gitignore)
│   ├── training/
│   ├── evaluation/
│   └── legacy/           # 存放以前旧任务留下的 nohup logs
├── outputs/              # 评估时生成的中间结果、生成数据等 (Gitignore)
├── reports/              # 包含 markdown/latex 报告汇总
├── scripts/              # 高层次的可执行脚本（按类别规范命名，如 train_model.sh, eval_model.sh）
├── src/                  # 核心源码
│   ├── data/             # DataLoader与数据预处理（重构重点：提取landmark序列）
│   ├── models/           # 模型定义
│   ├── baselines/        # 基线模型
│   ├── metrics/          # 临床医学评测指标 (AUC, Calibration 等)
│   └── utils/
├── run_experiment.py     # 统一的训练入口（支持参数化调用不同模型与配置）
├── run_baselines.py      # 统一的评估入口
└── README.md
```