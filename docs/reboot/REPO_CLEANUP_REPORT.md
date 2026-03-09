# Repository Cleanup Report (2026-03-09)

## 1. Moved Files (Archived)
以下文件已移动至 `archive/legacy_20260309/`，它们属于历史探索脚本、临时补丁或旧任务遗留物，虽不再作为主任务入口，但为保留实验证据而归档：

### Root Patch Scripts -> `archive/legacy_20260309/root_patches/`
- `fix2.py`, `fix_baselines.py`, `fix_eval.py` (及2-6)
- `fix_sample.py`, `fix_script.py`, `fix_wrapper.py`
- `rewrite.py`, `update_wrapper_early_stop.py`, `update_history.py`
- `append_profile.py`, `copy_script.py`, `debug_run.py`, `explore_schema.py`, `test_focal.py`
- `run_server.sh`, `run_all_noleak_baselines.sh`

### scripts/ Probes -> `archive/legacy_20260309/scripts_probes/`
- `ablate_causal_tabdiff_ranking_loss.py`
- `ablate_causal_tabdiff_readout_guidance.py`
- `analyze_causal_tabdiff_pilot_thresholds.py`
- `audit_baseline_report.py`
- `audit_causal_treatment_candidates.py`
- `build_noleak_metadata.py`
- `check_imbalance_threshold_proof.py`
- `check_stasy_leakage.py`
- `rehearse_causal_tabdiff_controlled_preview.py`
- `run_causal_tabdiff_final_gated_rehearsal.py`
- `run_causal_tabdiff_large_controlled_pilot.py`
- `run_causal_tabdiff_multiseed_pilot.py`
- `run_causal_tabdiff_single_model_pilot.py`
- `run_causal_tabdiff_v2_locked_5seed.py`
- `search_causal_tabdiff_v2_champion.py`
- `smoke_causal_tabdiff_*.py` (全部副本)

### Legacy Logs & Temp -> `archive/legacy_20260309/legacy_nohup/`
- `logs/` 下所有 `nohup_*.log` 和 `.pid`
- `logs/` 下所有历史 `.md` 实验记录
- `temp_stasy/`, `temp_tabdiff/`, `temp_tabsyn/`

## 2. Deleted Files
以下文件已被永久删除，主要为缓存和无保留价值的中间过程文本：
- `.history/` (开发备份)
- `__pycache__/` (Python 字节码缓存)
- 根目录临时调试输出：`debug_output*.txt`, `col_check.txt`, `cols.txt`, `trace.txt`, `proof_out.txt`, `test.log`, `true_error.log`

## 3. Kept Files (核心保留清单)
- **Data**: `data/**` (未做任何变动)
- **Docs**: `docs/proposal/**`, `docs/dataset/**`, `docs/reboot/**` (核心理论与记忆文档)
- **Source**: `src/**` (包含 `baselines/`, `models/`, `data/`)
- **Main Scripts**: `run_experiment.py`, `run_baselines.py` (主任务入口)
- **Metrics/Logic**: 显式保留了所有与 `ATE_Bias`, `Wasserstein`, `CMD`, `causal guidance`, `alpha_target` 相关的代码路径，即便部分脚本已归档，其逻辑在 `src/` 中依然完整。

## 4. 风险提示与归档说明
- **归档而非删除原因**：许多归档的脚本包含特定的超参设置和因果评估分支，未来在 Landmark 重构完成后，可能需要参考其中的特定引导（Guidance）参数。
- **不可使用警告**：不建议再直接运行 `archive/` 下的脚本，除非明确了解其历史背景。后续开发应基于 `src/` 和新的主入口。

## 5. 新目录结构摘要
- `src/`: 核心业务逻辑
- `scripts/`: 保留的核心实用脚本（如训练/评估封装）
- `configs/`: 配置文件集中地
- `reports/`: 实验报告 (markdown/latex)
- `archive/`: 历史证据库
- `logs/`, `checkpoints/`, `outputs/`: 运行时动态产生物（已加入 .gitignore）
