# CausalTabDiff Pre-Full Gate Report

- Generated at: 2026-03-07T09:55:00+00:00
- Scope: pre-full-scale gate decision for `CausalTabDiff` under no-leak NLST setting
- Treatment gate: `cigsmok` only
- Metadata gate: `src/data/dataset_metadata_noleak.json`
- Readout gate: `50/50 blend`

## 1. 架构评估结论

**结论：暂不放行 full-scale training。**

当前系统已经从“禁止实验”提升到“通过受控 pilot 验证”，但仍未达到“可以直接上全量正式训练”的门槛。

这不是阻断性失败，而是**审慎延迟放行**：
- 因果接入路径已从伪 treatment / 随机 `Y` 修复到真实 treatment + in-model outcome head
- no-leak 约束已被严格执行
- 多轮受控 pilot 显示存在重复性的弱信号
- 但下游判别效用仍偏弱，且 prevalence-aware 增益跨种子仍不完全一致

## 2. 已满足的门控条件

1. 真实 treatment 已绑定为 `cigsmok`
2. no-leak 元数据已锁定为 `src/data/dataset_metadata_noleak.json`
3. `Y` 已进入模型契约，不再允许 wrapper 随机占位
4. 采样阶段当前最稳的 readout 已确认为 `50/50 blend`
5. 已完成多级别验证：
   - smoke / blocker
   - controlled preview
   - stronger single-model pilot
   - multi-seed pilot
   - larger controlled 3-seed pilot

## 3. 关键证据

### A. controlled preview
来源：[logs/testing/causal_tabdiff_controlled_preview.md](logs/testing/causal_tabdiff_controlled_preview.md)

- `ATE_Bias = 0.041025`
- `Wasserstein = 1.276005`
- `CMD = 0.003729`
- `TSTR_AUC = 0.610303`
- `TSTR_F1 = 0.082902`

含义：模型通过了第一次中等强度预演闸门。

### B. stronger single-model pilot
来源：[logs/testing/causal_tabdiff_single_model_pilot.md](logs/testing/causal_tabdiff_single_model_pilot.md)

- `ATE_Bias = 0.014587`
- `Wasserstein = 0.665603`
- `CMD = 0.009883`
- `TSTR_AUC = 0.553381`
- `TSTR_F1 = 0.000000`

含义：单次更强配置下，固定 `0.5` 阈值会导致 F1 塌缩。

### C. threshold analysis
来源：[logs/testing/causal_tabdiff_pilot_threshold_analysis.md](logs/testing/causal_tabdiff_pilot_threshold_analysis.md)

- 默认阈值 `0.5`：`F1 = 0.000000`
- 真实患病率匹配阈值：`F1 = 0.058333`
- 生成患病率匹配阈值：`F1 = 0.052402`
- oracle 阈值：`F1 = 0.069444`
- `PR_AUC = 0.034014`

含义：问题主要不是“完全无排序信号”，而是“信号弱且固定阈值过硬”。

### D. multi-seed pilot
来源：[logs/testing/causal_tabdiff_multiseed_pilot.md](logs/testing/causal_tabdiff_multiseed_pilot.md)

- `Base TSTR_AUC = 0.577027 ± 0.024149`
- `Base TSTR_F1 @0.5 = 0.065739 ± 0.020295`
- `PR_AUC = 0.042132 ± 0.008784`
- `Real-prev F1 = 0.056690 ± 0.018202`
- `Seeds with base F1 > 0: 5/5`
- `Seeds with real-prev F1 > 0.05: 2/5`

含义：跨种子已有稳定非零效用，但 robustness 还不够强。

### E. large controlled pilot
来源：[logs/testing/causal_tabdiff_large_controlled_pilot.md](logs/testing/causal_tabdiff_large_controlled_pilot.md)

- `Base TSTR_AUC = 0.599046 ± 0.022175`
- `Base TSTR_F1 @0.5 = 0.066856 ± 0.009807`
- `PR_AUC = 0.042965 ± 0.003657`
- `Real-prev F1 = 0.059541 ± 0.015358`
- `Seeds with base F1 > 0.05: 3/3`
- `Seeds with real-prev F1 > 0.05: 2/3`

含义：在更强受控配置下，默认阈值下的非零效用已更稳定，但 prevalence-aware 表现仍未完全统一。

## 4. 医学因果视角下的判定

### 理论维度
- 当前 treatment 选择 `cigsmok` 合理。
- 泄漏/后处理变量未被重新引入。
- 因果目标与 no-leak 审计保持一致。

### 数据维度
- 极端不平衡仍然是核心困难。
- 真实阳性率极低，导致固定阈值极易把弱信号压成全阴性。
- `PR_AUC` 仍低，说明排序信号虽存在，但临床上还偏弱。

### 逻辑维度
- 现阶段适合“继续受控验证”，不适合“直接正式 full-scale 结论化训练”。
- 若现在直接上 full-scale，最大风险不是训练崩溃，而是**把仍偏弱的下游效用过度解释**。

## 5. 放行判定

**正式判定：不放行 full-scale training。**

### 允许的下一步
1. 固化 prevalence-aware 评估口径到正式实验脚本
2. 执行一次单模型 full-scale 前的最终 rehearsal / gated run
3. 将 full-scale 视作“待批准动作”，而非默认下一步

### 不允许的下一步
1. 直接把当前结果当作最终正式实验结果
2. 在未固化评估口径前启动无人值守全量长期训练
3. 用单一 `0.5` 阈值结论覆盖当前全部效用判断

## 6. 建议执行顺序

### 推荐方案
1. 先把 prevalence-aware 评估写入正式实验流程
2. 再做一次 final gated rehearsal
3. 若 final gated rehearsal 继续保持：
   - `Base TSTR_F1` 不退化
   - `AUC` 不明显回落
   - prevalence-aware F1 不崩
   再申请 full-scale 单模型 run

## 7. 最终一句话

当前 `CausalTabDiff` 已经从“不可实验”进入“可受控推进”，但**还没有进入“可直接全量放行”**。
