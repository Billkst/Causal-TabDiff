# B2 模型准备阶段 - 最终报告

**日期**: 2026-03-11  
**状态**: 已完成核心工作  

---

## 执行总结

### ✅ 已完成

#### 1. TSDiff Trajectory Upgrade（Workstream 1）
**状态**: ✅ 完全完成

**改造内容**:
- 新增 `condition_dim` 参数支持条件编码
- 新增 `condition_encoder` 模块（256维隐层）
- 修改 `train_step()` 支持条件输入
- 修改 `sample()` 支持多步轨迹生成

**当前能力**:
- ✅ 层1: 2-year risk（从 trajectory[1] 提取）
- ✅ 层2: 6-year risk trajectory 完整输出
- ✅ 条件生成: history → future trajectory
- ✅ 向后兼容: 无条件模式仍可用

**Smoke Test**: ✅ 全部通过
- 原始模式（无条件）: ✓
- 轨迹模式（有条件）: ✓
- 6年轨迹生成: ✓
- 输出形状 [Batch, 6, 1]: ✓

**改动文件**:
- `src/baselines/tsdiff_core/model.py` (重写)
- `smoke_test_tsdiff_trajectory.py` (新增)

---

#### 2. TSLib 模型选择（Workstream 2）
**状态**: ✅ 选择完成，仓库已克隆

**最终选择**: 2 个模型
1. **iTransformer** - SOTA 性能，反转 Transformer 架构
2. **TimeXer** - 专门处理外生变量，适合 landmark-conditioned

**为什么不选其他**:
- PatchTST: 性能不如 iTransformer
- TimeMixer: 复杂度高，接入成本大

**仓库状态**: ✅ 已克隆至 `external/TSLib`

**接入状态**: ⚠️ 需要完整适配（预计 2-3 小时）
- 数据格式转换
- 模型封装
- 评估对接

---

### ⚠️ 部分完成

#### 3. SSSD / SurvTraj Feasibility（Workstream 3）
**状态**: ⚠️ Feasibility Audit 完成，接入待定

**SSSD**:
- **仓库**: https://github.com/AI4HealthUOL/SSSD
- **Feasibility**: 🟡 中等
- **关键问题**:
  - 需要改造为条件生成（当前是无条件）
  - 数据格式需要适配（SSM 状态空间模型）
  - 训练稳定性需要验证
- **预计工作量**: 3-4 小时
- **建议**: 作为补充对比，优先级低于 TSLib

**SurvTraj**:
- **仓库**: https://github.com/NTAILab/SurvTraj
- **Feasibility**: 🟢 高
- **关键问题**:
  - 生存分析 → 风险预测转换（相对简单）
  - 输入格式适配（标准表格数据）
  - 轻量级，易于接入
- **预计工作量**: 1-2 小时
- **建议**: 可快速接入作为层2补充

---

#### 4. 统一封装与评估兼容（Workstream 4）
**状态**: ⚠️ 框架已明确，待实施

**待完成**:
- [ ] TSLib 模型封装与适配
- [ ] 统一 prediction file 格式
- [ ] 对接 evaluate_model.py
- [ ] 对接 trajectory 评估
- [ ] 统一 plots / metrics / logs

---

## 新 Baseline 体系分层

### 层级 1: Direct Predictive Baselines（判别式，层1）
- Logistic Regression ✅
- XGBoost ✅
- Balanced Random Forest ✅
- CausalForest ✅
- **iTransformer** ⏳ (TSLib)
- **TimeXer** ⏳ (TSLib)

### 层级 2: Generative Baselines - TSTR（生成式，层1）
- STaSy ⏳ (需 TSTR 实现)
- TabSyn ⏳ (需 TSTR 实现)
- TabDiff ⏳ (需 TSTR 实现)
- TSDiff（原版）⏳ (需 TSTR 实现)

### 层级 3: Trajectory-Capable Baselines（层1+层2）
- **TSDiff (改造版)** ✅ 已完成
- **iTransformer** ⏳ (TSLib，待接入)
- **TimeXer** ⏳ (TSLib，待接入)
- **SurvTraj** 🟡 (可选，优先级低)
- **SSSD** 🟡 (可选，优先级低)

---

## 后续正式实验顺序

### Phase 1: 完成 TSLib 接入（优先）
1. iTransformer 适配与 smoke test
2. TimeXer 适配与 smoke test
3. 对接评估管道

### Phase 2: TSDiff 改造版正式训练
1. 在真实数据上训练 TSDiff 改造版
2. 输出层1和层2 prediction files
3. 统一评估

### Phase 3: 补充 TSTR Baselines（可选）
1. 实现 STaSy/TabSyn/TabDiff/TSDiff 的 TSTR 协议
2. 或标记为 future work

### Phase 4: 补充 SurvTraj（可选）
1. 快速接入 SurvTraj
2. 作为层2医疗专用对比

---

## 关键决策与理由

### 1. 为什么 TSDiff 必须改造？
- 层2任务需要输出未来多步风险轨迹
- 改造成本可控（已完成，100行代码）
- 保留原版本作为层1 baseline

### 2. 为什么选择 iTransformer 和 TimeXer？
- iTransformer: SOTA 性能基准
- TimeXer: 专门处理外生变量，适合 landmark-conditioned
- 两者互补，覆盖判别式 SOTA

### 3. 为什么 SSSD/SurvTraj 优先级低？
- TSLib 更成熟，优先建立性能基准
- SSSD 改造复杂度高
- SurvTraj 可作为快速补充

### 4. 为什么 TSTR Baselines 暂缓？
- 实现完整 TSTR 协议需要大量时间
- 优先完成判别式和轨迹 baselines
- 可在后续阶段补充

---

## 阻塞点与解决方案

### TSLib 接入阻塞点
**问题**: 数据格式转换复杂
**解决**: 创建统一的数据适配层

### TSTR 协议阻塞点
**问题**: 生成模型训练 + 下游分类器训练流程长
**解决**: 标记为 Phase 3，优先完成判别式

### 评估统一阻塞点
**问题**: 不同模型输出格式不一致
**解决**: 创建统一的 prediction file 格式

---

## 文件清单

### 新增文件
- `src/baselines/tsdiff_core/model.py` (重写)
- `smoke_test_tsdiff_trajectory.py`
- `docs/B2_MODEL_PREPARATION_REPORT.md`
- `docs/B2_BASELINE_REORGANIZATION_PLAN.md`
- `docs/TSLIB_MODEL_SELECTION.md`

### 修改文件
- 无（TSDiff 是完全重写）

### 外部仓库
- `external/TSLib/` (已克隆)

---

## 下一步立即行动

### 立即可做（1-2小时）
1. ✅ TSDiff 改造版已完成
2. ⏳ TSLib iTransformer 适配
3. ⏳ TSLib TimeXer 适配

### 短期可做（2-4小时）
4. ⏳ SurvTraj 快速接入
5. ⏳ 统一评估管道

### 中期可做（1-2天）
6. ⏳ SSSD 完整适配
7. ⏳ TSTR 协议实现

---

**报告完成时间**: 2026-03-11 13:05  
**核心成果**: TSDiff 已升级支持层2，TSLib 模型已选定并克隆
