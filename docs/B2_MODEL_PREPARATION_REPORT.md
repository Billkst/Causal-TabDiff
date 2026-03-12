# B2 模型准备阶段 - 执行报告

**日期**: 2026-03-11  
**状态**: 进行中  

---

## Workstream 1: TSDiff Trajectory Upgrade ✅ 完成

### 改造内容
1. **新增 condition_dim 参数**: 支持条件编码
2. **新增 condition_encoder**: 编码历史特征 → 潜在表示
3. **修改 train_step()**: 支持条件输入
4. **修改 sample()**: 支持多步轨迹生成

### 当前能力
- ✅ **层1**: 2-year risk（从 trajectory[1] 提取）
- ✅ **层2**: 6-year risk trajectory 完整输出
- ✅ **条件生成**: history → future trajectory
- ✅ **向后兼容**: 无条件模式仍可用

### Smoke Test 结果
```
✓ 原始模式（无条件）: 正常
✓ 轨迹模式（有条件）: 正常
✓ 6年轨迹生成: 正常
✓ 输出形状: [Batch, 6, 1] ✓
```

### 改动文件
- `src/baselines/tsdiff_core/model.py` (完全重写)
- `smoke_test_tsdiff_trajectory.py` (新增)

---

## Workstream 2: TSLib 双模型接入 🔄 进行中

### 模型选择（待确定）
候选模型：
- TimeXer (支持外生变量)
- iTransformer (SOTA 性能)
- PatchTST (高效 Transformer)
- TimeMixer (多尺度混合)

### 选择标准
1. 支持多变量输入
2. 支持多步预测
3. 代码成熟度
4. 接入复杂度

### 当前状态
- ⏳ 待选定 2 个模型
- ⏳ 待克隆 TSLib 仓库
- ⏳ 待适配数据格式

---

## Workstream 3: SSSD / SurvTraj Feasibility ⏳ 待开始

### SSSD
- **仓库**: https://github.com/AI4HealthUOL/SSSD
- **状态**: 待 feasibility audit
- **关键问题**: 
  - 数据格式兼容性
  - 条件生成改造难度
  - 训练稳定性

### SurvTraj
- **仓库**: https://github.com/NTAILab/SurvTraj
- **状态**: 待 feasibility audit
- **关键问题**:
  - 生存分析 → 风险预测转换
  - 输入格式适配
  - 轨迹输出对齐

---

## Workstream 4: 统一封装与评估兼容 ⏳ 待开始

### 待完成
- [ ] 统一 prediction file 格式
- [ ] 对接 evaluate_model.py
- [ ] 对接 trajectory 评估
- [ ] 统一 plots / metrics / logs

---

## 下一步行动

### 立即执行
1. 选定 TSLib 的 2 个模型
2. 克隆 TSLib 仓库并适配
3. SSSD feasibility audit
4. SurvTraj feasibility audit

### 预计时间
- TSLib 接入: 2-3 小时
- SSSD audit: 1 小时
- SurvTraj audit: 1 小时
- 统一封装: 1-2 小时

---

**更新时间**: 2026-03-11 12:58
