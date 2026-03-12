# B2-2B: Baseline 体系重组方案

**日期**: 2026-03-11  
**状态**: 待执行  
**目标**: 补充并重组 baseline 体系，支持层1（2-year risk）和层2（risk trajectory）

---

## 一、固定前提（不可更改）

### 1.1 任务定义
- **层1**: landmark-conditioned 2-year first-lung-cancer risk prediction
- **层2**: future risk trajectory generation

### 1.2 数据口径
- 主建模表: `unified_person_landmark_table.pkl`
- Split: pid-level (60/20/20)
- 标签: `y_2year`, `trajectory_target`, `trajectory_valid_mask`

### 1.3 评估口径
- 评估入口: `evaluate_model.py`
- 阈值选择: 验证集 F1 最大化
- 测试集: 固定阈值

### 1.4 已有 Baseline（保留）
- LR, XGBoost, BRF, CausalForest
- STaSy, TabSyn, TabDiff, TSDiff

---

## 二、GitHub 搜索结果汇总

### 2.1 A 类候选（强推荐，可参加层1+层2）

#### 1. Time-Series-Library (TSLib)
- **仓库**: https://github.com/thuml/Time-Series-Library
- **Stars**: 11.7k | **更新**: 2026年1月
- **类型**: Transformer（TimesNet/iTransformer/PatchTST/TimeMixer/TimeXer）
- **训练代码**: ✅ 完整
- **判别式/生成式**: 判别式
- **支持时序**: ✅
- **支持轨迹**: ✅ 多步预测
- **支持风险预测**: ✅
- **推荐理由**:
  - 清华大学开源，包含最新 SOTA 模型
  - 支持 5 大任务（预测/填补/异常检测/分类/生成）
  - 代码结构清晰，易于修改
  - 活跃维护，社区支持好

#### 2. SSSD
- **仓库**: https://github.com/AI4HealthUOL/SSSD
- **Stars**: 332 | **更新**: 2025年5月
- **类型**: Diffusion + Structured State Space Models
- **训练代码**: ✅ 完整
- **判别式/生成式**: 生成式
- **支持时序**: ✅ 专为时序表格设计
- **支持轨迹**: ✅ 支持预测未来时间步
- **支持风险预测**: ✅ 可改造为条件生成
- **推荐理由**:
  - 专为时序表格数据设计，支持长期依赖建模
  - 论文发表在 TMLR，方法成熟
  - 与 TSDiff 形成对比（SSM vs MLP）

#### 3. SurvTraj
- **仓库**: https://github.com/NTAILab/SurvTraj
- **Stars**: 2 | **更新**: 2024年2月
- **类型**: Autoencoder for Survival Trajectories
- **训练代码**: ✅ 完整
- **判别式/生成式**: 生成式
- **支持时序**: ✅
- **支持轨迹**: ✅ 专门生成生存轨迹
- **支持风险预测**: ✅ 生存分析
- **推荐理由**:
  - 专门为生存轨迹生成设计
  - 医疗应用背景，可解释性强
  - 直接适配层2任务

### 2.2 B 类候选（可部分适配）

#### 4. Darts
- **仓库**: https://github.com/unit8co/darts
- **Stars**: 9.2k
- **类型**: 统计+深度学习（ARIMA/LSTM/Transformer/N-BEATS/TFT）
- **推荐理由**: API 友好，适合快速原型验证

#### 5. rotroff-lab Deep Learning EHR Trajectories
- **仓库**: https://github.com/rotroff-lab/Deep_learning_longitudinal_EHR_trajectories
- **Stars**: 4
- **类型**: 多种架构（CNN-RNN/LSTM-FCN/ResNet/Autoencoder）
- **推荐理由**: 专门为纵向 EHR 轨迹设计

---

## 三、最终推荐模型（3个）

### 3.1 首选：Time-Series-Library
- **参加层级**: 层1 + 层2
- **接入难度**: 中等
- **与现有模型关系**: 作为判别式 baseline，对比生成式方法

### 3.2 次选：SSSD
- **参加层级**: 层1 + 层2
- **接入难度**: 中等
- **与现有模型关系**: 同为 Diffusion，但架构不同（SSM vs MLP）

### 3.3 备选：SurvTraj
- **参加层级**: 层2（主要）
- **接入难度**: 低
- **与现有模型关系**: 专注轨迹生成，补充层2多样性

---

## 四、模型层级支持矩阵

| 模型 | 层1 (2-year risk) | 层2 (risk trajectory) | 说明 |
|------|------------------|---------------------|------|
| **Time-Series-Library** | ✅ | ✅ | 直接支持多步预测 |
| **SSSD** | ✅ | ✅ | 条件生成支持两层 |
| **SurvTraj** | ⚠️ 部分 | ✅ | 主要用于轨迹 |
| **TSDiff（现有）** | ✅ | ⚠️ 需改造 | 当前只支持单步 |

---

## 五、TSDiff 改造方案（参加层2）

### 5.1 当前状态
- **输入**: `[Batch, Seq=1, Features]` - 单时间点
- **输出**: `[Batch, Seq=1, Features]` - 单时间点生成
- **问题**: 无法输出轨迹序列

### 5.2 改造方案

#### 方案 A：扩展序列长度（推荐）

**需要新增的组件**:
1. **Condition Encoder**: 编码历史 landmark 特征
2. **训练数据格式调整**: `(x_history, y_trajectory)` → 序列
3. **输出格式对齐**: 对接 `trajectory_target` 和 `trajectory_valid_mask`

**改造工作量**:
- 代码修改: 约 100-150 行
- 训练调整: 需要重新训练（支持序列输出）
- 评估适配: 需要对接 trajectory 评估逻辑

---

## 六、新 Baseline 体系分层组织

### 层级 1：Direct Predictive Baselines（判别式，层1）
- Logistic Regression
- XGBoost
- Balanced Random Forest
- CausalForest
- **Time-Series-Library (TimeXer/iTransformer)** ← 新增

### 层级 2：Generative Baselines - TSTR（生成式，层1）
- STaSy
- TabSyn
- TabDiff
- TSDiff（当前版本）

### 层级 3：Trajectory-Capable Advanced Baselines（层1+层2）
- **TSDiff (改造版)** ← 升级
- **SSSD** ← 新增
- **Time-Series-Library (多步预测)** ← 新增
- **SurvTraj** ← 新增

### 汇报结构
```
表1: Direct Predictive Baselines (层1)
表2: Generative/TSTR Baselines (层1)
表3: Trajectory Baselines (层2)
表4: Joint Performance (层1+层2)
```

---

## 七、实施顺序

### Phase 1: 快速验证（1-2天）
1. **Time-Series-Library**
   - 接入 TimeXer 或 iTransformer
   - 验证层1性能
   - 测试多步预测（层2）

### Phase 2: TSDiff 改造（2-3天）
2. **TSDiff 升级**
   - 实现条件编码器
   - 支持序列输出
   - 对接 trajectory 评估

### Phase 3: 补充对比（3-5天）
3. **SSSD** - 提供 Diffusion 对比
4. **SurvTraj** - 补充医疗专用方法

---

## 八、关键决策点

### 8.1 为什么选择这3个模型？
- **Time-Series-Library**: SOTA 性能，代码成熟，社区活跃
- **SSSD**: 方法新颖，与 TSDiff 形成对比
- **SurvTraj**: 医疗专用，可解释性强

### 8.2 为什么不选择其他模型？
- **TabDDPM/TabSyn/TabDiff**: 纯静态表格，不支持时序
- **GluonTS**: 学习曲线陡，非表格专用
- **Darts**: API 友好但需要大量适配工作

### 8.3 TSDiff 为什么必须改造？
- 当前只支持单时间点生成
- 层2任务需要输出未来多步风险轨迹
- 改造成本可控（100-150行代码）

---

## 九、风险与缓解

### 9.1 风险
1. **Time-Series-Library 数据格式不兼容**: 需要适配当前 landmark 格式
2. **SSSD 条件生成改造复杂**: 可能需要修改核心架构
3. **TSDiff 改造后性能下降**: 序列输出可能影响生成质量

### 9.2 缓解措施
1. 先用小数据集验证可行性
2. 保留原始 TSDiff 版本作为 baseline
3. 分阶段实施，每个模型独立评估

---

## 十、成功标准

### 10.1 Phase 1 成功标准
- Time-Series-Library 在层1上 AUROC > 0.55
- 多步预测在层2上能输出完整轨迹

### 10.2 Phase 2 成功标准
- TSDiff 改造版能输出 6 步轨迹
- 轨迹评估指标正常计算

### 10.3 Phase 3 成功标准
- 所有新模型完成 5 seeds 训练
- 生成完整的对比表格

---

**方案制定完成，等待执行确认。**
