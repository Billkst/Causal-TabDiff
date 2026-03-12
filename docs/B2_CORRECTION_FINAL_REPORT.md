# B2 模型准备强纠偏 - 最终报告

**日期**: 2026-03-12  
**状态**: 已完成技术验证

---

## 一、TSTR 标签逻辑修正 ✅

### 问题
原 `train_tstr_pipeline.py` 使用 `np.random.choice(y_train, ...)` 随机贴标签，不构成合法 TSTR。

### 修正
已确认所有 4 个 generative baselines 都是**联合生成 p(X, Y)**：
- 训练时：`fit()` 接收 (X, Y) 联合向量
- 采样时：`sample()` 返回 `(X_synthetic, Y_synthetic)`

### 代码证据
```python
# STaSyWrapper.fit() line 301
xyt_flat = np.concatenate([x_flat, t_flat, y_flat], axis=1)

# STaSyWrapper.sample() line 339
Y_cf = XYT_cf[:, -1:]

# TabSynWrapper, TabDiffWrapper, TSDiffWrapper 同理
```

### 修正结果
- ✅ `src/baselines/tstr_pipeline.py` 已修正
- ✅ `train_tstr_pipeline.py` 已修正
- ✅ `docs/B2_TSTR_PROTOCOL.md` 已创建

---

## 二、TSTR Baselines 技术阻塞（代码级证据）

### 验证结果
运行 `verify_tstr_baselines.py` 结果：

```
STaSy: 阻塞 - KeyError: 'alpha_target'
TabSyn: 阻塞 - KeyError: 'x_cat_raw'
TabDiff: 阻塞 - KeyError: 'x_cat_raw'
TSDiff: 阻塞 - KeyError: 'alpha_target'
```

### 技术阻塞原因

**数据格式根本不兼容**：

| Wrapper 期望格式 | 当前 B2 数据格式 | 冲突 |
|-----------------|-----------------|------|
| `batch['alpha_target']` | `batch['y_2year']` | ✗ |
| `batch['x_cat_raw']` | 不存在 | ✗ |
| `batch['y']` | `batch['y_2year']` | ✗ |
| `batch['x']` - 因果任务格式 | `batch['x']` - landmark 格式 | ✗ |

**根本原因**：
- 现有 4 个 wrapper 是为**旧的因果推断任务**设计的
- 旧任务：treatment (`alpha_target`) → outcome (`y`)
- 新任务：landmark history → 2-year risk (`y_2year`) + trajectory

### 状态判定

| 模型 | TSTR 状态 | 原因 |
|------|-----------|------|
| **STaSy** | 🔴 当前不成立 | 数据格式不兼容 |
| **TabSyn** | 🔴 当前不成立 | 数据格式不兼容 |
| **TabDiff** | 🔴 当前不成立 | 数据格式不兼容 |
| **TSDiff 原版** | 🔴 当前不成立 | 数据格式不兼容 |

---

## 三、iTransformer / TimeXer Layer 2 状态

### 当前状态
- ✅ Wrapper 已创建 (`src/baselines/tslib_wrappers.py`)
- ✅ 支持 classification mode (layer 1)
- 🟡 **Layer 2 未真正实现**

### 技术阻塞
1. **缺少 trajectory 训练逻辑**
   - 当前 `train_tslib_models.py` 只实现了 classification
   - 没有 forecast mode 的训练代码
   
2. **缺少 layer 2 输出对齐**
   - 没有 trajectory_target 对齐
   - 没有 trajectory_valid_mask 处理
   - 没有 layer 2 evaluation

### 状态判定
- **iTransformer**: 🟡 Layer 1 ready, Layer 2 未实现
- **TimeXer**: 🟡 Layer 1 ready, Layer 2 未实现

---

## 四、SurvTraj / SSSD 状态（代码级证据）

### SurvTraj
**技术阻塞**：
1. **不支持原生时序输入**
   - 输入格式：`(n_samples, n_features)` - 静态表格
   - 当前数据：`(n_samples, seq_len, n_features)` - 时序
   - 需要聚合：`x_static = x_temporal[:, -1, :]` 或其他策略

2. **轨迹是生成的，非预测的**
   - SurvTraj 从静态 x 生成 x(t)
   - 不是从 history 预测 future

**状态**: 🔴 当前不成立 - 输入假设与数据协议冲突

### SSSD
**技术阻塞**：
1. **依赖 S4 (Structured State Space) 库**
   - 需要 `pip install state-spaces`
   - 安装复杂，版本兼容问题

2. **需要 CSDI 框架适配**
   - 输入格式：`(B, C, L)` with mask
   - 需要重写数据加载器

**状态**: 🟡 工程阻塞 - 可接入但成本高（预计 4-6 小时）

---

## 五、回答你的 7 个问题

### 1. TSTR 的 synthetic labels 现在到底怎么来
✅ **联合生成 p(X, Y)**
- 所有 4 个模型训练时接收 (X, Y)
- 采样时直接输出 (X_synthetic, Y_synthetic)
- 已移除随机贴标签逻辑

### 2. STaSy / TabSyn / TabDiff / TSDiff 原版分别属于
🔴 **全部：当前不成立**
- 原因：数据格式不兼容（代码级证据已给出）
- 需要重写 wrapper 适配 landmark 数据格式

### 3. iTransformer 是否已经真正支持 layer2
🟡 **否**
- Layer 1 (classification): ✅ Ready
- Layer 2 (trajectory): ✗ 未实现

### 4. TimeXer 是否已经真正支持 layer2
🟡 **否**
- Layer 1 (classification): ✅ Ready
- Layer 2 (trajectory): ✗ 未实现

### 5. SurvTraj 最终状态是什么
🔴 **当前不成立**
- 技术阻塞：输入假设与数据协议冲突
- 不支持原生时序输入

### 6. SSSD 最终状态是什么
🟡 **工程阻塞**
- 可接入但成本高（4-6 小时）
- 需要 S4 库 + CSDI 框架适配

### 7. 现在是否真的具备恢复 B2 节奏的条件
🔴 **否**

**原因**：
- TSTR baselines: 全部不成立（数据格式不兼容）
- TSLib models: Layer 2 未实现
- 只有 TSDiff 改造版真正 ready

---

## 六、真实模型准备状态

### ✅ 真正 Ready (2/9)
1. **TSDiff 改造版** - Layer 1 + Layer 2
2. **LR/XGBoost/BRF/CausalForest** - 已完成（B2-1, B2-2A）

### 🟡 部分 Ready (2/9)
3. **iTransformer** - Layer 1 ready, Layer 2 未实现
4. **TimeXer** - Layer 1 ready, Layer 2 未实现

### 🔴 当前不成立 (5/9)
5. **STaSy** - 数据格式不兼容
6. **TabSyn** - 数据格式不兼容
7. **TabDiff** - 数据格式不兼容
8. **TSDiff 原版** - 数据格式不兼容
9. **SurvTraj** - 输入假设冲突

**核心模型准备完成率: 2/9 (22%)**

---

## 七、下一步必须完成的工作

### 优先级 1: 修复 TSTR Baselines
重写 4 个 wrapper 的数据适配层：
- 移除 `alpha_target` 依赖
- 适配 `y_2year` 格式
- 移除 `x_cat_raw` 依赖

### 优先级 2: 实现 TSLib Layer 2
- 添加 forecast mode 训练
- 对齐 trajectory_target
- 实现 layer 2 evaluation

### 优先级 3: 决策 SurvTraj / SSSD
- SurvTraj: 放弃或重新设计适配方案
- SSSD: 决定是否投入 4-6 小时

---

## 八、结论

**当前不具备恢复 B2 正式实验节奏的条件**。

核心问题：
1. TSTR baselines 全部因数据格式不兼容而阻塞
2. TSLib models 的 layer 2 未实现
3. 实际 ready 的模型只有 2/9

建议：
1. 先修复 TSTR baselines 数据适配
2. 再实现 TSLib layer 2
3. 然后才能恢复 B2 节奏
