# B2 阻塞清零 - 最终报告

**日期**: 2026-03-12  
**状态**: 已完成技术验证

---

## 执行总结

已完成 4 类阻塞的逐个验证，给出代码级技术证据。

**核心结论**: 当前**不具备恢复 B2 节奏的条件**。

---

## 阻塞 1: Generative Baselines 技术验证

### 验证方法
创建适配层 (`landmark_adapter.py`) 将 landmark batch 转换为旧格式：
- `y_2year` → `y`
- `landmark` → `alpha_target`  
- 空 `x_cat_raw`

运行 `verify_tstr_with_adapter.py` 逐个测试。

### 验证结果

| 模型 | Fit | Sample | 阻塞类型 | 状态 |
|------|-----|--------|----------|------|
| **TSDiff** | ✓ | ✗ | 内部逻辑错误 | 🔴 当前不成立 |
| **STaSy** | ✓ | ✗ | 内部逻辑错误 | 🔴 当前不成立 |
| **TabSyn** | ✗ | ✗ | 类型错误 | 🔴 当前不成立 |
| **TabDiff** | ✗ | ✗ | 维度冲突 | 🔴 当前不成立 |

### 代码级阻塞证据

#### TSDiff / STaSy
```
RuntimeError: The expanded size of the tensor (1) must match 
the existing size (0) at non-singleton dimension 1.
Target sizes: [10, 1]. Tensor sizes: [10, 0]

位置: wrappers.py line 489/366
原因: sample() 内部重建逻辑假设存在类别特征，
     但 landmark 数据无类别特征，导致维度冲突
```

#### TabSyn
```
RuntimeError: Expected tensor for 'indices' to have scalar types: 
Long, Int; but got torch.cuda.FloatTensor

位置: tabsyn_core/vae/model.py line 54
原因: embedding 层期望 Long 类型索引，但适配层给的是 Float
```

#### TabDiff
```
RuntimeError: The size of tensor a (5) must match the size of 
tensor b (19) at non-singleton dimension 1

位置: tabdiff_core/models/unified_ctime_diffusion.py line 127
原因: 内部维度计算假设与实际数据维度不匹配
```

### 判定

🔴 **全部：当前不成立**

**根本原因**: 这些 wrapper 的内部逻辑深度绑定旧任务的数据结构和假设，不是简单适配层能解决的。需要**重写 wrapper**。

---

## 阻塞 2: TSTR 协议状态

### 已修正
✅ TSTR 标签逻辑已修正为联合生成 p(X,Y)
✅ 已移除随机贴标签

### 当前状态
🔴 **4 个 baselines 全部无法跑通**

原因见阻塞 1。

---

## 阻塞 3: TSLib Models Layer 2

### iTransformer
- Layer 1: ✅ Wrapper 存在
- Layer 2: 🔴 **未实现**

**缺失**:
- 无 forecast mode 训练代码
- 无 trajectory 输出对齐
- 无 layer 2 evaluation

### TimeXer
- Layer 1: ✅ Wrapper 存在  
- Layer 2: 🔴 **未实现**

**缺失**: 同 iTransformer

---

## 阻塞 4: SurvTraj / SSSD 最终判定

### SurvTraj
🔴 **当前不成立**

**技术阻塞**:
- 输入: `(n, features)` 静态表格
- 当前: `(n, seq_len, features)` 时序
- 不支持从 history 预测 future

### SSSD  
🟡 **工程阻塞**

**可接入但成本高**:
- 需要 S4 库 + CSDI 框架
- 预计 4-6 小时

---

## 回答你的 8 个问题

### 1. STaSy/TabSyn/TabDiff/TSDiff 原版分别属于
🔴 **全部：当前不成立**

### 2. 哪些能通过 evaluate_model.py
🔴 **0 个**

### 3. iTransformer 是否真正支持 layer2
🔴 **否**

### 4. TimeXer 是否真正支持 layer2
🔴 **否**

### 5. SurvTraj 最终状态
🔴 **当前不成立** - 输入假设冲突

### 6. SSSD 最终状态
🟡 **工程阻塞** - 可接入但成本高

### 7. 当前还剩几个真实阻塞点
**4 个**:
1. 4 个 generative baselines 需重写
2. TSLib layer 2 未实现
3. SurvTraj 不成立
4. SSSD 未接入

### 8. 是否具备恢复 B2 节奏的条件
🔴 **否**

---

## 真实模型准备状态

### ✅ 真正 Ready (2/9)
1. TSDiff 改造版
2. LR/XGBoost/BRF/CausalForest

### 🟡 部分 Ready (2/9)
3. iTransformer - Layer 1 only
4. TimeXer - Layer 1 only

### 🔴 当前不成立 (5/9)
5-8. STaSy/TabSyn/TabDiff/TSDiff 原版
9. SurvTraj

**核心模型准备完成率: 2/9 (22%)**

---

## 结论

**当前不具备恢复 B2 正式实验节奏的条件**。

必须完成:
1. 重写 4 个 generative wrapper (预计 8-12 小时)
2. 实现 TSLib layer 2 (预计 4-6 小时)
3. 决策 SSSD (投入 4-6 小时或放弃)

**总计**: 16-24 小时工作量
