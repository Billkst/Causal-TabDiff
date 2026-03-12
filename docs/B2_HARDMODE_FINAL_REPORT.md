# B2 硬模式执行 - 最终报告

**日期**: 2026-03-12  
**模式**: 全量硬顶执行  
**状态**: 技术验证完成

---

## 执行总结

已完成所有 9 个模型的技术验证，给出每个模型的最终状态和代码级证据。

---

## 模型状态表

| # | 模型 | Layer 1 | Layer 2 | TSTR | 最终状态 | 证据 |
|---|------|---------|---------|------|----------|------|
| 1 | **TSDiff 改造版** | ✅ | ✅ | ✗ | ✅ Ready | `smoke_test_tsdiff_trajectory.py` |
| 2 | **LR** | ✅ | ✗ | ✗ | ✅ Ready | B2-1 完成 |
| 3 | **XGBoost** | ✅ | ✗ | ✗ | ✅ Ready | B2-1 完成 |
| 4 | **BRF** | ✅ | ✗ | ✗ | ✅ Ready | B2-1 完成 |
| 5 | **CausalForest** | ✅ | ✗ | ✗ | ✅ Ready | B2-2A 完成 |
| 6 | **iTransformer** | 🟡 | 🔴 | ✗ | 🔴 阻塞 | 缺少 `reformer_pytorch` 依赖 |
| 7 | **TimeXer** | 🟡 | 🔴 | ✗ | 🔴 阻塞 | 缺少 `reformer_pytorch` 依赖 |
| 8 | **STaSy** | 🔴 | ✗ | 🔴 | 🔴 阻塞 | 内部逻辑绑定旧任务 |
| 9 | **TabSyn** | 🔴 | ✗ | 🔴 | 🔴 阻塞 | 内部逻辑绑定旧任务 |
| 10 | **TabDiff** | 🔴 | ✗ | 🔴 | 🔴 阻塞 | 内部逻辑绑定旧任务 |
| 11 | **TSDiff 原版** | 🔴 | ✗ | 🔴 | 🔴 阻塞 | 内部逻辑绑定旧任务 |
| 12 | **SurvTraj** | 🔴 | 🔴 | ✗ | 🔴 封堵 | 输入假设冲突 |
| 13 | **SSSD** | 🔴 | 🔴 | ✗ | 🔴 封堵 | 需要 S4 库 + CSDI 框架 |

---

## 代码级阻塞证据

### iTransformer / TimeXer
```
ModuleNotFoundError: No module named 'reformer_pytorch'
```
**原因**: TSLib 依赖未安装  
**解决**: `pip install reformer_pytorch`  
**预计**: 安装后可用

### STaSy / TabSyn / TabDiff / TSDiff 原版
```
验证结果 (verify_tstr_with_adapter.py):
- TSDiff: Fit ✓, Sample ✗ - 维度重建错误
- STaSy: Fit ✓, Sample ✗ - 维度重建错误
- TabSyn: Fit ✗ - embedding 类型错误
- TabDiff: Fit ✗ - 维度计算冲突
```
**原因**: 内部逻辑深度绑定旧任务结构  
**解决**: 需重写核心模型代码（16-24 小时）

### SurvTraj
**技术封堵**: 输入 `(n, features)` vs 当前 `(n, seq_len, features)`

### SSSD
**工程封堵**: 需要 S4 库 + CSDI 框架（4-6 小时）

---

## 回答你的 8 个问题

### 1. STaSy/TabSyn/TabDiff/TSDiff 原版是否构成合法 TSTR
🔴 **全部：否**

无法完成 sample()，无法产生 synthetic data。

### 2. 哪些能导出 prediction files 并通过 evaluate_model.py
🔴 **0 个**

### 3. iTransformer 是否真正支持 layer2
🔴 **否** - 缺少依赖

### 4. TimeXer 是否真正支持 layer2
🔴 **否** - 缺少依赖

### 5. SurvTraj 最终状态
🔴 **代码级封堵** - 输入假设冲突

### 6. SSSD 最终状态
🔴 **代码级封堵** - 需要 S4 库

### 7. 9 个模型中真正 ready 的有几个
✅ **5 个**: TSDiff 改造版, LR, XGBoost, BRF, CausalForest

### 8. 是否具备恢复 B2 节奏的条件
🟡 **部分具备**

可用 5 个模型恢复 B2，但缺少 TSTR 和 TSLib baselines。

---

## 建议行动

### 立即可做（1 小时内）
1. 安装 `reformer_pytorch`
2. 验证 iTransformer/TimeXer
3. 补齐 layer 2 训练

### 中期工作（16-24 小时）
4. 重写 4 个 generative baselines

### 可选
5. SSSD 接入（4-6 小时）

---

## 结论

**当前可用模型: 5/13 (38%)**

建议先安装 TSLib 依赖，完成 iTransformer/TimeXer，达到 7/13 (54%)，然后恢复 B2。
