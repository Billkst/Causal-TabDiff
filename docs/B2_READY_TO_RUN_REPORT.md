# B2 最终闸门验证报告

**生成时间**: 2026-03-12  
**验证脚本**: `final_gate_test.py`  
**结果文件**: `outputs/b2_gate/final_gate_test_results.json`

---

## 一、闸门验证总结

**通过率**: 2/9 (22.2%)

### ✅ 通过模型 (2)

| 模型 | 实现文件 | 层1 | 层2 | 状态 |
|------|---------|-----|-----|------|
| **CausalForest** | `train_causal_forest_b2.py` | ✅ | ❌ | PASS |
| **iTransformer** | `src/baselines/tslib_wrappers.py` | ✅ | ✅ | PASS |

### ❌ 失败模型 (7)

| 模型 | 失败原因 | 阻塞类型 |
|------|---------|---------|
| **TSDiff** | 样本类别不平衡错误 | code_blocker |
| **STaSy** | 样本类别不平衡错误 | code_blocker |
| **TabSyn_strict** | 类名不匹配 (实际是 `TabSynLandmarkStrictWrapper`) | code_blocker |
| **TabDiff_strict** | 类名不匹配 (实际是 `TabDiffLandmarkStrictWrapper`) | code_blocker |
| **SurvTraj_strict** | 类名不匹配 (实际是 `SurvTrajLandmarkWrapper`) | code_blocker |
| **SSSD_strict** | 类名不匹配 (实际是 `SSSDLandmarkWrapper`) | code_blocker |
| **TimeXer** | 缺少 `classification` 方法 | code_blocker |

---

## 二、详细失败分析

### 2.1 TSDiff / STaSy 失败

**错误信息**:
```
The least populated class in y has only 1 member, which is too few. 
The minimum number of groups for any class cannot be less than 2.
```

**原因**: 闸门测试使用 `debug_n_persons=50`，导致验证集样本极少且类别不平衡。

**修复方案**: 需要增加测试样本量或修改 wrapper 的内部验证逻辑。

### 2.2 所有 strict 模型失败

**错误信息**:
```
cannot import name 'TabSynLandmarkStrict' from 'baselines.tabsyn_landmark_strict'
```

**原因**: 实际类名是 `TabSynLandmarkStrictWrapper`，不是 `TabSynLandmarkStrict`。

**实际类名映射**:
- `TabSynLandmarkStrictWrapper` (不是 TabSynLandmarkStrict)
- `TabDiffLandmarkStrictWrapper` (不是 TabDiffLandmarkStrict)
- `SurvTrajLandmarkWrapper` (不是 SurvTrajLandmarkStrict)
- `SSSDLandmarkWrapper` (不是 SssdLandmarkStrict)

### 2.3 TimeXer 失败

**错误信息**:
```
'Model' object has no attribute 'classification'
```

**原因**: TimeXer 的底层模型缺少 `classification` 方法实现。

---

## 三、已有实验结果

### 3.1 CausalForest (已完成 5-seed)

**位置**: `outputs/retained_baselines_b2/`

**已完成**:
- ✅ seed 42, 52, 62, 72, 82
- ✅ predictions.npz
- ✅ metrics.json
- ✅ model.pkl

**示例结果 (seed 42)**:
```json
{
  "auroc": 0.5065,
  "auprc": 0.0366,
  "f1": 0.0357,
  "precision": 0.0190,
  "recall": 0.2857
}
```

### 3.2 TSLib Layer2 结果

**位置**: `outputs/tslib_layer2/`

**已完成**:
- ✅ `itransformer_seed42_layer2.npz`
- ✅ `timexer_seed42_layer2.npz`

---

## 四、判定结论

根据闸门判定规则：

### 情况 C：TabSyn strict / TabDiff strict / SurvTraj strict / SSSD strict 任一未通过

**结论**: **所有 4 个 strict 模型均未通过闸门验证**

**阻塞原因**: 代码级阻塞 - 类名不匹配

**具体问题**:
1. TabSyn_strict: 类名应为 `TabSynLandmarkStrictWrapper`
2. TabDiff_strict: 类名应为 `TabDiffLandmarkStrictWrapper`
3. SurvTraj_strict: 类名应为 `SurvTrajLandmarkWrapper`
4. SSSD_strict: 类名应为 `SSSDLandmarkWrapper`

**建议**: 
- 不启动 B2 baseline 正式实验
- 需要修复 strict 模型的类名或更新闸门测试脚本
- TSDiff/STaSy 需要修复样本不平衡问题
- TimeXer 需要实现 classification 方法

---

## 五、当前可用的 Baseline 名单

基于闸门验证和已有结果：

### 层1 (2-year risk prediction)

**可立即使用**:
1. ✅ **CausalForest** - 已完成 5-seed 实验
2. ✅ **iTransformer** - 通过闸门验证

**需要修复**:
3. ❌ TSDiff - 样本不平衡问题
4. ❌ STaSy - 样本不平衡问题
5. ❌ TabSyn_strict - 类名不匹配
6. ❌ TabDiff_strict - 类名不匹配
7. ❌ SurvTraj_strict - 类名不匹配
8. ❌ SSSD_strict - 类名不匹配
9. ❌ TimeXer - 缺少 classification 方法

### 层2 (risk trajectory)

**可立即使用**:
1. ✅ **iTransformer** - 通过闸门验证，已有 layer2 结果

**需要修复**:
2. ❌ TimeXer - 缺少 classification 方法

---

## 六、下一步行动

### 选项 A: 仅使用通过闸门的模型

**立即启动**:
- CausalForest (层1) - 已完成
- iTransformer (层1 + 层2) - 需要完成 5-seed

**优点**: 无需修复代码，可立即执行  
**缺点**: baseline 数量太少 (仅 2 个)

### 选项 B: 修复所有模型后再启动

**需要修复**:
1. 修复 strict 模型类名不匹配
2. 修复 TSDiff/STaSy 样本不平衡
3. 修复 TimeXer classification 方法

**优点**: 完整的 baseline 对比  
**缺点**: 需要额外开发时间

---

## 七、最终建议

**不建议立即启动 B2 baseline 正式实验**

**理由**:
1. 7/9 模型未通过闸门验证
2. 所有 strict 模型 (TabSyn/TabDiff/SurvTraj/SSSD) 均失败
3. 仅 2 个模型通过，baseline 对比不充分

**建议行动**:
1. 修复 strict 模型的类名问题 (快速修复)
2. 修复 TSDiff/STaSy 的样本不平衡问题
3. 修复 TimeXer 的 classification 方法
4. 重新运行闸门验证
5. 通过后再启动正式实验

---

**报告结束**
