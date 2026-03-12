# B2 Baseline Gate 验证报告 V8 - Model-Specific Repairs

**生成时间**: 2026-03-12  
**验证版本**: V8 (After Model-Specific Repairs)  
**数据**: `unified_person_landmark_table.pkl` (200 persons debug mode)

---

## 执行摘要

**总体结果**: 6/9 模型通过 (66.7%)

**通过模型 (6)**:
1. ✅ CausalForest - Layer1 PASS
2. ✅ TSDiff - Layer1 PASS  
3. ✅ SurvTraj_strict - Layer1 PASS
4. ✅ SSSD_strict - Layer1 PASS
5. ✅ STaSy - Layer1 PASS (本轮新修复)
6. ✅ TabDiff_strict - Layer1 PASS (本轮新修复)

**失败模型 (3)**:
1. ❌ TabSyn_strict - torch.arange 参数错误
2. ❌ iTransformer - 缺少 predict 方法
3. ❌ TimeXer - 缺少 exog_in 参数

---

## 本轮修复成果 (V7 → V8)

### 成功修复的模型

#### 1. STaSy ✅
**V7 错误**: `AttributeError: 'activation'`  
**根因**: 缺失多个必需配置字段  
**修复内容**:
- 补全 config.model 字段: conditional, embedding_type, fourier_scale, layer_type, scale_by_sigma
- 补全 config.optim 字段: beta1, eps, warmup, grad_clip
- 补全 config.sampling 字段: method, predictor, corrector, snr, n_steps_each, noise_removal, probability_flow
- 禁用 spl (设置 spl=False)
- 保存 config 供 sampling 使用

**修改文件**: `src/baselines/stasy_landmark_wrapper.py`

#### 2. TabDiff_strict ✅
**V7 错误**: `mixed_loss() takes 2 args but 3 given`  
**根因**: 接口签名不匹配  
**修复内容**:
- `categories=[]` → `categories=np.array([])`
- `mixed_loss(xy, None)` → `mixed_loss(xy)` 并解包返回值
- `y_only_model=False` → `y_only_model=None`
- 添加 noise_dist, edm_params, sampler_params
- 修复 sample 调用签名

**修改文件**: `src/baselines/tabdiff_landmark_strict.py`

#### 3. iTransformer (部分修复) ⚠️
**V7 错误**: `No module named 'models.iTransformer'`  
**根因**: TSLib 导入命名空间冲突  
**修复内容**:
- 使用 importlib 动态加载
- 在执行模块前将 TSLib 路径加入 sys.path
- 注册模块到 sys.modules

**修改文件**: `src/baselines/tslib_wrappers.py`  
**当前状态**: 导入成功，但 gate 脚本调用方式不匹配

#### 4. TimeXer (部分修复) ⚠️
**V7 错误**: `No module named 'models.iTransformer'`  
**根因**: 与 iTransformer 共用 TSLib 导入  
**修复内容**: 复用 iTransformer 的修复方案  
**当前状态**: 导入成功，但 gate 脚本调用方式不匹配

---

## 剩余问题分析

### TabSyn_strict ❌
**错误**: `arange() received an invalid combination of arguments`  
**位置**: TabSyn core 内部 (diffusion_utils.py 或 model.py)  
**根因**: torch.arange 调用参数顺序错误，可能是 PyTorch 版本兼容性问题  
**blocker_type**: wrapper_interface_bug (需要进一步调试 sample 阶段)

### iTransformer ❌
**错误**: `'iTransformerWrapper' object has no attribute 'predict'`  
**根因**: Gate 脚本调用了不存在的 predict 方法  
**解决方案**: Gate 脚本应该调用 forward() 而非 predict()  
**blocker_type**: gate_script_bug (非 wrapper 问题)

### TimeXer ❌
**错误**: `TimeXerWrapper.__init__() missing 1 required positional argument: 'exog_in'`  
**根因**: Gate 脚本未提供 exog_in 参数  
**解决方案**: Gate 脚本需要传递 exog_in 参数  
**blocker_type**: gate_script_bug (非 wrapper 问题)

---

## 修复文件清单

### 本轮修改的文件
1. `src/baselines/stasy_landmark_wrapper.py` - STaSy 配置补全
2. `src/baselines/tabdiff_landmark_strict.py` - TabDiff 接口修复
3. `src/baselines/tslib_wrappers.py` - TSLib 导入修复
4. `src/baselines/tabsyn_landmark_strict.py` - TabSyn EDMLoss 调用修复

### 生成的验证文件
1. `repair_stasy_smoke.py` - STaSy 独立验证
2. `repair_tabsyn_strict_smoke.py` - TabSyn 独立验证
3. `repair_tabdiff_strict_smoke.py` - TabDiff 独立验证
4. `repair_itransformer_smoke.py` - iTransformer 独立验证
5. `repair_timexer_smoke.py` - TimeXer 独立验证

### 输出文件
- `outputs/model_repairs/*.json` - 各模型修复结果
- `logs/model_repairs/*.log` - 各模型修复日志
- `docs/model_repairs/*_repair_report.md` - 各模型详细报告
- `outputs/b2_gate/final_gate_test_results_v8.json` - V8 总 gate 结果

---

## 下一步建议

### 立即可执行 (6 个 PASS 模型)
当前 6 个通过的模型已经可以进入 baseline 正式对比实验：
- CausalForest
- TSDiff
- SurvTraj_strict
- SSSD_strict
- STaSy
- TabDiff_strict

### 需要进一步修复 (3 个 FAIL 模型)

#### TabSyn_strict (优先级: 高)
- 调试 torch.arange 参数问题
- 可能需要检查 PyTorch 版本兼容性
- 或者修改 TabSyn core 中的 arange 调用

#### iTransformer & TimeXer (优先级: 中)
- 修复 gate 脚本调用方式
- iTransformer: 使用 forward() 而非 predict()
- TimeXer: 提供 exog_in 参数
- 这两个是 gate 脚本问题，wrapper 本身已修复

---

## 验收标准回顾

### Layer1 PASS 标准 (6/9 满足)
- ✅ create/load success
- ✅ fit/train success
- ✅ prediction export ready
- ✅ evaluate_model ready

### Layer2 PASS 标准 (0/9 满足)
- ⚠️ 当前无模型通过 layer2 验证
- iTransformer 和 TimeXer 有 layer2 能力，但 gate 调用失败

---

## 结论

**当前状态**: 6/9 模型可用于 baseline 实验

**建议行动**:
1. **立即启动**: 使用 6 个 PASS 模型进行 baseline 对比实验
2. **并行修复**: 继续修复 TabSyn_strict (torch.arange 问题)
3. **低优先级**: 修复 gate 脚本对 TSLib 模型的调用方式

**不建议**: 等待全部 9 个模型通过再启动实验 (6 个已足够覆盖主要 baseline 类型)
