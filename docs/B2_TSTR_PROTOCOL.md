# B2 TSTR 协议定义

**日期**: 2026-03-12  
**状态**: 已修正

---

## 一、什么是合法的 TSTR

**TSTR (Train on Synthetic, Test on Real)** 要求：
1. 在真实训练集上训练生成模型
2. 生成**合法的 synthetic (X, Y) 联合样本**
3. 在 synthetic (X, Y) 上训练下游分类器
4. 在真实 val/test 上评估

**关键要求**: Synthetic labels 必须来自生成模型，不能随机贴标签。

---

## 二、当前项目的 TSTR 协议

### 方案：联合生成 p(X, Y)

所有 4 个 generative baselines 都采用**联合生成**方式：
- 训练时：将 (X, Y) 作为联合向量输入
- 采样时：直接生成 (X_synthetic, Y_synthetic)

**代码证据**:
- `STaSyWrapper.fit()`: `xyt_flat = np.concatenate([x_flat, t_flat, y_flat], axis=1)`
- `TabSynWrapper.fit()`: Y 作为最后一个 categorical feature
- `TabDiffWrapper.fit()`: Y 作为最后一个 categorical feature
- `TSDiffWrapper.fit()`: `x_combined = torch.cat([x_num, t_tgt, y_tgt], dim=1)`

**采样输出**:
- 所有 4 个模型的 `sample()` 方法都返回 `(X_cf, Y_cf)`
- Y_cf 是生成模型输出的 synthetic label

---

## 三、各模型 TSTR 状态

| 模型 | 联合生成 | TSTR 状态 | 证据 |
|------|----------|-----------|------|
| **STaSy** | ✅ | 标准 TSTR ready | `sample()` 返回 `(X_cf, Y_cf)` |
| **TabSyn** | ✅ | 标准 TSTR ready | `sample()` 返回 `(X_cf, Y_cf)` |
| **TabDiff** | ✅ | 标准 TSTR ready | `sample()` 返回 `(X_cf, Y_cf)` |
| **TSDiff** | ✅ | 标准 TSTR ready | `sample()` 返回 `(X_cf, Y_cf)` |

---

## 四、已修正的错误

### 错误做法（已移除）
```python
# ❌ 错误：随机贴标签
X_synthetic = gen_model.sample(n_samples, device)
y_synthetic = np.random.choice(y_train, size=n_samples, replace=True)
```

### 正确做法（当前实现）
```python
# ✅ 正确：联合生成
X_synthetic, Y_synthetic = gen_model.sample(n_samples, device)
```

---

## 五、TSTR Pipeline 流程

```python
# 1. 训练生成模型
gen_model.fit(train_loader, epochs, device)

# 2. 生成 synthetic (X, Y)
X_syn, Y_syn = gen_model.sample(n_samples, device)

# 3. 训练 XGBoost
classifier = XGBClassifier()
classifier.fit(X_syn, Y_syn)

# 4. 在真实数据上评估
y_pred = classifier.predict_proba(X_test)[:, 1]
```

---

## 六、验证要求

每个 TSTR baseline 必须证明：
1. ✅ `fit()` 接口存在且可用
2. ✅ `sample()` 返回 `(X, Y)` 而非只有 X
3. ✅ Y 是生成模型输出，非随机贴标签
4. ✅ 能在真实数据上训练
5. ✅ 能生成合法 synthetic data
6. ✅ 能导出 prediction file
7. ✅ 能通过 `evaluate_model.py`

---

## 七、结论

**当前 TSTR 协议已修正为标准联合生成方式**。

所有 4 个 generative baselines 都满足标准 TSTR 要求。
