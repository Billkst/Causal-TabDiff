# 日志实现对比表

## 1. 日志频率对比

| 脚本 | 当前频率 | 改进后 | 改进幅度 |
|------|--------|--------|---------|
| train_tslib_models.py | 每 10 epoch | 每 epoch | 10x |
| train_generative_strict.py | 无 epoch 日志 | 每 epoch | ∞ |
| train_tstr_pipeline.py | 无 epoch 日志 | 每 epoch | ∞ |
| run_experiment_landmark.py | 每 epoch | 每 epoch | - |

## 2. 输出信息对比

### train_tslib_models.py

**当前**：
```
Epoch 10/100 | TrainLoss 0.1834 | ValLoss 0.2017 | BestValLoss 0.2017
```

**改进后**：
```
Epoch 10/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.1834 | ValLoss 0.2017 | BestValMetric 0.2017 | Time 8.2s
```

**新增信息**：Seed、LR、Time

---

### run_experiment_landmark.py

**当前**：
```
Epoch 1/100 | Loss: 0.5234
```

**改进后**：
```
Epoch 1/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.5234
```

**新增信息**：Seed、LR、标准化字段名

---

## 3. 文件日志对比

| 脚本 | 当前 | 改进后 |
|------|------|--------|
| train_tslib_models.py | ✗ 无 | ✓ logs/train_tslib_models.log |
| train_generative_strict.py | ✗ 无 | ✓ logs/train_generative_strict.log |
| train_tstr_pipeline.py | ✗ 无 | ✓ logs/train_tstr_pipeline.log |
| run_experiment_landmark.py | ✓ 有 | ✓ logs/training/run_landmark.log |

---

## 4. 缓冲行为对比

| 方式 | 当前 | 改进后 |
|------|------|--------|
| 终端输出 | print(flush=True) ✓ | print(flush=True) ✓ |
| 文件写入 | 无 | 行缓冲 + flush ✓ |
| 实时性 | 部分 | 完全 ✓ |

---

## 5. 指标支持对比

| 指标 | 当前 | 改进后 |
|------|------|--------|
| TrainLoss | ✓ | ✓ |
| ValLoss | ✓ | ✓ |
| AUPRC | ✗ | ✓ (可选) |
| AUROC | ✗ | ✓ (可选) |
| F1 | ✗ | ✓ (可选) |
| 自定义指标 | ✗ | ✓ |

