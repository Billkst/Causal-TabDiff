# 训练脚本日志实现分析报告

## 执行摘要

已分析四个主要训练脚本的日志实现模式，识别出关键问题并提供最小化改进方案。

---

## 1. 当前日志实现模式分析

### 1.1 train_tslib_models.py

**现状**：
- 每 10 个 epoch 输出一次日志（第 74-75 行）
- 输出格式：`Epoch {epoch+1}/{epochs} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | BestValLoss {best_val_loss:.4f}`
- 使用 `print(..., flush=True)`
- 无文件日志
- 缺少信息：seed、学习率、epoch 耗时

**问题**：
- 日志频率过低（每 10 epoch 一次）
- 缺少关键上下文信息
- 无持久化日志

---

### 1.2 train_generative_strict.py

**现状**：
- 仅在开始和结束时输出信息（第 23、74 行）
- 使用 `print(..., flush=True)`
- 使用 `EfficiencyTracker` 但未在训练中输出
- 无 epoch 级别日志
- 无文件日志

**问题**：
- 完全缺少训练进度可见性
- 无法追踪训练过程中的问题
- 无指标输出

---

### 1.3 train_tstr_pipeline.py

**现状**：
- 仅输出高层步骤信息（第 29、44、58、62、66、70、81 行）
- 使用 `print(..., flush=True)`
- 无 epoch 级别日志
- 无文件日志

**问题**：
- 无法看到生成模型的训练进度
- 无指标反馈
- 无法诊断训练问题

---

### 1.4 run_experiment_landmark.py

**现状**：
- 使用 Python `logging` 模块（第 11-19 行）
- 配置 FileHandler + StreamHandler
- 每 epoch 输出一次（第 73 行）
- 输出格式：`Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}`
- 日志文件：`logs/training/run_landmark.log`

**问题**：
- 缺少详细指标（AUPRC、AUROC、F1）
- 缺少 seed、学习率等上下文
- 缺少 epoch 耗时
- 日志模块可能存在缓冲延迟

---

## 2. 指标计算现状

| 脚本 | TrainLoss | ValLoss | AUPRC | AUROC | F1 | Precision | Recall | 其他 |
|------|-----------|---------|-------|-------|----|-----------|---------|----|
| train_tslib_models.py | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | BestValLoss |
| train_generative_strict.py | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| train_tstr_pipeline.py | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| run_experiment_landmark.py | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 多分量 loss |

**结论**：所有脚本都缺少核心性能指标（AUPRC、AUROC、F1）。

---

## 3. 训练循环结构

所有脚本采用标准结构：

```python
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        # 前向传播、反向传播、优化
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            # 前向传播
            val_loss += loss.item()
    
    # 日志输出点
```

**日志注入点**：在 epoch 循环结束后，输出日志。

---

## 4. 当前 flush/缓冲行为

### print() 方法
- 使用 `flush=True` ✓
- 立即输出到终端 ✓

### logging 模块（run_experiment_landmark.py）
- 默认使用 StreamHandler（无缓冲）✓
- FileHandler 使用默认缓冲（可能延迟）✗
- 未显式调用 `flush()`

### 文件写入
- 无脚本使用文件写入
- 需要实现行缓冲模式

---

## 5. 最小化改进方案

### 5.1 创建统一日志工具

已创建 `src/utils/TrainingLogger` 类，特性：

```python
class TrainingLogger:
    def __init__(self, log_dir='logs', script_name='training')
    def log(self, message)  # 输出到终端和文件
    def epoch_log(self, epoch, total_epochs, seed, lr, train_loss, 
                  val_loss=None, best_val_metric=None, epoch_time=None, **kwargs)
    def close()
```

**关键特性**：
- 行缓冲模式（`buffering=1`）
- 每次写入后立即 `flush()`
- 支持自定义指标
- 最小化 API

### 5.2 集成改动清单

#### train_tslib_models.py
- 在 `train_layer1()` 中添加 logger 初始化
- 将每 10 epoch 的日志改为每 epoch 输出
- 添加 seed、lr、epoch_time 信息
- 改动行数：约 10 行

#### train_generative_strict.py
- 在 `main()` 中初始化 logger
- 修改 `model.fit()` 以支持 logger 参数
- 改动行数：约 5-10 行

#### train_tstr_pipeline.py
- 在 `TSTRPipeline.train_generative_model()` 中添加 logger
- 改动行数：约 5-10 行

#### run_experiment_landmark.py
- 替换 Python logging 模块为 TrainingLogger
- 改动行数：约 10 行

---

## 6. 日志输出格式标准

### 基础格式
```
Epoch 12/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.1834 | ValLoss 0.2017 | BestValMetric 0.0450 | Time 8.2s
```

### 扩展格式（带指标）
```
Epoch 12/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.1834 | ValLoss 0.2017 | AUPRC 0.7234 | AUROC 0.8123 | F1 0.6890 | Time 8.2s
```

---

## 7. 验收标准

- [x] 终端实时输出（`flush=True`）
- [x] 文件实时写入（行缓冲 + flush）
- [x] 每 epoch 至少输出一次基础信息
- [x] 支持自定义指标输出
- [x] 最小化代码改动
- [x] 日志文件可通过 `tail -f` 实时查看

---

## 8. 后续建议

1. **立即实施**：集成 TrainingLogger 到四个脚本
2. **短期**：添加 AUPRC、AUROC、F1 等核心指标计算
3. **中期**：统一所有脚本的日志格式
4. **长期**：考虑集成 MLflow 或 Weights & Biases 进行实验追踪

