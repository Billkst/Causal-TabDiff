# 训练脚本日志集成指南

## 概述
已创建 `src/utils/TrainingLogger` 工具类，支持实时终端输出和文件日志记录。

## 集成方式

### 1. train_tslib_models.py

**改动位置**：`train_layer1()` 函数

```python
from src.utils import TrainingLogger

def train_layer1(model, train_loader, val_loader, epochs, device, seed, lr=1e-3):
    logger = TrainingLogger('logs', 'train_tslib_models')
    
    # ... 现有代码 ...
    
    with tracker.track_training():
        for epoch in range(epochs):
            epoch_start = time.time()
            # ... 训练代码 ...
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # 每个 epoch 输出日志
            logger.epoch_log(
                epoch=epoch+1,
                total_epochs=epochs,
                seed=seed,
                lr=lr,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val_metric=best_val_loss,
                epoch_time=epoch_time
            )
    
    logger.close()
    return model, tracker
```

**调用改动**：
```python
model, tracker = train_layer1(model, train_loader, val_loader, args.epochs, device, args.seed, args.lr)
```

---

### 2. train_generative_strict.py

**改动位置**：`main()` 函数中的 `model.fit()` 调用

需要修改 `model.fit()` 方法以支持日志回调，或在外层包装：

```python
from src.utils import TrainingLogger

def main():
    # ... 现有代码 ...
    
    logger = TrainingLogger('logs', 'train_generative_strict')
    logger.log(f"=== 训练 {args.model.upper()}_strict (Seed {args.seed}) ===")
    
    with tracker.track_training():
        model.fit(train_loader, args.epochs, device, logger=logger)
    
    logger.close()
```

或者在 `model.fit()` 内部集成日志（见下文）。

---

### 3. train_tstr_pipeline.py

**改动位置**：`TSTRPipeline.train_generative_model()` 方法

在 `baselines/tstr_pipeline.py` 中修改：

```python
def train_generative_model(self, train_loader, epochs, device, logger=None):
    if logger is None:
        from src.utils import TrainingLogger
        logger = TrainingLogger('logs', 'train_tstr_pipeline')
    
    for epoch in range(epochs):
        # ... 训练代码 ...
        
        logger.epoch_log(
            epoch=epoch+1,
            total_epochs=epochs,
            seed=self.seed,
            lr=self.current_lr,
            train_loss=train_loss,
            epoch_time=epoch_time
        )
    
    logger.close()
```

---

### 4. run_experiment_landmark.py

**改动位置**：替换 Python logging 模块

```python
from src.utils import TrainingLogger

def main():
    # ... 现有代码 ...
    
    logger = TrainingLogger('logs/training', 'run_landmark')
    logger.log(f"Device: {device}")
    logger.log("Training started")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # ... 训练代码 ...
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.epoch_log(
            epoch=epoch+1,
            total_epochs=args.epochs,
            seed=args.seed,
            lr=1e-3,  # 从 optimizer 获取
            train_loss=avg_loss
        )
    
    logger.log("Training complete")
    logger.close()
```

---

## 日志输出格式示例

```
Epoch 12/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.1834 | ValLoss 0.2017 | BestValMetric 0.0450 | Time 8.2s
```

## 日志文件位置

- `logs/train_tslib_models.log`
- `logs/train_generative_strict.log`
- `logs/train_tstr_pipeline.log`
- `logs/training/run_landmark.log`

## 实时查看日志

```bash
tail -f logs/train_tslib_models.log
```

## 关键特性

✓ 终端实时输出（`flush=True`）
✓ 文件实时写入（行缓冲模式）
✓ 每 epoch 输出基础信息
✓ 支持自定义指标输出
✓ 最小化代码改动
