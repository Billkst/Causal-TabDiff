# 日志工具快速参考

## 基本用法

```python
from src.utils import TrainingLogger

# 初始化
logger = TrainingLogger('logs', 'my_script')

# 输出普通日志
logger.log("Training started")

# 输出 epoch 日志
logger.epoch_log(
    epoch=1,
    total_epochs=100,
    seed=42,
    lr=1e-3,
    train_loss=0.5,
    val_loss=0.6,
    best_val_metric=0.6,
    epoch_time=8.2
)

# 输出自定义指标
logger.epoch_log(
    epoch=1,
    total_epochs=100,
    seed=42,
    lr=1e-3,
    train_loss=0.5,
    AUPRC=0.72,
    AUROC=0.81,
    F1=0.69
)

# 关闭
logger.close()
```

## 输出示例

```
Epoch 1/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.5000 | ValLoss 0.6000 | BestValMetric 0.6000 | Time 8.2s
```

## 日志文件位置

- `logs/my_script.log`

## 实时查看

```bash
tail -f logs/my_script.log
```

## 关键特性

- ✓ 终端和文件同时输出
- ✓ 行缓冲 + 自动 flush
- ✓ 支持自定义指标
- ✓ 最小化 API
