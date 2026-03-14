# 实现示例

## 示例 1: train_tslib_models.py 改进

### 改动前
```python
def train_layer1(model, train_loader, val_loader, epochs, device, lr=1e-3):
    # ... 代码 ...
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | BestValLoss {best_val_loss:.4f}", flush=True)
```

### 改动后
```python
from src.utils import TrainingLogger

def train_layer1(model, train_loader, val_loader, epochs, device, seed, lr=1e-3):
    logger = TrainingLogger('logs', 'train_tslib_models')
    # ... 代码 ...
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

### 调用改动
```python
# 改动前
model, tracker = train_layer1(model, train_loader, val_loader, args.epochs, device, args.lr)

# 改动后
model, tracker = train_layer1(model, train_loader, val_loader, args.epochs, device, args.seed, args.lr)
```

---

## 示例 2: run_experiment_landmark.py 改进

### 改动前
```python
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training/run_landmark.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

for epoch in range(args.epochs):
    # ... 训练代码 ...
    logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}")
```

### 改动后
```python
from src.utils import TrainingLogger

logger = TrainingLogger('logs/training', 'run_landmark')

for epoch in range(args.epochs):
    # ... 训练代码 ...
    logger.epoch_log(
        epoch=epoch+1,
        total_epochs=args.epochs,
        seed=args.seed,
        lr=1e-3,
        train_loss=total_loss/len(train_loader)
    )

logger.close()
```

---

## 示例 3: 带自定义指标的输出

```python
from src.utils import TrainingLogger
from sklearn.metrics import roc_auc_score, average_precision_score

logger = TrainingLogger('logs', 'train_with_metrics')

for epoch in range(epochs):
    # ... 训练代码 ...
    
    # 计算指标
    y_pred = model.predict(val_loader)
    y_true = get_labels(val_loader)
    auprc = average_precision_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    
    logger.epoch_log(
        epoch=epoch+1,
        total_epochs=epochs,
        seed=42,
        lr=1e-3,
        train_loss=train_loss,
        val_loss=val_loss,
        AUPRC=auprc,
        AUROC=auroc,
        epoch_time=epoch_time
    )

logger.close()
```

---

## 示例 4: 后台运行（nohup）

```bash
# 启动训练
nohup python -u train_tslib_models.py --model itransformer --seed 42 > logs/train_tslib_models.log 2>&1 &

# 实时查看日志
tail -f logs/train_tslib_models.log

# 查看进程
ps aux | grep train_tslib_models.py
```

---

## 日志输出对比

### 改进前
```
Epoch 10/100 | TrainLoss 0.1834 | ValLoss 0.2017 | BestValLoss 0.2017
Epoch 20/100 | TrainLoss 0.1523 | ValLoss 0.1876 | BestValLoss 0.1876
```

### 改进后
```
Epoch 1/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.5234 | ValLoss 0.5412 | BestValMetric 0.5412 | Time 8.2s
Epoch 2/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.4123 | ValLoss 0.4234 | BestValMetric 0.4234 | Time 7.9s
Epoch 3/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.3456 | ValLoss 0.3567 | BestValMetric 0.3567 | Time 8.1s
```

