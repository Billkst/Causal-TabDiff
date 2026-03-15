# PyTorch 训练与评估最佳实践技术调研报告

## 目录
1. PyTorch 训练循环标准实现模式
2. sklearn.metrics 指标计算正确方法
3. 二分类任务指标计算
4. Python 实时日志输出最佳实践
5. PyTorch 模型保存和加载
6. 训练过程中的验证集评估

---

## 1. PyTorch 训练循环标准实现模式

### 1.1 标准训练循环结构

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据移动到设备
        inputs = batch['x'].to(device)
        targets = batch['y'].to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 1.2 完整训练流程

```python
def train_model(model, train_loader, val_loader, epochs, device, log_path):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 打开日志文件（行缓冲模式）
    log_file = open(log_path, 'w', buffering=1)
    
    best_val_auprc = 0.0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证阶段
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # 更新最佳指标
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 实时输出（终端 + 日志文件）
        msg = (f"Epoch {epoch}/{epochs} | "
               f"TrainLoss {train_loss:.4f} | "
               f"ValLoss {val_loss:.4f} | "
               f"ValAUPRC {val_metrics['auprc']:.4f} | "
               f"ValAUROC {val_metrics['auroc']:.4f} | "
               f"BestAUPRC {best_val_auprc:.4f} | "
               f"Time {epoch_time:.1f}s")
        
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log_file.close()
    return model
```

---

## 2. sklearn.metrics 指标计算正确方法

### 2.1 AUPRC (Average Precision Score)

```python
from sklearn.metrics import average_precision_score

# 正确用法
y_true = np.array([0, 0, 1, 1])
y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])

auprc = average_precision_score(y_true, y_pred_proba)
```

**关键注意事项：**
- 输入必须是概率值（0-1 之间），不是 logits
- y_true 必须是二值标签（0 或 1）
- 当只有一个类别时返回 NaN，需要提前检查

### 2.2 AUROC (ROC AUC Score)

```python
from sklearn.metrics import roc_auc_score

# 正确用法
auroc = roc_auc_score(y_true, y_pred_proba)
```

**关键注意事项：**
- 同样需要概率值，不是 logits
- 对类别不平衡更鲁棒
- 当只有一个类别时会报错，需要提前检查

### 2.3 常见错误避免

```python
# ❌ 错误：使用 logits 而不是概率
logits = model(x)
auprc = average_precision_score(y_true, logits)  # 错误！

# ✅ 正确：先转换为概率
logits = model(x)
proba = torch.sigmoid(logits).cpu().numpy()
auprc = average_precision_score(y_true, proba)

# ❌ 错误：没有检查类别数量
auprc = average_precision_score(y_true, y_pred_proba)  # 可能报错

# ✅ 正确：提前检查
if len(np.unique(y_true)) < 2:
    auprc = np.nan
else:
    auprc = average_precision_score(y_true, y_pred_proba)
```

---

## 3. 二分类任务指标计算方法

### 3.1 完整指标计算函数

```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)

def compute_binary_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    计算二分类任务的完整指标集
    
    Args:
        y_true: 真实标签 (0/1)
        y_pred_proba: 预测概率 (0-1)
        threshold: 二值化阈值
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 确保输入格式正确
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    # 检查类别数量
    if len(np.unique(y_true)) < 2:
        return {
            'auroc': np.nan,
            'auprc': np.nan,
            'f1': np.nan,
            'precision': np.nan,
            'recall': np.nan
        }
    
    # 排序指标（不需要阈值）
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    
    # 二值化预测
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # 阈值相关指标
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'threshold': threshold
    }
```

### 3.2 最优阈值搜索

```python
def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """
    在验证集上搜索最优阈值
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_binary, zero_division=0)
        elif metric == 'balanced_acc':
            cm = confusion_matrix(y_true, y_pred_binary)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                score = 0
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh, best_score
```

---

## 4. Python 实时日志输出最佳实践

### 4.1 核心原则

**必须同时满足：**
1. 终端实时可见
2. 日志文件实时写入
3. 可通过 `tail -f` 实时查看

### 4.2 实现方法

```python
import sys

# 方法1：print + flush=True
print("Training started", flush=True)

# 方法2：文件行缓冲模式
log_file = open('logs/training.log', 'w', buffering=1)
log_file.write("Training started\n")
log_file.flush()  # 确保立即写入

# 方法3：同时输出到终端和文件
def log_message(msg, log_file):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

# 方法4：使用 python -u 启动脚本（无缓冲模式）
# nohup python -u train.py > logs/train.log 2>&1 &
```

### 4.3 完整日志系统

```python
class TrainingLogger:
    def __init__(self, log_path):
        self.log_file = open(log_path, 'w', buffering=1)
        self.start_time = time.time()
    
    def log(self, msg):
        """同时输出到终端和文件"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        self.log_file.write(full_msg + '\n')
        self.log_file.flush()
    
    def log_epoch(self, epoch, total_epochs, metrics):
        """格式化输出 epoch 信息"""
        msg = (f"Epoch {epoch}/{total_epochs} | "
               f"TrainLoss {metrics['train_loss']:.4f} | "
               f"ValLoss {metrics['val_loss']:.4f} | "
               f"ValAUPRC {metrics['val_auprc']:.4f} | "
               f"ValAUROC {metrics['val_auroc']:.4f} | "
               f"Time {metrics['epoch_time']:.1f}s")
        self.log(msg)
    
    def close(self):
        elapsed = time.time() - self.start_time
        self.log(f"Total training time: {elapsed:.1f}s")
        self.log_file.close()
```

### 4.4 tqdm 使用规范

```python
from tqdm import tqdm

# 交互式终端：使用 tqdm
if sys.stdout.isatty():
    pbar = tqdm(dataloader, desc="Training", ncols=80)
    for batch in pbar:
        loss = train_step(batch)
        pbar.set_postfix({'loss': f'{loss:.4f}'})
else:
    # 非交互式（nohup）：关闭 tqdm，使用纯文本
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(batch)
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.4f}", flush=True)
```

---

## 5. PyTorch 模型保存和加载最佳实践

### 5.1 推荐方法：保存 state_dict

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

**优点：**
- 灵活性高，可以跨不同模型架构
- 文件更小
- 更安全，不依赖类定义

### 5.2 保存完整训练状态

```python
# 保存
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_auprc': best_val_auprc,
    'train_loss': train_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
best_val_auprc = checkpoint['best_val_auprc']
```

### 5.3 保存最佳模型

```python
best_val_auprc = 0.0

for epoch in range(epochs):
    train_loss = train_epoch(...)
    val_loss, val_metrics = evaluate(...)
    
    # 保存最佳模型
    if val_metrics['auprc'] > best_val_auprc:
        best_val_auprc = val_metrics['auprc']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_auprc': best_val_auprc,
        }, 'best_model.pth')
        print(f"✓ 保存最佳模型 (AUPRC: {best_val_auprc:.4f})", flush=True)
```

### 5.4 避免的错误

```python
# ❌ 错误：保存整个模型
torch.save(model, 'model.pth')  # 不推荐

# ❌ 错误：忘记 model.eval()
model.load_state_dict(torch.load('model.pth'))
predictions = model(test_data)  # 错误！Dropout 和 BN 仍在训练模式

# ✅ 正确
model.load_state_dict(torch.load('model.pth'))
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

---

## 6. 训练过程中的验证集评估

### 6.1 标准评估函数

```python
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    在验证集/测试集上评估模型
    """
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        inputs = batch['x'].to(device)
        targets = batch['y'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        
        # 收集预测和真实标签
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())
    
    # 合并所有批次
    y_pred_proba = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    
    # 计算指标
    metrics = compute_binary_metrics(y_true, y_pred_proba)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, metrics
```

### 6.2 高效评估策略

```python
def train_with_validation(model, train_loader, val_loader, epochs, device):
    """
    训练时定期验证
    """
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 每个 epoch 都验证（推荐）
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 或者每 N 个 epoch 验证一次（如果验证很慢）
        if epoch % 5 == 0:
            val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 输出关键指标
        print(f"Epoch {epoch} | "
              f"TrainLoss {train_loss:.4f} | "
              f"ValAUPRC {val_metrics['auprc']:.4f} | "
              f"ValAUROC {val_metrics['auroc']:.4f}", 
              flush=True)
```

### 6.3 避免数据泄漏

```python
# ❌ 错误：在验证集上选择阈值，然后在验证集上评估
threshold, _ = find_optimal_threshold(val_y_true, val_y_pred_proba)
val_metrics = compute_binary_metrics(val_y_true, val_y_pred_proba, threshold)

# ✅ 正确：在验证集上选择阈值，在测试集上评估
threshold, _ = find_optimal_threshold(val_y_true, val_y_pred_proba)
test_metrics = compute_binary_metrics(test_y_true, test_y_pred_proba, threshold)
```

---

## 7. 完整训练脚本示例

```python
import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    log_path = 'logs/training.log'
    
    # 初始化日志
    logger = TrainingLogger(log_path)
    logger.log(f"开始训练 | Device: {device} | Epochs: {epochs}")
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data()
    
    # 初始化模型
    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auprc = 0.0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # 保存最佳模型
        if val_metrics['auprc'] > best_val_auprc:
            best_val_auprc = val_metrics['auprc']
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 记录日志
        logger.log_epoch(epoch, epochs, {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auprc': val_metrics['auprc'],
            'val_auroc': val_metrics['auroc'],
            'epoch_time': epoch_time
        })
        
        # 每 10 个 epoch 输出完整指标
        if epoch % 10 == 0:
            logger.log(f"  详细指标 | F1: {val_metrics['f1']:.4f} | "
                      f"Precision: {val_metrics['precision']:.4f} | "
                      f"Recall: {val_metrics['recall']:.4f}")
    
    # 测试集评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.log(f"\n=== 最终测试结果 ===")
    logger.log(f"Test AUPRC: {test_metrics['auprc']:.4f}")
    logger.log(f"Test AUROC: {test_metrics['auroc']:.4f}")
    logger.log(f"Test F1: {test_metrics['f1']:.4f}")
    
    logger.close()

if __name__ == '__main__':
    main()
```

---

## 8. 关键要点总结

### 8.1 指标计算
- ✅ 使用概率值，不是 logits
- ✅ 提前检查类别数量
- ✅ 使用 `zero_division=0` 避免警告
- ✅ 在验证集上选择阈值，在测试集上评估

### 8.2 日志输出
- ✅ 使用 `print(..., flush=True)`
- ✅ 文件使用行缓冲模式 `buffering=1`
- ✅ 每次写入后立即 `flush()`
- ✅ 使用 `python -u` 启动脚本

### 8.3 模型保存
- ✅ 保存 `state_dict` 而不是整个模型
- ✅ 保存完整训练状态以便恢复
- ✅ 加载后调用 `model.eval()`
- ✅ 推理时使用 `torch.no_grad()`

### 8.4 训练监控
- ✅ 每个 epoch 输出基础指标
- ✅ 每 N 个 epoch 输出完整指标
- ✅ 记录最佳验证性能
- ✅ 保存最佳模型

---

## 9. 常见错误清单

| 错误 | 正确做法 |
|------|---------|
| 使用 logits 计算 AUPRC | 先用 sigmoid 转换为概率 |
| 没有检查类别数量 | 提前检查 `len(np.unique(y_true)) < 2` |
| print 不加 flush | 使用 `print(..., flush=True)` |
| 文件写入不 flush | 使用行缓冲或手动 flush |
| 保存整个模型 | 保存 `state_dict` |
| 加载后忘记 eval | 调用 `model.eval()` |
| 在验证集上评估阈值 | 在验证集选择，测试集评估 |
| 没有保存最佳模型 | 跟踪最佳指标并保存 |

