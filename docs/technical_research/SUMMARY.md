# PyTorch 训练与评估最佳实践 - 核心要点总结

## 📋 快速检查清单

### ✅ 指标计算（必须遵守）
```python
# ❌ 错误
auprc = average_precision_score(y_true, logits)

# ✅ 正确
proba = torch.sigmoid(logits).cpu().numpy()
if len(np.unique(y_true)) >= 2:
    auprc = average_precision_score(y_true, proba)
else:
    auprc = np.nan
```

### ✅ 实时日志输出（必须遵守）
```python
# 1. 打开日志文件时使用行缓冲
log_file = open(log_path, 'w', buffering=1)

# 2. 终端输出必须 flush
print(f"Epoch {epoch} | Loss {loss:.4f}", flush=True)

# 3. 文件写入后立即 flush
log_file.write(msg + '\n')
log_file.flush()

# 4. 启动脚本时使用 -u 参数
# python -u train.py
```

### ✅ 模型保存与加载（必须遵守）
```python
# ❌ 错误：保存整个模型
torch.save(model, 'model.pth')

# ✅ 正确：保存 state_dict
torch.save(model.state_dict(), 'model.pth')

# ✅ 加载时
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 必须调用！
```

### ✅ 训练循环标准结构
```python
for epoch in range(epochs):
    # 1. 训练阶段
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # 2. 验证阶段
    model.eval()
    with torch.no_grad():
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
    
    # 3. 保存最佳模型
    if val_metrics['auprc'] > best_auprc:
        best_auprc = val_metrics['auprc']
        torch.save(model.state_dict(), 'best_model.pth')
    
    # 4. 实时输出
    msg = f"Epoch {epoch}/{epochs} | TrainLoss {train_loss:.4f} | ValAUPRC {val_metrics['auprc']:.4f}"
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()
```

---

## 🎯 六大核心原则

### 1️⃣ 指标计算三要素
- **概率转换**：logits → sigmoid → numpy
- **类别检查**：`len(np.unique(y_true)) >= 2`
- **零除保护**：`zero_division=0`

### 2️⃣ 实时输出三保证
- **终端可见**：`print(..., flush=True)`
- **文件实时**：`buffering=1` + `flush()`
- **启动参数**：`python -u script.py`

### 3️⃣ 模型保存三原则
- **只存字典**：`state_dict` 而非整个模型
- **存完整状态**：包含 optimizer、epoch、best_metric
- **加载后 eval**：`model.eval()` + `torch.no_grad()`

### 4️⃣ 验证评估三步骤
- **切换模式**：`model.eval()`
- **禁用梯度**：`with torch.no_grad()`
- **计算指标**：AUROC、AUPRC、F1 等

### 5️⃣ 日志输出三层次
- **每个 epoch**：基础信息（loss、lr、time）
- **每 N epoch**：完整指标（AUPRC、AUROC、F1）
- **训练结束**：最终性能 + 最佳模型路径

### 6️⃣ 阈值选择三原则
- **验证集选择**：在 val set 上搜索最优阈值
- **测试集评估**：用选定阈值在 test set 上评估
- **不可泄露**：绝不在 test set 上选择阈值

---

## 🚨 十大常见错误

| # | 错误 | 后果 | 正确做法 |
|---|------|------|---------|
| 1 | 用 logits 计算 AUPRC | 指标完全错误 | 先 sigmoid 转概率 |
| 2 | 不检查类别数量 | 程序崩溃 | `if len(np.unique(y_true)) >= 2` |
| 3 | print 不加 flush | 日志延迟/丢失 | `print(..., flush=True)` |
| 4 | 文件不用行缓冲 | tail -f 看不到 | `open(..., buffering=1)` |
| 5 | 保存整个模型 | 跨版本不兼容 | 只保存 `state_dict` |
| 6 | 加载后不 eval | 推理结果错误 | `model.eval()` |
| 7 | 验证时不禁用梯度 | 内存溢出 | `with torch.no_grad()` |
| 8 | 在测试集选阈值 | 数据泄露 | 只在验证集选 |
| 9 | 不保存最佳模型 | 无法复现 | 跟踪 best_metric |
| 10 | 只输出 loss | 无法监控性能 | 定期输出 AUPRC/AUROC |

---

## 📊 标准训练日志格式

### 推荐格式（每个 epoch）
```
Epoch 12/100 | Seed 42 | LR 1.0e-3 | TrainLoss 0.1834 | ValLoss 0.2017 | ValAUPRC 0.0450 | BestAUPRC 0.0523 | Time 8.2s
```

### 完整指标输出（每 10 epoch）
```
Epoch 20/100 | Seed 42 | LR 1.0e-3
  Train: Loss=0.1652
  Val:   Loss=0.1889 | AUPRC=0.0512 | AUROC=0.6234 | F1=0.1234 | Precision=0.0987 | Recall=0.1567
  Best:  AUPRC=0.0523 (epoch 18)
  Time:  8.5s
```

---

## 🔧 实用代码模板

### 完整训练脚本模板
```python
import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score, average_precision_score

def train_model(model, train_loader, val_loader, epochs, device, log_path, seed):
    # 初始化
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    log_file = open(log_path, 'w', buffering=1)
    
    best_val_auprc = 0.0
    best_epoch = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        y_true_list, y_pred_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                proba = torch.sigmoid(logits).cpu().numpy()
                y_true_list.append(y.cpu().numpy())
                y_pred_list.append(proba)
        
        val_loss /= len(val_loader)
        y_true = np.concatenate(y_true_list).flatten()
        y_pred = np.concatenate(y_pred_list).flatten()
        
        # 计算指标
        if len(np.unique(y_true)) >= 2:
            val_auprc = average_precision_score(y_true, y_pred)
            val_auroc = roc_auc_score(y_true, y_pred)
        else:
            val_auprc = np.nan
            val_auroc = np.nan
        
        # 保存最佳模型
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_epoch = epoch
            torch.save(model.state_dict(), f'best_model_seed{seed}.pth')
        
        # 实时输出
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        msg = (f"Epoch {epoch}/{epochs} | Seed {seed} | LR {lr:.1e} | "
               f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
               f"ValAUPRC {val_auprc:.4f} | ValAUROC {val_auroc:.4f} | "
               f"BestAUPRC {best_val_auprc:.4f} | Time {epoch_time:.1f}s")
        
        print(msg, flush=True)
        log_file.write(msg + '\n')
        log_file.flush()
    
    log_file.close()
    print(f"\n训练完成 | 最佳 AUPRC: {best_val_auprc:.4f} (Epoch {best_epoch})", flush=True)
    return model
```

---

## 📚 参考资源

### 官方文档
- PyTorch 训练教程: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- sklearn.metrics API: https://scikit-learn.org/stable/modules/model_evaluation.html

### 关键函数文档
- `average_precision_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- `roc_auc_score`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- `torch.save`: https://pytorch.org/docs/stable/generated/torch.save.html

---

## ✅ 验收标准

一个训练脚本只有满足以下条件才算合格：

- [ ] 能在终端实时看到训练进展
- [ ] 能通过 `tail -f logs/xxx.log` 实时看到日志更新
- [ ] 每个 epoch 至少有基础状态输出
- [ ] 每隔固定轮数有核心指标输出（AUPRC、AUROC）
- [ ] 使用概率值计算指标，不是 logits
- [ ] 提前检查类别数量，避免崩溃
- [ ] 保存 state_dict 而非整个模型
- [ ] 加载后调用 model.eval()
- [ ] 验证时使用 torch.no_grad()
- [ ] 跟踪并保存最佳模型
- [ ] 日志中能够追溯：当前配置、seed、进度、最佳性能

---

**文档版本**: v1.0  
**创建日期**: 2026-03-14  
**适用项目**: Causal-TabDiff  
**维护者**: 项目团队
