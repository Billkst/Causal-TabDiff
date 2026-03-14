# 日志工具故障排查指南

## 常见问题

### 1. 日志文件未创建

**症状**：运行脚本后 `logs/` 目录为空

**原因**：
- logger 未调用 `close()`
- 脚本异常退出

**解决**：
```python
# 使用 context manager
with TrainingLogger('logs', 'my_script') as logger:
    logger.log("Training started")
    # ... 训练代码 ...
# 自动调用 close()

# 或显式调用
logger = TrainingLogger('logs', 'my_script')
try:
    # ... 训练代码 ...
finally:
    logger.close()
```

---

### 2. 日志文件内容为空

**症状**：日志文件存在但无内容

**原因**：
- 脚本未调用 `logger.log()` 或 `logger.epoch_log()`
- 缓冲未 flush

**解决**：
```python
# 确保调用了日志方法
logger.epoch_log(epoch=1, total_epochs=100, seed=42, lr=1e-3, train_loss=0.5)

# 确保调用了 close()
logger.close()
```

---

### 3. 日志输出不实时

**症状**：`tail -f logs/xxx.log` 无法实时看到更新

**原因**：
- 未使用行缓冲模式
- 未调用 `flush()`

**解决**：
TrainingLogger 已内置行缓冲和 flush，无需额外操作。

---

### 4. 导入错误

**症状**：`ImportError: cannot import name 'TrainingLogger'`

**原因**：
- sys.path 未包含项目根目录
- utils 包未正确初始化

**解决**：
```python
import sys
sys.path.insert(0, '/path/to/Causal-TabDiff')
from src.utils import TrainingLogger
```

---

### 5. 日志格式错误

**症状**：日志输出格式不符合预期

**原因**：
- 参数类型错误
- 参数值为 None

**解决**：
```python
# 确保参数类型正确
logger.epoch_log(
    epoch=1,           # int
    total_epochs=100,  # int
    seed=42,           # int
    lr=1e-3,           # float
    train_loss=0.5,    # float
    val_loss=0.6       # float (可选)
)
```

---

## 验证清单

- [ ] TrainingLogger 可导入
- [ ] 日志文件已创建
- [ ] 日志文件有内容
- [ ] 日志格式正确
- [ ] `tail -f` 可实时查看
- [ ] 脚本异常退出时日志仍保存

---

## 测试脚本

```python
import sys
sys.path.insert(0, '/home/UserData/ljx/Project_2/Causal-TabDiff')
from src.utils import TrainingLogger

# 测试基本功能
logger = TrainingLogger('logs', 'test_logger')
logger.log("Test message")
logger.epoch_log(
    epoch=1,
    total_epochs=10,
    seed=42,
    lr=1e-3,
    train_loss=0.5,
    val_loss=0.6,
    best_val_metric=0.6,
    epoch_time=8.2
)
logger.close()

print("✓ 测试完成，检查 logs/test_logger.log")
```

