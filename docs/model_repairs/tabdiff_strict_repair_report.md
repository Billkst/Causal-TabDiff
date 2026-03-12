# TabDiff_strict 修复报告

## 问题分析

### V7 错误信息
```
mixed_loss() takes 2 args but 3 given
```

### 根本原因
wrapper 中对 `mixed_loss` 的调用不匹配真实签名。

## 真实源码签名

### 1. UniModMLP.__init__
```python
def __init__(
    self, d_numerical, categories, num_layers, d_token,
    n_head=1, factor=4, bias=True, dim_t=512, use_mlp=True, **kwargs
)
```
- `categories` 必须是 numpy array，不能是 list
- 返回 `(x_num_pred, x_cat_pred)` 元组

### 2. UnifiedCtimeDiffusion.__init__
```python
def __init__(
    self,
    num_classes: np.array,
    num_numerical_features: int,
    denoise_fn,
    y_only_model,
    num_timesteps=1000,
    scheduler='power_mean',
    cat_scheduler='log_linear',
    noise_dist='uniform',
    edm_params={},
    noise_dist_params={},
    noise_schedule_params={},
    sampler_params={},
    device=torch.device('cpu'),
    **kwargs
)
```

### 3. mixed_loss 签名
```python
def mixed_loss(self, x):
    # 返回 (d_loss, c_loss) 元组
    return d_loss.mean(), c_loss.mean()
```
- **只接收 1 个参数 `x`**（不是 2 个）
- 返回 `(d_loss, c_loss)` 元组

## 修复清单

### 文件：`src/baselines/tabdiff_landmark_strict.py`

#### 修改 1：第 31-39 行 - UniModMLP 初始化
```python
# 修改前
denoise_fn = UniModMLP(
    d_numerical=self.total_dim,
    categories=[],  # ❌ 错误：list
    num_layers=4,
    d_token=256
).to(device)

# 修改后
denoise_fn_base = UniModMLP(
    d_numerical=self.total_dim,
    categories=np.array([]),  # ✅ 正确：numpy array
    num_layers=4,
    d_token=256
).to(device)
```

#### 修改 2：第 42-53 行 - UnifiedCtimeDiffusion 初始化
```python
# 添加必要参数
self.model = UnifiedCtimeDiffusion(
    num_classes=np.array([]),
    num_numerical_features=self.total_dim,
    denoise_fn=denoise_fn,
    y_only_model=None,  # ✅ 必须是 None 或模型，不能是 False
    num_timesteps=1000,
    scheduler='power_mean',
    cat_scheduler='log_linear',
    noise_dist='uniform_t',  # ✅ 支持的分布
    edm_params={'sigma_data': 0.5},  # ✅ 必须提供
    sampler_params={'stochastic_sampler': False, 'second_order_correction': False},  # ✅ 必须提供
    device=device
).to(device)
```

#### 修改 3：第 65-68 行 - mixed_loss 调用
```python
# 修改前
loss = self.model.mixed_loss(xy, None)  # ❌ 错误：传了 2 个参数

# 修改后
d_loss, c_loss = self.model.mixed_loss(xy)  # ✅ 正确：只传 1 个参数，返回元组
loss = d_loss + c_loss
```

#### 修改 4：第 78 行 - sample 调用
```python
# 修改前
samples = self.model.sample(n_samples, device)  # ❌ 错误：传了 device

# 修改后
samples = self.model.sample(n_samples)  # ✅ 正确：只传 num_samples
```

## Smoke 测试结果

### 测试项目
- ✅ Wrapper 创建成功
- ✅ 数据加载成功
- ✅ 1 epoch 训练成功
- ✅ 10 samples 采样成功

### 测试输出
```
Device: cuda
✓ Wrapper created
✓ Data created
✓ Fit success (1 epoch)
✓ Sample success (10 samples)
Result: PASS
```

## 关键发现

1. **mixed_loss 只接收 1 个参数**，不是 2 个
2. **categories 必须是 numpy array**，不能是 list
3. **y_only_model 必须是 None 或模型**，不能是布尔值
4. **必须提供 edm_params 和 sampler_params**
5. **noise_dist='uniform_t' 是支持的分布**

## 修复状态

✅ **PASS** - TabDiff_strict 已成功修复并通过 smoke 测试
