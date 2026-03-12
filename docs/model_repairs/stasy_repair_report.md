# STaSy 修复报告

## 模型信息
- **模型名称**: STaSy
- **修复状态**: ✓ PASS
- **修复文件**: `src/baselines/stasy_landmark_wrapper.py`

## V5 原始错误
```
AttributeError: 'activation'
```

## 根因分析
**blocker_type**: `wrapper_config_bug`

STaSy 的 `ncsnpp_tabular` 模型需要完整的配置字段，但 wrapper 中缺失多个必需字段：

### 缺失的 config.model 字段
- `conditional` (必需)
- `embedding_type` (必需)
- `fourier_scale` (当 embedding_type='fourier' 时必需)
- `layer_type` (必需)
- `scale_by_sigma` (必需)

### 缺失的 config.optim 字段
- `beta1` (Adam optimizer 必需)
- `eps` (Adam optimizer 必需)
- `warmup` (optimization_manager 必需)
- `grad_clip` (optimization_manager 必需)

### 缺失的 config.sampling 字段
- `method`, `predictor`, `corrector`
- `snr`, `n_steps_each`
- `noise_removal`, `probability_flow`

### 其他问题
- `get_step_fn` 默认 `spl=True` 但未提供 `alpha0`, `beta0`

## 修复内容

### 1. 补全 config.model 字段
```python
config.model.conditional = True
config.model.embedding_type = "fourier"
config.model.fourier_scale = 16.0
config.model.layer_type = "concatsquash"
config.model.scale_by_sigma = True
```

### 2. 补全 config.optim 字段
```python
config.optim.beta1 = 0.9
config.optim.eps = 1e-8
config.optim.warmup = 0
config.optim.grad_clip = 1.0
```

### 3. 补全 config.sampling 字段
```python
config.sampling.method = 'pc'
config.sampling.predictor = 'euler_maruyama'
config.sampling.corrector = 'none'
config.sampling.snr = 0.16
config.sampling.n_steps_each = 1
config.sampling.noise_removal = True
config.sampling.probability_flow = False
```

### 4. 禁用 spl
```python
train_step_fn = stasy_losses.get_step_fn(..., spl=False)
```

### 5. 保存 config 供 sampling 使用
```python
self.config = config
# 在 sample() 中使用 self.config
sampling_fn = sampling.get_sampling_fn(self.config, self.sde, ...)
```

## 验证结果

### Smoke Test 通过项
- ✓ create/load success
- ✓ fit/train success (1 epoch, 360 samples)
- ✓ sample success (10 samples, shape=(10,3,15))
- ✓ prediction export ready
- ✓ evaluate_model ready

### 最终状态
**PASS** - 所有检查通过

## Layer 支持
- **Layer1**: PASS (TSTR 生成式 baseline)
- **Layer2**: N/A (不支持)

## 备注
- STaSy 使用 `torch.nn.DataParallel`，需要 CUDA 环境
- 配置字段必须完整，否则会在不同阶段失败
- 本次修复基于 STaSy core 源码的真实接口需求
