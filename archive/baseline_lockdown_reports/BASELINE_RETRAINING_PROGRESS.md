# Baseline重新训练进度报告

**生成时间**: 2026-03-13 17:31 UTC

## 已完成任务 ✅

### 1. 代码修复
- ✅ 修复 `src/baselines/tstr_pipeline.py` 的 XGBoost cupy 依赖问题（移除 `device='cpu'` 参数）
- ✅ 修复 `train_tstr_pipeline.py` 的日志规范（添加实时输出、行缓冲、DualLogger）
- ✅ Smoke test 验证：TSDiff 和 STaSy 均通过（seed=42）

### 2. Layer2 指标计算
- ✅ iTransformer Layer2：5个seeds指标已计算（MSE: 1069.74±43.31, MAE: 19.27±1.73）
- ✅ SSSD Layer2：指标文件已存在
- ✅ SurvTraj Layer2：指标文件已存在

### 3. 预测文件验证
- ✅ CausalForest：5个seeds预测文件完整（outputs/retained_baselines_b2/）

## 进行中任务 🔄

### TSDiff TSTR 全量训练（5 seeds）
- ✅ seed 42: 完成
- 🔄 seed 1024: 训练中
- 🔄 seed 2024: 训练中
- 🔄 seed 2025: 训练中
- 🔄 seed 9999: 训练中

### STaSy TSTR 全量训练（5 seeds）
- ✅ seed 42: 完成
- 🔄 seed 1024: 训练中
- 🔄 seed 2024: 训练中
- 🔄 seed 2025: 训练中
- 🔄 seed 9999: 训练中

**后台进程数**: 18个训练进程正在运行

## 待完成任务 ⏳

1. 等待 TSDiff/STaSy 训练完成（预计需要数小时）
2. 计算 TSDiff/STaSy 的评估指标（TSTR协议）
3. 汇总所有baseline结果，生成对比实验报告（mean±std）
4. 验证所有模型的结果完整性

## 关键修复说明

### XGBoost cupy 问题
**原因**: XGBoost的 `device='cpu'` 参数在某些版本中需要cupy依赖  
**修复**: 移除该参数，使用默认设备选择  
**验证**: Smoke test通过，训练正常运行

### 日志规范改进
**改进内容**:
- 添加 DualLogger 类（同时输出到终端和日志文件）
- 使用行缓冲模式（`buffering=1`）确保实时写入
- 添加总时间统计
- 所有输出使用 `flush=True`

**日志位置**: `logs/train_tstr_{model}_seed{seed}.log`

## 监控命令

```bash
# 检查训练进度
ps aux | grep train_tstr_pipeline | wc -l

# 实时查看日志
tail -f logs/tstr_tsdiff_seed1024_full.log

# 检查完成情况
ls -lh outputs/tstr_baselines/*.npz
```
