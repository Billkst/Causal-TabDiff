# Causal-TabDiff 部署与执行命令

本文档包含了在训练服务器上执行最终部署和全量训练的脚本指令。

## 1. 环境初始化
请确保服务器上已经激活了 `causal_tabdiff` Conda 环境。如果尚未创建，请执行以下命令：

```bash
conda create -n causal_tabdiff python=3.9 -y
conda activate causal_tabdiff
```

## 2. 依赖安装
安装 `requirements.txt` 中指定的所有依赖库：

```bash
pip install -r requirements.txt
```

## 3. Baseline Evaluation 测试运行
为了测评我们构建的 Causal-TabDiff，需要优先执行 Baseline 测试基准。这会利用多随机数种子（默认 5 个）自动计算各类算法的 `mean ± std`。

> **注意**：此处包含 `Causal Forest`，以及通过预留 Wrapper 封装的其余深度学习 SOTA 基准。在主线自研模型跑通之前，这里只含有开源基线网络的评测。

```bash
# 创建一个名为 "baseline_eval" 的新 screen 会话
screen -S baseline_eval

conda activate causal_tabdiff
mkdir -p logs/evaluation

# 启动包含 5 个 baselines 与目前自身代码评测框架的全量执行：
python run_baselines.py | tee logs/evaluation/baselines_full.log

# （要将 screen 会话挂起并退回主终端，请按：Ctrl+A，然后按 D）
```

## 4. 全量数据 / 服务器训练执行 (主线自研模型)
使用 `screen` 启动完整的由我们研究团队提出的 Causal-TabDiff 核心模型的训练和评估。

> **注意**：在服务器上进行真实训练时，**绝对不能**添加 `--debug_mode` 参数。该参数是为了在本地笔记本等弱算力设备上强制截断数据集并使用 CPU 进行冒烟测试而设计的。

```bash
# 创建一个名为 "causal_run" 的新 screen 会话
screen -S causal_run

# 确保在 screen 会话中激活环境
conda activate causal_tabdiff

# 确保日志存储目录存在
mkdir -p logs/training

# 执行实验脚本，并将输出日志保存到 logs/training/training_run.log 中
python run_experiment.py | tee logs/training/training_run.log

# （要将 screen 会话挂起并退回主终端，请按：Ctrl+A，然后按 D）
# （如果后续需要重新连接查看训练进度日志，请执行：screen -r causal_run）
```
