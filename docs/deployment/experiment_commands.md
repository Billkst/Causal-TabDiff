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

## 3. 全量数据 / 服务器训练执行
使用 `screen` 启动完整的模型训练和评估，以防 SSH 会话断开导致训练终止。同时，我们使用 `tee` 命令将训练时的标准输出同步保存到日志文件中。

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
