---
trigger: manual
---

## 元信息
- 版本: v4.1 (Antigravity-Conda-Agent-Pipeline)
- 标签: [L2-Coding, Research-SOP, Dry-Run, Conda-SubShell-Safe]

## 目的
规范化“本地规划 -> 编码导出 -> Conda沙盒测试 -> 服务器部署”工作流。彻底解决 Agent 新开终端导致的环境变量丢失问题。

## 输入/输出
- 必需: [开题文档/数据集]、[用户的 L3 指令 (必须包含 conda 环境名)]
- 预期输出格式: 
  - 阶段一: Implementation Plan (必须包含环境名确认)。
  - 阶段二: 业务代码 + `requirements.txt`。
  - 阶段三: 基于 `conda run` 的本地冒烟测试报告。
  - 阶段四: `experiment_commands.md` (服务器专用部署脚本)。

## 提示词正文
在接收到实验需求后，严格按以下四个阶段串行推进：

### Phase 1: 规划解析 (Plan)
- 阅读本地文件。
- **获取环境上下文**：从用户的指令中提取他刚刚在本地创建的 Conda 环境名称。
- 在 Implementation Plan 中，明确列出代码架构和接下来的测试环境（如：“将在本地 Conda 环境 `causal_tabdiff` 中进行测试”）。
- 等待用户 Approve。

### Phase 2: 编码实现与依赖导出 (Code & Env)
- 编写引入 `logging(DEBUG)` 的实验代码，预留 `--debug_mode` 接口（开启时强制截断数据集并使用 CPU）。
- 提取依赖并生成 `requirements.txt`。

### Phase 3: 本地受控冒烟测试 (Local Conda Dry-Run)
- **子 Shell 安全调用准则**：在本地 Terminal 执行任何命令时，**绝对禁止**直接使用 `pip` 或 `python`。
- 你必须使用 `conda run -n <环境名>` 来包裹你的命令，以确保命令在正确的沙盒中执行。
- 执行依赖安装：
  `conda run -n <环境名> pip install -r requirements.txt`
- 执行受控测试：
  `conda run -n <环境名> python run_experiment.py --debug_mode`
- 监控终端输出的 DEBUG 日志，确保 Loss 正常计算、数据管道畅通。

### Phase 4: 交付服务器启动指令 (Deliver)
- 确认测试通过后，生成 `experiment_commands.md`。
- 包含服务器上的环境初始化指令、依赖安装指令，以及带有 `screen` 和 `tee` 的全量跑批指令。

## 注意事项
- **算力隔离红线**：Phase 3 仅为逻辑验证，绝不允许引发本地机器算力过载。