---
trigger: manual
---

## 元信息
- 版本: v5.0 (Antigravity-Conda-Memory-Pipeline)
- 标签: [L2-Coding, Research-SOP, Dry-Run, Conda-SubShell, Memory-Ledger]

## 目的
规范化“记忆读取 -> 本地规划 -> 编码测试 -> 记忆写入 -> 服务器部署”的闭环工作流。彻底解决跨会话的上下文断层与环境变量丢失问题。

## 输入/输出
- 必需: [用户的 L3 指令]
- 隐含输入: 项目根目录下的 `history.json` (记录历史开发进度)
- 预期输出格式: 
  - 阶段 0: 静默读取 `history.json`。
  - 阶段 1: Implementation Plan。
  - 阶段 2: 业务代码 + 自动更新 `requirements.txt`。
  - 阶段 3: 基于 `conda run -n causal_tabdiff` 的本地冒烟测试。
  - 阶段 4: 写入 `history.json` 并交付 `experiment_commands.md`。

## 提示词正文
在接收到开发/实验需求后，必须严格按以下五个阶段串行推进：

### Phase 0: 记忆同步 (Memory Sync)
- 强制静默读取项目根目录的 `history.json`（若无则跳过），理解当前项目的最新进展和待办事项。

### Phase 1: 规划解析 (Plan)
- 在 Planning Mode 下输出任务执行计划。
- 明确列出接下来要修改的代码架构。
- **环境锁定**：所有后续终端操作默认锁定在本地 Conda 环境 `causal_tabdiff` 中。
- 等待用户 Approve。

### Phase 2: 编码实现与依赖更新 (Code & Env)
- 编写引入 `logging(DEBUG)` 的代码，预留/使用 `--debug_mode` 接口（开启时强制截断数据集、极小 Epoch 并使用 CPU）。
- 提取新增的第三方库，增量更新 `requirements.txt`。

### Phase 3: 本地受控冒烟测试 (Local Conda Dry-Run)
- **子 Shell 安全调用准则**：在本地 Terminal 执行任何测试命令时，必须使用 `conda run -n causal_tabdiff <command>` 来包裹命令。
- 执行本地安装与极小样本测试，监控终端 DEBUG 日志，确保数据流、Loss 计算无逻辑 Bug。

### Phase 4: 记忆固化与交付 (Memorize & Deliver)
- 测试通过后，**必须向 `history.json` 追加一条记录**，写明本次完成的任务模块和关键验证结果（如 Baseline 跑通）。
- 仅当需要服务器跑全量数据时，更新或生成 `experiment_commands.md` (包含 screen 和 tee 指令)。

## 注意事项
- **算力隔离红线**：Phase 3 仅为逻辑验证，绝不允许引发本地机器算力过载。
- **导师隔离协议**：若涉及“复现开源原论文性能”的需求，仅在 history.json 记录 TODO，绝不在当前管道中下载外部公开数据集。