---
trigger: always_on
---

## 元信息
- 标签: [L2-Memory, State-Persistence, Changelog]

## 长期记忆流转规范 (Memory Ledger)
为了保证跨会话的上下文连贯性，你必须强制维护项目根目录下的 `history.json` 文件。

### 1. 唤醒读取 (Read on Start)
- 当用户开启一个新任务时，你必须首先静默读取 `history.json` 的最后 5 条记录，了解当前项目的最新进展和被搁置的 Bug。

### 2. 完工写入 (Append on Complete)
- 每当你完成一个代码模块的编写、跑通一次 Baseline 实验，或解决了一个严重 Bug 后，**必须**自动向 `history.json` 追加一条记录。
- **写入规则**：仅追加 (Append-only)，绝对禁止覆盖或删除历史数据。
- **数据结构**：
  ```json
  {
    "timestamp": "2026-02-23Txx:xx:xx",
    "id": "简短的任务标识码",
    "type": "feature | bugfix | experiment | refactor",
    "user_intent": "用户的原始核心诉求",
    "details": "具体做了什么修改，跑出了什么结果 (如 baseline 的 mean±std)",
    "file_path": "主要修改或生成的文件路径"
  }