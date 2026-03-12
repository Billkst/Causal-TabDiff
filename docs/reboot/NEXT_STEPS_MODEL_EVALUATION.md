# 下一步行动：模型评估与选型

## 执行时间
2026-03-11

## 当前状态
已完成2022-2024年表格生成模型的初步调研，识别出3个高优先级候选模型。

---

## 立即行动项（按优先级排序）

### 🔴 优先级1: 验证TSDiff可行性（预计1-2天）

**目标**: 确认TSDiff是否能直接适配"条件输入→轨迹输出"任务

**具体步骤**:
1. 克隆TSDiff仓库到本地
   ```bash
   cd /tmp
   git clone https://github.com/amazon-science/unconditional-time-series-diffusion
   cd unconditional-time-series-diffusion
   ```

2. 阅读关键文件
   - `README.md` - 了解基本用法
   - `requirements.txt` - 检查依赖
   - `src/models/` - 理解模型架构
   - `src/data/` - 理解数据格式

3. 回答关键问题
   - [ ] 是否支持条件输入？
   - [ ] 是否支持多变量时间序列？
   - [ ] 是否支持变长序列？
   - [ ] 训练接口是否清晰？
   - [ ] 是否有预训练模型？

4. 运行示例代码
   ```bash
   # 在小数据集上测试
   python examples/train.py --dataset toy
   ```

5. 输出评估报告
   - 文件: `docs/reboot/TSDIFF_FEASIBILITY_REPORT.md`
   - 内容: 可行性、改造难度、预期时间

---

### 🟡 优先级2: 评估TabSyn架构（预计1天）

**目标**: 理解TabSyn的扩散机制，评估扩展到时序的难度

**具体步骤**:
1. 克隆TabSyn仓库
   ```bash
   cd /tmp
   git clone https://github.com/amazon-science/tabsyn
   cd tabsyn
   ```

2. 阅读核心代码
   - `models/diffusion.py` - 扩散过程实现
   - `models/encoder.py` - 特征编码
   - `train.py` - 训练流程

3. 回答关键问题
   - [ ] 扩散过程是否容易扩展到3D张量？
   - [ ] 是否有时序注意力机制？
   - [ ] 噪声调度是否可配置？
   - [ ] 代码模块化程度如何？

4. 输出评估报告
   - 文件: `docs/reboot/TABSYN_TEMPORAL_EXTENSION_ANALYSIS.md`

---

### 🟢 优先级3: 搜索最新论文（预计半天）

**目标**: 确保没有遗漏2024年的最新模型

**搜索关键词**:
1. `longitudinal tabular generation 2024`
2. `time-varying tabular synthesis 2024`
3. `conditional trajectory generation medical`
4. `tabular diffusion temporal 2024`

**搜索平台**:
- arXiv (https://arxiv.org)
- Papers with Code (https://paperswithcode.com)
- Google Scholar

**输出**:
- 文件: `docs/reboot/LATEST_PAPERS_2024.md`
- 内容: 论文列表、是否有开源代码、是否值得尝试

---

## 决策点（1周后）

完成上述3个优先级任务后，召开技术选型会议，决定：

### 决策1: 选择主要技术路线
- [ ] **路线A**: 直接使用TSDiff（如果可行性高）
- [ ] **路线B**: 改造TabSyn支持时序（如果TSDiff不适配）
- [ ] **路线C**: 改造STaSy支持时序（如果前两者都有问题）

### 决策2: 制定实施计划
- 预期时间线
- 里程碑设置
- 风险缓解措施

---

## 成功标准

### TSDiff验证成功标准
- ✅ 能在toy数据集上运行
- ✅ 支持条件输入
- ✅ 支持多变量时间序列
- ✅ 代码清晰易懂
- ✅ 改造工作量<2周

### TabSyn评估成功标准
- ✅ 扩散机制清晰
- ✅ 代码模块化良好
- ✅ 有明确的时序扩展路径
- ✅ 改造工作量<4周

---

## 风险与应对

### 风险1: TSDiff不支持条件输入
**应对**: 立即转向TabSyn或STaSy

### 风险2: 所有模型都需要大幅改造
**应对**: 考虑从头实现简化版时序扩散模型

### 风险3: 训练时间过长
**应对**: 使用更小的模型或更少的扩散步数

---

## 资源需求

### 计算资源
- GPU: 至少1张V100或A100
- 内存: 至少32GB
- 存储: 至少100GB（用于实验日志）

### 时间资源
- 模型验证: 1-2天
- 代码改造: 1-4周
- 实验调试: 1-2周
- 总计: 2-6周

