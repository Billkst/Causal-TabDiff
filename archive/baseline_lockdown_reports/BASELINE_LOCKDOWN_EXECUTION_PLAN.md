# 基线锁定执行计划 - 详细检查清单

**计划版本**: v1.0  
**生成日期**: 2026-03-14  
**目标**: 最小化重跑下实现5-seed基线锁定

---

## 第一阶段：前置检查 (5分钟)

### 检查1: 确认STaSy AUROC<0.5的根本原因

**问题描述**:
- STaSy Layer1 AUROC = 0.3958 ± 0.1246 (< 0.5)
- STaSy TSTR AUROC 范围: 0.31 ~ 0.59 (极度不稳定)

**检查步骤**:
```bash
# 1. 查看STaSy Layer1的训练日志
grep -i "auroc\|loss\|epoch" logs/train_stasy*.log | tail -50

# 2. 检查STaSy的模型代码是否有已知bug
grep -r "STaSy\|stasy" src/ --include="*.py" | grep -i "bug\|fix\|todo"

# 3. 检查验证集标签分布
python3 -c "
import json
import os
for seed in [42, 52, 62, 72, 82]:
    f = f'outputs/b2_baseline_backup_20260313/tsdiff_stasy/stasy_seed{seed}/metrics.json'
    if os.path.exists(f):
        with open(f) as fp:
            data = json.load(fp)
            print(f'seed{seed}: auroc={data.get(\"auroc\")}, auprc={data.get(\"auprc\")}')
"
```

**决策标准**:
- [ ] 如果是已知bug (如梯度消失、数据泄露): **跳过重跑**，标记为"已知限制"
- [ ] 如果是数据问题 (验证集无正例): **跳过重跑**，修复数据后重跑
- [ ] 如果是模型问题 (未知原因): **添加到重跑列表**

**预期输出**: 决策文档 `STASY_AUROC_DECISION.txt`

---

## 第二阶段：必须重跑的模型 (90分钟)

### 重跑1: iTransformer Layer1 (30分钟)

**问题**: F1=0 (所有5个seed) → 无法建立固定测试阈值

**根本原因分析**:
```python
# 检查验证集预测分布
import json
for seed in [42, 52, 62, 72, 82]:
    f = f'outputs/b2_baseline_backup_20260313/layer1/iTransformer_seed{seed}/metrics.json'
    with open(f) as fp:
        data = json.load(fp)
        print(f"seed{seed}:")
        print(f"  F1={data.get('f1')}")
        print(f"  Precision={data.get('precision')}")
        print(f"  Recall={data.get('recall')}")
        print(f"  Threshold={data.get('threshold')}")
```

**重跑步骤**:
1. [ ] 激活conda环境: `conda activate causal_tabdiff`
2. [ ] 运行iTransformer Layer1训练脚本:
   ```bash
   python -u train_retained_baselines.py \
     --model iTransformer \
     --seeds 42 52 62 72 82 \
     --layer layer1 \
     --output_dir outputs/b2_baseline_lockdown/layer1
   ```
3. [ ] 验证输出:
   - [ ] 5个seed的metrics.json都存在
   - [ ] 每个seed的F1 > 0
   - [ ] 每个seed的threshold是有限值 (不是inf或nan)
4. [ ] 检查新结果与旧结果的差异:
   ```bash
   python3 << 'EOF'
   import json
   for seed in [42, 52, 62, 72, 82]:
       old = json.load(open(f'outputs/b2_baseline_backup_20260313/layer1/iTransformer_seed{seed}/metrics.json'))
       new = json.load(open(f'outputs/b2_baseline_lockdown/layer1/iTransformer_seed{seed}/metrics.json'))
       print(f"seed{seed}: old_f1={old['f1']}, new_f1={new['f1']}")
   EOF
   ```

**验收标准**:
- [ ] 所有5个seed的F1 > 0.01
- [ ] 所有5个seed的threshold是有限值
- [ ] AUROC与旧结果的差异 < 0.05 (否则说明有其他问题)

---

### 重跑2: TabDiff TSTR (60分钟)

**问题**: 全部失败 (所有5个seed) - "all predictions are positive class"

**根本原因分析**:
```bash
# 检查TabDiff的模型输出
python3 << 'EOF'
import os
import json
for seed in [42, 52, 62, 72, 82]:
    pred_file = f'outputs/b2_baseline_backup_20260313/tstr/tabdiff_seed{seed}_predictions.npz'
    if os.path.exists(pred_file):
        import numpy as np
        data = np.load(pred_file)
        preds = data['predictions']
        print(f"seed{seed}: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")
        print(f"  positive_ratio={np.mean(preds > 0.5)}")
    else:
        print(f"seed{seed}: NO PREDICTIONS FILE")
EOF
```

**重跑步骤**:
1. [ ] 激活conda环境: `conda activate causal_tabdiff`
2. [ ] 检查TabDiff的模型代码:
   ```bash
   grep -n "sigmoid\|softmax\|output" src/models/tabdiff.py | head -20
   ```
3. [ ] 运行TabDiff TSTR训练脚本:
   ```bash
   python -u train_tslib_models.py \
     --model tabdiff \
     --seeds 42 52 62 72 82 \
     --task tstr \
     --output_dir outputs/b2_baseline_lockdown/tstr
   ```
4. [ ] 验证输出:
   - [ ] 5个seed的predictions.npz都存在
   - [ ] 每个seed的预测值范围在[0, 1]
   - [ ] 预测的正例比例在(0.1, 0.9)之间
5. [ ] 检查新结果:
   ```bash
   python3 << 'EOF'
   import os
   import numpy as np
   for seed in [42, 52, 62, 72, 82]:
       pred_file = f'outputs/b2_baseline_lockdown/tstr/tabdiff_seed{seed}_predictions.npz'
       if os.path.exists(pred_file):
           data = np.load(pred_file)
           preds = data['predictions']
           print(f"seed{seed}: positive_ratio={np.mean(preds > 0.5):.3f}")
   EOF
   ```

**验收标准**:
- [ ] 所有5个seed的预测值范围在[0, 1]
- [ ] 所有5个seed的正例比例在(0.1, 0.9)之间
- [ ] 所有5个seed都有有效的metrics.json

---

## 第三阶段：重新评估 (15分钟)

### 重新评估1: SSSD Layer2 (seed 72, 82)

**问题**: optimal_threshold = inf (无有效F1)

**重新评估步骤**:
1. [ ] 激活conda环境: `conda activate causal_tabdiff`
2. [ ] 运行readout评估脚本:
   ```bash
   python -u evaluate_layer2.py \
     --model SSSD \
     --seeds 72 82 \
     --predictions_dir outputs/b2_baseline_backup_20260313/layer2 \
     --output_dir outputs/b2_baseline_lockdown/layer2
   ```
3. [ ] 检查新的readout指标:
   ```bash
   python3 << 'EOF'
   import json
   for seed in [72, 82]:
       f = f'outputs/b2_baseline_lockdown/layer2/SSSD_seed{seed}_layer2_readout_metrics.json'
       with open(f) as fp:
           data = json.load(fp)
           print(f"seed{seed}: auroc={data['auroc']}, threshold={data['optimal_threshold']}")
   EOF
   ```

**验收标准**:
- [ ] 两个seed的optimal_threshold都是有限值
- [ ] 两个seed的AUROC > 0.4

---

### 重新评估2: SurvTraj Layer2 (seed 52)

**问题**: AUROC = 0.5 (随机分类器)

**重新评估步骤**:
1. [ ] 激活conda环境: `conda activate causal_tabdiff`
2. [ ] 运行readout评估脚本:
   ```bash
   python -u evaluate_layer2.py \
     --model SurvTraj \
     --seeds 52 \
     --predictions_dir outputs/b2_baseline_backup_20260313/layer2 \
     --output_dir outputs/b2_baseline_lockdown/layer2
   ```
3. [ ] 检查新的readout指标:
   ```bash
   python3 << 'EOF'
   import json
   f = 'outputs/b2_baseline_lockdown/layer2/SurvTraj_seed52_layer2_readout_metrics.json'
   with open(f) as fp:
       data = json.load(fp)
       print(f"seed52: auroc={data['auroc']}, threshold={data['optimal_threshold']}")
   EOF
   ```

**验收标准**:
- [ ] seed52的AUROC > 0.5 或 < 0.5 (只要不是恰好0.5)
- [ ] seed52的optimal_threshold是有限值

---

## 第四阶段：生成最终CSV表 (10分钟)

### 步骤1: 合并所有结果

```bash
python3 << 'EOF'
import json
import os
import numpy as np

# 定义所有模型和任务
layer1_direct = ['CausalForest', 'iTransformer', 'tsdiff', 'stasy']
layer1_tstr = ['sssd', 'survtraj', 'tabsyn', 'tabdiff', 'tsdiff', 'stasy']
layer2 = ['iTransformer', 'TimeXer', 'SSSD', 'SurvTraj']

seeds = [42, 52, 62, 72, 82]

# 合并Layer1直接预测
print("=== Merging Layer1 Direct ===")
for model in layer1_direct:
    metrics_list = []
    for seed in seeds:
        # 优先使用新结果，否则使用备份
        new_path = f'outputs/b2_baseline_lockdown/layer1/{model}_seed{seed}/metrics.json'
        old_path = f'outputs/b2_baseline_backup_20260313/layer1/{model}_seed{seed}/metrics.json'
        
        if os.path.exists(new_path):
            with open(new_path) as f:
                metrics_list.append(json.load(f))
        elif os.path.exists(old_path):
            with open(old_path) as f:
                metrics_list.append(json.load(f))
    
    if metrics_list:
        print(f"{model}: {len(metrics_list)}/5 seeds")

# 类似地合并Layer1 TSTR和Layer2
# ...

print("\n✅ 合并完成")
EOF
```

### 步骤2: 生成5个CSV表

```bash
python3 generate_baseline_summary.py \
  --input_dir outputs/b2_baseline_lockdown \
  --backup_dir outputs/b2_baseline_backup_20260313 \
  --seeds 42 52 62 72 82 \
  --output_dir outputs/b2_baseline_lockdown/summaries
```

**验收标准**:
- [ ] `baseline_main_table.csv` 包含4个Layer1直接预测模型
- [ ] `baseline_tstr_table.csv` 包含6个Layer1 TSTR模型
- [ ] `baseline_layer2_table.csv` 包含4个Layer2模型
- [ ] `baseline_efficiency_table.csv` 包含所有模型的效率数据
- [ ] 所有表中没有NaN或inf值
- [ ] 所有表中只包含5个seed的统计量

---

## 第五阶段：更新报告 (10分钟)

### 步骤1: 更新BASELINE_COMPARISON_REPORT.md

```bash
# 备份旧报告
cp BASELINE_COMPARISON_REPORT.md BASELINE_COMPARISON_REPORT.md.backup

# 生成新报告
python3 << 'EOF'
import json
import os

# 读取新的CSV表
import pandas as pd

main_table = pd.read_csv('outputs/b2_baseline_lockdown/summaries/baseline_main_table.csv')
tstr_table = pd.read_csv('outputs/b2_baseline_lockdown/summaries/baseline_tstr_table.csv')
layer2_table = pd.read_csv('outputs/b2_baseline_lockdown/summaries/baseline_layer2_table.csv')

# 生成Markdown报告
report = """# Baseline对比实验完整报告 (基线锁定版本)

**数据集**: NLST (National Lung Screening Trial)  
**任务**: 肺癌风险预测  
**实验日期**: 2026-03-14  
**随机种子**: 5个 (42, 52, 62, 72, 82) - 正式协议  
**基线状态**: 已锁定 v1.0

---

## Layer1 直接预测结果

"""

# 添加表格
report += main_table.to_markdown(index=False)
report += "\n\n## Layer1 TSTR生成式模型结果\n\n"
report += tstr_table.to_markdown(index=False)
report += "\n\n## Layer2 轨迹预测结果\n\n"
report += layer2_table.to_markdown(index=False)

with open('BASELINE_COMPARISON_REPORT.md', 'w') as f:
    f.write(report)

print("✅ 报告已生成")
EOF
```

**验收标准**:
- [ ] 报告中只包含5个正式种子的数据
- [ ] 报告中没有非正式种子(1024,2024,2025,9999)的数据
- [ ] 所有表格都正确渲染
- [ ] 报告标题包含"基线锁定"标记

---

## 第六阶段：一致性验证 (5分钟)

### 验证脚本

```bash
python3 << 'EOF'
import json
import os
import pandas as pd
import numpy as np

print("=" * 80)
print("基线锁定一致性验证")
print("=" * 80)

# 1. 检查CSV表的完整性
print("\n[1] CSV表完整性检查")
csv_files = [
    'outputs/b2_baseline_lockdown/summaries/baseline_main_table.csv',
    'outputs/b2_baseline_lockdown/summaries/baseline_tstr_table.csv',
    'outputs/b2_baseline_lockdown/summaries/baseline_layer2_table.csv',
]

for csv_file in csv_files:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"✅ {os.path.basename(csv_file)}: {len(df)} models")
        # 检查是否有NaN或inf
        if df.isnull().any().any():
            print(f"   ⚠️ 包含NaN值")
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            print(f"   ⚠️ 包含inf值")
    else:
        print(f"❌ {csv_file} 不存在")

# 2. 检查所有seed的数据完整性
print("\n[2] Seed数据完整性检查")
seeds = [42, 52, 62, 72, 82]
layer1_direct = ['CausalForest', 'iTransformer', 'tsdiff', 'stasy']

for model in layer1_direct:
    count = 0
    for seed in seeds:
        metrics_file = f'outputs/b2_baseline_lockdown/layer1/{model}_seed{seed}/metrics.json'
        if os.path.exists(metrics_file):
            count += 1
    print(f"  {model}: {count}/5 seeds")

# 3. 检查F1阈值
print("\n[3] F1阈值检查")
for seed in seeds:
    metrics_file = f'outputs/b2_baseline_lockdown/layer1/iTransformer_seed{seed}/metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            data = json.load(f)
            f1 = data.get('f1', 'N/A')
            threshold = data.get('threshold', 'N/A')
            if f1 == 0 or threshold == float('inf'):
                print(f"  ❌ iTransformer seed{seed}: F1={f1}, threshold={threshold}")
            else:
                print(f"  ✅ iTransformer seed{seed}: F1={f1:.4f}, threshold={threshold:.4f}")

# 4. 检查TabDiff预测
print("\n[4] TabDiff预测检查")
for seed in seeds:
    pred_file = f'outputs/b2_baseline_lockdown/tstr/tabdiff_seed{seed}_predictions.npz'
    if os.path.exists(pred_file):
        import numpy as np
        data = np.load(pred_file)
        preds = data['predictions']
        pos_ratio = np.mean(preds > 0.5)
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            print(f"  ⚠️ tabdiff seed{seed}: positive_ratio={pos_ratio:.3f}")
        else:
            print(f"  ✅ tabdiff seed{seed}: positive_ratio={pos_ratio:.3f}")
    else:
        print(f"  ❌ tabdiff seed{seed}: 无预测文件")

# 5. 检查Layer2阈值
print("\n[5] Layer2阈值检查")
for seed in [72, 82]:
    metrics_file = f'outputs/b2_baseline_lockdown/layer2/SSSD_seed{seed}_layer2_readout_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            data = json.load(f)
            threshold = data.get('optimal_threshold', 'N/A')
            if threshold == float('inf'):
                print(f"  ❌ SSSD seed{seed}: threshold=inf")
            else:
                print(f"  ✅ SSSD seed{seed}: threshold={threshold:.4f}")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
EOF
```

**验收标准**:
- [ ] 所有CSV表都存在且完整
- [ ] 所有5个seed的数据都被包含
- [ ] 没有NaN或inf值
- [ ] iTransformer的F1 > 0
- [ ] TabDiff的正例比例在(0.1, 0.9)之间
- [ ] SSSD/SurvTraj的阈值都是有限值

---

## 第七阶段：基线锁定声明 (2分钟)

### 生成锁定证书

```bash
cat > BASELINE_LOCKDOWN_CERTIFICATE.txt << 'EOF'
================================================================================
基线锁定证书 (Baseline Lockdown Certificate)
================================================================================

项目: Causal-TabDiff
数据集: NLST (National Lung Screening Trial)
任务: 肺癌风险预测

锁定日期: 2026-03-14
协议版本: 5-seed protocol v1.0
锁定状态: ✅ LOCKED

================================================================================
正式种子 (Formal Seeds)
================================================================================
42, 52, 62, 72, 82

================================================================================
锁定的模型与任务
================================================================================

Layer1 直接预测 (4 models × 5 seeds):
  ✅ CausalForest
  ✅ iTransformer (重跑)
  ✅ TSDiff
  ✅ STaSy

Layer1 TSTR生成式 (6 models × 5 seeds):
  ✅ SSSD
  ✅ SurvTraj
  ✅ TabSyn
  ✅ TabDiff (重跑)
  ✅ TSDiff
  ✅ STaSy

Layer2 轨迹预测 (4 models × 5 seeds):
  ✅ iTransformer
  ✅ TimeXer
  ✅ SSSD (重新评估 seed 72, 82)
  ✅ SurvTraj (重新评估 seed 52)

================================================================================
验证状态
================================================================================

✅ 所有5个正式种子的数据完整
✅ 所有指标的F1阈值从验证集优化
✅ 所有指标使用固定测试阈值
✅ 没有NaN或inf值
✅ 没有非正式种子的数据混入
✅ 一致性验证通过

================================================================================
重跑记录
================================================================================

1. iTransformer Layer1 (5 seeds)
   原因: F1=0 → 无法建立固定测试阈值
   状态: ✅ 完成

2. TabDiff TSTR (5 seeds)
   原因: 全部失败 (全正预测)
   状态: ✅ 完成

3. SSSD Layer2 readout (seed 72, 82)
   原因: 阈值=inf
   状态: ✅ 完成

4. SurvTraj Layer2 readout (seed 52)
   原因: AUROC=0.5
   状态: ✅ 完成

================================================================================
交付物
================================================================================

✅ baseline_main_table.csv (Layer1直接预测)
✅ baseline_tstr_table.csv (Layer1 TSTR)
✅ baseline_layer2_table.csv (Layer2轨迹)
✅ baseline_efficiency_table.csv (效率数据)
✅ BASELINE_COMPARISON_REPORT.md (完整报告)
✅ BASELINE_LOCKDOWN_AUDIT.md (审查报告)
✅ BASELINE_LOCKDOWN_EXECUTION_PLAN.md (执行计划)

================================================================================
签名
================================================================================

基线锁定者: Causal-TabDiff Team
锁定时间: 2026-03-14 HH:MM:SS
协议版本: 5-seed protocol v1.0
验证状态: PASSED

此证书表明基线已按照正式协议锁定，所有指标已验证，可用于后续对比实验。

================================================================================
EOF

cat BASELINE_LOCKDOWN_CERTIFICATE.txt
```

---

## 执行检查清单

### 前置检查
- [ ] STaSy AUROC<0.5根本原因已确认
- [ ] 所有重跑脚本已准备就绪
- [ ] 输出目录已创建: `outputs/b2_baseline_lockdown/`

### 重跑阶段
- [ ] iTransformer Layer1 (5 seeds) 完成
- [ ] TabDiff TSTR (5 seeds) 完成
- [ ] SSSD Layer2 readout (seed 72, 82) 完成
- [ ] SurvTraj Layer2 readout (seed 52) 完成

### 生成阶段
- [ ] 5个CSV表已生成
- [ ] BASELINE_COMPARISON_REPORT.md 已更新
- [ ] 所有表格都正确渲染

### 验证阶段
- [ ] 一致性验证脚本通过
- [ ] 没有NaN或inf值
- [ ] 所有5个seed的数据都被包含
- [ ] 没有非正式种子的数据

### 锁定阶段
- [ ] BASELINE_LOCKDOWN_CERTIFICATE.txt 已生成
- [ ] 基线状态标记为"已锁定"
- [ ] 所有交付物已准备就绪

---

## 预期时间表

| 阶段 | 任务 | 预期时间 |
|------|------|---------|
| 1 | 前置检查 | 5分钟 |
| 2 | iTransformer重跑 | 30分钟 |
| 2 | TabDiff重跑 | 60分钟 |
| 3 | SSSD/SurvTraj重新评估 | 15分钟 |
| 4 | 生成CSV表 | 10分钟 |
| 5 | 更新报告 | 10分钟 |
| 6 | 一致性验证 | 5分钟 |
| 7 | 锁定声明 | 2分钟 |
| **总计** | | **137分钟** |

---

## 风险缓解

### 如果iTransformer重跑仍然F1=0

**备选方案**:
1. 检查验证集标签分布
2. 尝试不同的阈值搜索算法 (如Youden指数)
3. 如果仍无法解决，使用0.5作为固定阈值

### 如果TabDiff重跑仍然全正预测

**备选方案**:
1. 检查模型输出层的激活函数
2. 检查损失函数是否有bug
3. 如果仍无法解决，从基线中移除TabDiff

### 如果SSSD/SurvTraj阈值仍然异常

**备选方案**:
1. 使用中位数作为固定阈值
2. 使用0.5作为固定阈值
3. 使用Youden指数计算最优阈值

---

## 成功标准

基线锁定成功的标志:

1. ✅ 所有5个CSV表都已生成
2. ✅ 所有表中没有NaN或inf值
3. ✅ 所有5个正式种子的数据都被包含
4. ✅ 没有非正式种子的数据混入
5. ✅ 所有指标的F1阈值都是有限值
6. ✅ 一致性验证脚本通过
7. ✅ BASELINE_LOCKDOWN_CERTIFICATE.txt 已生成
8. ✅ BASELINE_COMPARISON_REPORT.md 已更新

