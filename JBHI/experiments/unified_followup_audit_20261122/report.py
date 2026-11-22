"""Render the public aggregate report. No case details are read by the renderer."""
import re
from common import HERE,LOGS,PRIVATE,STATS,ARMS,NOTICE,obj
import numpy as np
import pandas as pd

NAMES={'R':'冻结 R（λ=1，γ=2）','aux025':'λ=0.25','aux05':'λ=0.5','aux2':'λ=2',
       'gamma0':'γ=0','gamma1':'γ=1','gamma3':'γ=3'}


def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
                     *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def seconds(label):
    p=LOGS/(label+'-time.txt')
    return float(re.search(r'elapsed_seconds=([.0-9]+)',p.read_text())[1])


def main():
    r=pd.read_csv(STATS/'param_sensitivity_metrics.csv',float_precision='round_trip')
    tau=pd.read_csv(STATS/'tau_sweep.csv').query("kind=='mean'")
    comp=pd.read_csv(STATS/'complexity_benchmark.csv')
    paired=obj(STATS/'param_sensitivity_paired.json')
    checks=obj(STATS/'verification.json')
    counts=obj(STATS/'case_selection.json')
    agg=counts['dev0_seed42']
    m=r.groupby('arm').mean(numeric_only=True)
    dev2=r.query('fold==2 and seed==42').set_index('arm')
    env=obj(PRIVATE/'benchmark_env.json')
    cpu={v['field']:v['data'] for v in __import__('json').loads(env['lscpu'])['lscpu']}
    text=[]
    def paragraph(t):
        text.append(t)
    def block(title,headers,data):
        text.extend([title,NOTICE+'。',markdown(headers,data)])
    paragraph('# Causal-TabDiff 统一随访协议 · 论文写作补齐审计报告')
    paragraph('**'+NOTICE+'。** 两年肺癌排序和下一轮 CT 尺寸预测分别评价。分析限于既有 10,594 名开发患者的 31,587 个预测时点。每折内部保留拟合、校准与外折评估角色，折间按患者划分。三折与随机种子 42、52、62 形成九个等权单元。')
    p={c['arm']:c for c in paired['contrasts']}
    crossing=sum(c['lo']<=0<=c['hi'] for c in p.values())
    paragraph(f'冻结 R 的平均精确率为 {m.loc["R","ap"]:.5f}。六项单因素变更中，{crossing} 项相对 R 的差值区间包含零。阈值从 0.50 降至 0.10 时，条件非零敏感度由 {tau.iloc[-1].nonzero_sensitivity:.1%} 升至 {tau.iloc[1].nonzero_sensitivity:.1%}，条件零特异度由 {tau.iloc[-1].zero_specificity:.1%} 降至 {tau.iloc[1].zero_specificity:.1%}。全肺最大径平均绝对误差保持 {tau.iloc[-1].whole_mae:.3f} mm。')
    block('## 稀疏门的解码阈值',['τ','条件零特异度','条件非零敏感度','全肺误差 mm','部位误差 mm','生长误差 mm','最大径违例¹','生长违例¹'],
          [[f'{a.tau:.2f}',f'{a.zero_specificity:.2%}',f'{a.nonzero_sensitivity:.2%}',f'{a.whole_mae:.4f}',f'{a.site_mae:.4f}',f'{a.growth_mae:.4f}',int(a.max_violation),int(a.growth_violation)] for a in tau.itertuples()])
    paragraph('¹ 各档各单元两种违例数都是零。条件指标给定真实主导部位，在真实阳性测量报告的非主导部位与生长标记上计算。误差则使用模型自行预测的主导类别，覆盖所有合格 CT。阈值只改变非主导部位和生长径是否输出正值，全肺径由主导类别和正径回归决定。降低阈值可找回部分额外正值，同时使更多真实零值出现非零预测。[54 个单元与六行均值](statistics/tau_sweep.csv)')
    rows=[]
    for k in ARMS:
        x=m.loc[k];a=paired['absolute_ap'][k]
        rows.append([NAMES[k],f'{x.ap:.5f} [{a["lo"]:.5f}, {a["hi"]:.5f}]',
                     '参考' if k=='R' else f'{p[k]["delta"]:+.5f} [{p[k]["lo"]:+.5f}, {p[k]["hi"]:+.5f}]',
                     f'{x.auc:.4f}',f'{x.f1:.4f}',f'{x.mcc:.4f}',int(dev2.loc[k,'selected_epoch'])])
    block('## 辅助损失权重与焦点损失指数',['设置','平均精确率（95% CI）','相对 R 的差（95% CI）','AUROC','F1','MCC','开发折 2 / 种子 42 选中轮次'],rows)
    paragraph('平均精确率（AUPRC）沿用原协议的 average precision。AUROC 为受试者工作特征曲线下面积，F1 为精确率与召回率的调和平均，MCC 为马修斯相关系数。原双重加权、随机抽样、随机流与损失不确定性加权保持一致。λ 行固定 γ=2，γ 行固定 λ=1。每次微调从本折本种子的有效历史前缀预训练权重初始化。校准 AUPRC 严格上升时才更新检查点，同分保留较早轮次，连续 30 轮无改善停止，200 轮封顶。')
    block('### 极早期检查点的单元复核',['设置','选中轮次','实际停止轮次','本单元外折 AUPRC','同折另外两个种子的平均 AUPRC'],
          [[NAMES[k],int(dev2.loc[k,'selected_epoch']),min(200,int(dev2.loc[k,'selected_epoch'])+30),
            f'{dev2.loc[k,"ap"]:.5f}',f'{r[(r.arm==k)&(r.fold==2)&(r.seed!=42)].ap.mean():.5f}'] for k in ARMS])
    early=[NAMES[k] for k in ARMS if int(dev2.loc[k,'selected_epoch'])<=3]
    paragraph('在开发折 2、种子 42，冻结 R 在第 3 轮取得最佳校准值，第 33 轮停止。以原异常的第 3 轮为描述性参照，选中轮次不晚于第 3 轮的设置包括'+'、'.join(early)+'。单元间差异应与九单元配对结果合读。')
    devgrid=r.pivot(index=['fold','seed'],columns='arm',values='ap')
    diff=devgrid['aux05']-devgrid['R']
    paragraph(f'辅助权重 λ=0.5 的九单元差值为 {p["aux05"]["delta"]:+.5f}，逐项 95% 区间 [{p["aux05"]["lo"]:+.5f}, {p["aux05"]["hi"]:+.5f}]。其中开发折 2、种子 42 的差值为 {diff.loc[(2,42)]:+.5f}，其余八个单元的平均差为 {diff.drop(index=(2,42)).mean():+.5f}。这项拆分描述检查点敏感性，正式估计始终包含全部九个单元。')
    rows=[]
    for k in ARMS:
        x=m.loc[k]
        rows.append([NAMES[k],f'{x.coverage1:.2%} / {x.enrichment1:.2f}×',f'{x.coverage5:.2%} / {x.enrichment5:.2f}×',
                    f'{x.zero_specificity:.2%}',f'{x.nonzero_sensitivity:.2%}',
                    f'{x.whole_mae:.3f} / {x.site_mae:.3f} / {x.growth_mae:.3f}',f'{x.max_violation:g} / {x.growth_violation:g}'])
    block('### 高分层覆盖与测量质量',['设置','前 1% 覆盖 / 富集','前 5% 覆盖 / 富集','条件零特异度','条件非零敏感度','全肺 / 部位 / 生长 MAE，mm','最大径 / 生长违例'],rows)
    paragraph('高分层覆盖率是该层捕获的阳性时点占全部阳性时点的比例。富集倍数以评估折阳性率为基准，边界同分按份额计入。分类阈值由校准集最大 F1 选出。两侧 95% 置信区间采用原协议的 2,000 次患者级分层配对重抽样。每折按患者是否有阳性时点分层，同一患者的全部时点在一次抽样中共享权重，各模型及三个种子共享该折抽样。每次先汇总九单元，再取百分位数。这些是逐项区间，适用于当前已拟合模型。[63 组完整指标及选中轮次](statistics/param_sensitivity_metrics.csv) · [配对差值与区间](statistics/param_sensitivity_paired.json)')
    block('## 平衡病例的筛选与群体背景',['病例主题','确定性首层条件与排序','去除已选患者后的合格时点'],
          [['机制正向例','T0/T1，合格下一轮 CT，两年阳性。B 违例，R 满足约束且全肺误差更小，R 风险前 10%。按误差改善降序',list(counts['counts'].values())[0]],
           ['障碍与终点分化例','T0/T1，R 全肺误差小于 N 但漏报真实额外部位或生长标记。优先两年阳性且 R 分位低于 N，再按分位差降序',list(counts['counts'].values())[1]],
           ['纯历史部署边界例','T2 两年阳性，R、N 均未进前 5%。按两者分位差的绝对值降序',list(counts['counts'].values())[2]]])
    paragraph(f'三个主题均在首选开发折 0、种子 42 命中，无需放宽条件。三位患者互异。在该单元的 {agg["eligible"]:,} 个合格时点，B 有 {agg["b_max_violations"]} 个全肺与部位最大径违例，R 有 {agg["r_max_violations"]} 个。真实阳性测量报告含 {agg["extra_positive_sites"]} 个非主导阳性部位或生长标记，常规自主解码在 τ=0.50 漏报 {agg["missed_extra_sites"]} 个，降至 0.10 后找回其中 {agg["recovered_extra_sites"]} 个。{agg["positive_t2"]} 个两年阳性 T2 时点中，{agg["both_outside_top5"]} 个同时落在 R 与 N 的前 5% 之外。这表明当前 14 维短历史和基线吸烟条件未能把这些时点排入高分层。分位排名表达相对顺序。仅限本地的三张详表见受限目录，年龄按 5 岁区间分组。[去重后的命中数和全折背景](statistics/case_selection.json)')
    paragraph('病例按固定折和种子选择。同层比较分数相同则取冻结评估行序中较早者。风险分位使用折内同分中秩。下轮 CT 尺寸仅作事后对照，T2 的诊断头输出仅作结构展示。病例 2 在预先固定的 τ=0.10 同批重放，全体额外项的漏报统计使用自由解码，与首表的给定主导类别统计分母不同。')
    block('## 计算图、拟合成本与单线程推理',['模型','登记网络参数','训练图参数¹','部署调用参数','单次拟合中位秒²','其中微调或基线拟合秒','1,000 时点中位 ms [Q1, Q3]'],
          [[a.model,f'{a.registered_parameters:,}',f'{a.training_graph_parameters:,}',f'{a.deployment_parameters:,}',f'{a.fit_seconds_median:.2f}',f'{a.fine_seconds_median:.2f}',f'{a.median_ms:.2f} [{a.q25_ms:.2f}, {a.q75_ms:.2f}]'] for a in comp.itertuples()])
    paragraph(f'硬件为 {cpu["Model name:"]}，{cpu["Core(s) per socket:"]} 个物理核、{cpu["CPU(s):"]} 个逻辑处理器。PyTorch {env["torch"]}。待微调结束后，基准进程固定到逻辑核 {env["pinned_logical_cpu"]}，算子与并行任务线程数均为 1。所有模型使用开发折 0、种子 42 的冻结权重与同一批 1,000 个已编码的评估时点，单次整批前向，无拆批。每模型预热 5 次，独立计时 20 次，使用单调高精度时钟。Q1、Q3 为耗时的第 25 和第 75 百分位。训练耗时来自既有归档，延迟为本次实测。')
    paragraph('¹ 登记网络参数为模型注册参数总数。训练图统计训练损失反向可达的网络参数和实际参与损失的标量权重，A/B/C/R 为 3 个，D/N 为 2 个。部署列统计运行原始风险路径实际读取的网络参数，包含已执行但输出被舍弃的子计算。因此个别模型的部署调用数可高于反向可达数。CSV 另列训练前向读取数。A/B/C/D/R/N 的日常路径只编码已知历史、池化并调用风险头，六者均调用 5,133 个参数。B/C/D/R/N 的 CT 报告头用于辅助监督和诊断展示。TSDiff 耗时包含其原实现对输入的梯度引导。')
    paragraph('部署伪代码对应冻结 R 的临床风险路径。令 T 为历史轮次数、d 为每轮输入维度，两个注意力轴计算分数矩阵与加权求和的主项分别为 O(T²d) 和 O(d²T)，另有线性投影、前馈层与输出头的开销。本实验 T≤3，d=14。')
    paragraph("""```text
输入：截至当前轮的历史 X，已知轮次掩码 M，基线吸烟条件 a
H ← 在 M 标记的轮次上执行条件时间轴与特征轴注意力
h ← 对 H 的已知轮次取平均
风险排序分数 ← sigmoid(风险头(h))
需要测量分析时：以 h 预测主导类别与全肺径，按 τ 解码额外部位及生长径
```""")
    paragraph('² 每个模型的九次正式拟合分别计时取中位数。扩散模型按每单元的预训练时间加微调时间计算从头独立拟合的成本。共享预训练只需实际执行一次。A/B/C/D 复用原统一协议的 complete.json 与 pretrain.log，R/N 复用修复归档的 budgets.csv 和 pretrain.csv。五个深度基线复用各自 complete.json 的完整拟合耗时，含原定 45 轮及校准候选评估。基准计时排除文件读取、数据预处理、参数追踪和结果比对。[11 模型参数量与延迟](statistics/complexity_benchmark.csv)')
    paragraph('## 实现假设与验收')
    paragraph('按作者确认，涉及患者的所有运算在本地开发工作站执行。SSH 仅用于连通与硬件核查。三个案例的详表仅写入本地 private/case_studies_local_only.md，不纳入审查分支。预留患者未进入输入清单。零值判定沿用归档定义，解剖违例容差为 10⁻⁵ mm。')
    block('### 可回读验收记录',['核查项','结果'],[
        ['冻结阈值 τ=0.50，全测量列最大绝对误差',checks['tau_max_abs']],
        ['冻结 R 九组引用与独立复算相对归档风险表的最大绝对误差',checks['r_metrics_max_abs']],
        ['冻结预训练复用 / 新微调 / 最佳权重预测重放','9 / 54 / 63'],
        ['筛选病例 / 基准模型 / 配对差值','3 / 11 / 6'],
        ['本次 54 项微调墙钟耗时',f'{seconds("fits"):.2f} 秒'],
        ['统计配对 / 串行延迟基准墙钟耗时',f'{seconds("pairs"):.2f} / {seconds("benchmark"):.2f} 秒']])
    paragraph('在具备受限开发注册表、冻结检查点及依赖的工作区，将本包放在两个冻结实验的同级目录，设置 CTD_PYTHON 后运行 reproduce_audit.sh。首次运行完成全部微调，已有完整拟合记录时复用相同设置的输出。五种深度比较模型依赖原统一协议的适配器与对应上游代码，其中 SurvTraj 使用登记版本 f5dbc2593091341b0f78bee098972d83e5f521f6。独立审查分支只含源程序与聚合证据。[复现脚本](reproduce_audit.sh) · [数值检查](statistics/verification.json) · [冻结依赖与既有复现说明](https://github.com/Billkst/Causal-TabDiff/tree/160fe3f7c0fee74e10a634e7ed5ed0668d0fe858)')
    (HERE/'论文写作补齐审计报告.md').write_text('\n\n'.join(text)+'\n')


if __name__=='__main__':
    main()
