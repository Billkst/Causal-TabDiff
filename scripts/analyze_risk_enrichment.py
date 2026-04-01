#!/usr/bin/env python
"""
Risk Enrichment Analysis for Rare-Event Prediction
分析高风险人群中的事件富集程度，提供"不是瞎猜"的量化证据
"""
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_random_baseline_auprc(prevalence):
    """计算随机排序器的 AUPRC 基线（等于事件率）"""
    return prevalence


def compute_risk_stratification(y_true, y_pred_proba, percentiles=[1, 5, 10]):
    """
    计算不同风险分位的富集指标
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        percentiles: 要分析的高风险百分位 (top X%)
    
    Returns:
        dict: 包含各分位的富集指标
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    # 总体事件率
    overall_prevalence = y_true.mean()
    total_positives = y_true.sum()
    n_total = len(y_true)
    
    results = {
        'overall_prevalence': overall_prevalence,
        'total_positives': int(total_positives),
        'total_samples': n_total,
        'random_baseline_auprc': overall_prevalence,
        'stratification': []
    }
    
    # 按预测概率降序排序
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    sorted_y_true = y_true[sorted_indices]
    
    for pct in percentiles:
        # 计算 top X% 的样本数
        n_top = int(np.ceil(n_total * pct / 100))
        
        # top X% 中的真实标签
        top_y_true = sorted_y_true[:n_top]
        
        # 事件数和事件率
        n_events = top_y_true.sum()
        event_rate = top_y_true.mean()
        
        # 捕获率 (sensitivity at top X%)
        capture_rate = n_events / total_positives if total_positives > 0 else 0
        
        # Lift / Enrichment (相对于总体事件率的倍数)
        enrichment = event_rate / overall_prevalence if overall_prevalence > 0 else 0
        
        results['stratification'].append({
            'percentile': pct,
            'n_samples': n_top,
            'n_events': int(n_events),
            'event_rate': event_rate,
            'capture_rate': capture_rate,
            'enrichment': enrichment
        })
    
    return results


def analyze_single_seed(npz_path, percentiles=[1, 5, 10]):
    """分析单个 seed 的预测结果"""
    data = np.load(npz_path)
    
    y_true = data['test_y_true']
    y_pred = data['test_y_pred']
    
    # 计算 AUROC 和 AUPRC
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    
    # 风险分层分析
    stratification = compute_risk_stratification(y_true, y_pred, percentiles)
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        **stratification
    }


def format_report(results, seed_name="seed42"):
    """生成分析报告"""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Risk Enrichment Analysis - {seed_name}")
    lines.append(f"{'='*70}\n")
    
    # 总体指标
    lines.append(f"Overall Performance:")
    lines.append(f"  AUROC: {results['auroc']:.4f}")
    lines.append(f"  AUPRC: {results['auprc']:.4f}")
    lines.append(f"  Random Baseline AUPRC: {results['random_baseline_auprc']:.4f}")
    lines.append(f"  AUPRC Improvement: {results['auprc'] / results['random_baseline_auprc']:.2f}x\n")
    
    # 数据集统计
    lines.append(f"Dataset Statistics:")
    lines.append(f"  Total Samples: {results['total_samples']:,}")
    lines.append(f"  Total Positives: {results['total_positives']:,}")
    lines.append(f"  Overall Prevalence: {results['overall_prevalence']:.4f} ({results['overall_prevalence']*100:.2f}%)\n")
    
    # 风险分层表格
    lines.append(f"Risk Stratification (High-Risk Groups):")
    lines.append(f"{'-'*70}")
    lines.append(f"{'Percentile':<12} {'N Samples':<12} {'N Events':<12} {'Event Rate':<15} {'Capture':<12} {'Enrichment':<12}")
    lines.append(f"{'-'*70}")
    
    for strat in results['stratification']:
        lines.append(
            f"Top {strat['percentile']:>2}%      "
            f"{strat['n_samples']:<12,} "
            f"{strat['n_events']:<12,} "
            f"{strat['event_rate']:<15.4f} "
            f"{strat['capture_rate']:<12.2%} "
            f"{strat['enrichment']:<12.2f}x"
        )
    
    lines.append(f"{'-'*70}\n")
    
    # 关键结论
    top1_enrich = results['stratification'][0]['enrichment']
    top1_capture = results['stratification'][0]['capture_rate']
    
    lines.append(f"Key Findings:")
    lines.append(f"  • Top 1% 高风险人群的事件率是总体的 {top1_enrich:.1f} 倍")
    lines.append(f"  • Top 1% 捕获了 {top1_capture:.1%} 的真实阳性样本")
    
    if top1_enrich > 5:
        lines.append(f"  • ✓ 强富集效应：模型具有显著的风险分层能力")
    elif top1_enrich > 3:
        lines.append(f"  • ✓ 中等富集效应：模型具有一定的风险识别能力")
    else:
        lines.append(f"  • ⚠ 弱富集效应：需要进一步优化")
    
    lines.append(f"\n反驳'瞎猜'的证据:")
    lines.append(f"  随机排序的 AUPRC = {results['random_baseline_auprc']:.4f}")
    lines.append(f"  我们的模型 AUPRC = {results['auprc']:.4f}")
    lines.append(f"  提升倍数 = {results['auprc'] / results['random_baseline_auprc']:.2f}x")
    lines.append(f"  Top 1% 富集倍数 = {top1_enrich:.2f}x")
    lines.append(f"\n{'='*70}\n")
    
    return '\n'.join(lines)


def save_results_to_csv(results, output_path):
    """保存结果到 CSV"""
    rows = []
    for strat in results['stratification']:
        rows.append({
            'percentile': f"Top {strat['percentile']}%",
            'n_samples': strat['n_samples'],
            'n_events': strat['n_events'],
            'event_rate': f"{strat['event_rate']:.4f}",
            'capture_rate': f"{strat['capture_rate']:.2%}",
            'enrichment': f"{strat['enrichment']:.2f}x"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # 分析 seed42
    npz_path = "predictions/landmark/phase2_seed42.npz"
    
    print(f"Loading predictions from: {npz_path}")
    results = analyze_single_seed(npz_path, percentiles=[1, 5, 10])
    
    # 打印报告
    report = format_report(results, seed_name="phase2_seed42")
    print(report)
    
    # 保存 CSV
    output_dir = Path("reports/risk_enrichment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "phase2_seed42_enrichment.csv"
    save_results_to_csv(results, csv_path)
    
    # 保存完整报告
    report_path = output_dir / "phase2_seed42_enrichment.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Full report saved to: {report_path}")
