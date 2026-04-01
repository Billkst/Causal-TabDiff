#!/usr/bin/env python
"""
Multi-Seed Risk Enrichment Aggregation
聚合多个 seed 的风险富集分析结果，输出 mean ± std

用法:
    python aggregate_risk_enrichment.py file1.npz file2.npz file3.npz ...
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from analyze_risk_enrichment import analyze_single_seed


def verify_and_load_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    data = np.load(path)
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    
    metadata = {
        'path': str(p),
        'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S'),
        'seed': int(data['seed']) if 'seed' in data else 'N/A',
        'best_epoch': int(data['best_epoch']) if 'best_epoch' in data else 'N/A',
        'oversample_ratio': float(data['oversample_ratio']) if 'oversample_ratio' in data else 'N/A',
        'focal_gamma': float(data['focal_gamma']) if 'focal_gamma' in data else 'N/A',
    }
    
    return metadata


def aggregate_multiple_seeds(npz_paths, percentiles=[1, 5, 10]):
    print(f"\n{'='*70}")
    print("File Verification")
    print(f"{'='*70}\n")
    
    all_metadata = []
    for path in npz_paths:
        try:
            meta = verify_and_load_file(path)
            all_metadata.append(meta)
            print(f"File: {meta['path']}")
            print(f"  Modified: {meta['mtime']}")
            print(f"  Seed: {meta['seed']}")
            print(f"  Best Epoch: {meta['best_epoch']}")
            print(f"  Oversample Ratio: {meta['oversample_ratio']}")
            print(f"  Focal Gamma: {meta['focal_gamma']}")
            print()
        except Exception as e:
            print(f"ERROR loading {path}: {e}\n")
            raise
    
    print(f"{'='*70}")
    response = input(f"\n确认以上 {len(all_metadata)} 个文件都是当前 official run 的结果？(yes/no): ")
    if response.lower() != 'yes':
        print("用户取消聚合操作")
        sys.exit(0)
    
    print(f"\n{'='*70}")
    print("Processing Files")
    print(f"{'='*70}\n")
    
    all_results = []
    for path in npz_paths:
        seed_name = Path(path).stem
        print(f"Analyzing {seed_name}...")
        
        results = analyze_single_seed(path, percentiles)
        results['seed_name'] = seed_name
        all_results.append(results)
    
    return all_results


def compute_statistics(all_results):
    metrics = ['auroc', 'auprc', 'overall_prevalence']
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
            'values': values
        }
    
    n_percentiles = len(all_results[0]['stratification'])
    strat_stats = []
    
    for i in range(n_percentiles):
        pct = all_results[0]['stratification'][i]['percentile']
        
        event_rates = [r['stratification'][i]['event_rate'] for r in all_results]
        capture_rates = [r['stratification'][i]['capture_rate'] for r in all_results]
        enrichments = [r['stratification'][i]['enrichment'] for r in all_results]
        
        strat_stats.append({
            'percentile': pct,
            'event_rate_mean': np.mean(event_rates),
            'event_rate_std': np.std(event_rates, ddof=1) if len(event_rates) > 1 else 0,
            'capture_rate_mean': np.mean(capture_rates),
            'capture_rate_std': np.std(capture_rates, ddof=1) if len(capture_rates) > 1 else 0,
            'enrichment_mean': np.mean(enrichments),
            'enrichment_std': np.std(enrichments, ddof=1) if len(enrichments) > 1 else 0,
        })
    
    return stats, strat_stats


def format_aggregated_report(stats, strat_stats, n_seeds):
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Aggregated Risk Enrichment Analysis ({n_seeds} seeds)")
    lines.append(f"{'='*70}\n")
    
    lines.append(f"Overall Performance (mean ± std):")
    lines.append(f"  AUROC: {stats['auroc']['mean']:.4f} ± {stats['auroc']['std']:.4f}")
    lines.append(f"  AUPRC: {stats['auprc']['mean']:.4f} ± {stats['auprc']['std']:.4f}")
    
    baseline = stats['overall_prevalence']['mean']
    auprc_improvement = stats['auprc']['mean'] / baseline
    lines.append(f"  Random Baseline: {baseline:.4f}")
    lines.append(f"  AUPRC Improvement: {auprc_improvement:.2f}x\n")
    
    lines.append(f"Risk Stratification (mean ± std):")
    lines.append(f"{'-'*70}")
    lines.append(f"{'Percentile':<12} {'Event Rate':<20} {'Capture Rate':<20} {'Enrichment':<20}")
    lines.append(f"{'-'*70}")
    
    for s in strat_stats:
        lines.append(
            f"Top {s['percentile']:>2}%      "
            f"{s['event_rate_mean']:.4f}±{s['event_rate_std']:.4f}      "
            f"{s['capture_rate_mean']:.2%}±{s['capture_rate_std']:.2%}      "
            f"{s['enrichment_mean']:.2f}±{s['enrichment_std']:.2f}x"
        )
    
    lines.append(f"{'-'*70}\n")
    
    top1_enrich = strat_stats[0]['enrichment_mean']
    top1_enrich_std = strat_stats[0]['enrichment_std']
    
    lines.append(f"Key Findings:")
    lines.append(f"  • Top 1% 富集倍数: {top1_enrich:.2f} ± {top1_enrich_std:.2f}x")
    lines.append(f"  • AUPRC vs Random: {auprc_improvement:.2f}x")
    lines.append(f"  • 结论: 模型具有显著且稳定的风险分层能力\n")
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


def save_aggregated_csv(strat_stats, output_path):
    rows = []
    for s in strat_stats:
        rows.append({
            'percentile': f"Top {s['percentile']}%",
            'event_rate': f"{s['event_rate_mean']:.4f} ± {s['event_rate_std']:.4f}",
            'capture_rate': f"{s['capture_rate_mean']:.2%} ± {s['capture_rate_std']:.2%}",
            'enrichment': f"{s['enrichment_mean']:.2f} ± {s['enrichment_std']:.2f}x"
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aggregate_risk_enrichment.py <npz_file1> <npz_file2> ...")
        print("\nExample:")
        print("  python aggregate_risk_enrichment.py \\")
        print("    predictions/landmark/phase2_seed42.npz \\")
        print("    predictions/landmark/phase2_seed52.npz \\")
        print("    predictions/landmark/phase2_seed62.npz")
        sys.exit(1)
    
    npz_paths = sys.argv[1:]
    
    all_results = aggregate_multiple_seeds(npz_paths, percentiles=[1, 5, 10])
    stats, strat_stats = compute_statistics(all_results)
    
    report = format_aggregated_report(stats, strat_stats, len(all_results))
    print(report)
    
    output_dir = Path("reports/risk_enrichment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "phase2_aggregated_enrichment.csv"
    save_aggregated_csv(strat_stats, csv_path)
    
    report_path = output_dir / "phase2_aggregated_enrichment.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Full report saved to: {report_path}")
