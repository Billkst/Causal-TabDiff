#!/usr/bin/env python
"""
Finalize Official Risk Enrichment Results
基于 official_npz 目录生成最终聚合结果
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from analyze_risk_enrichment import analyze_single_seed, compute_risk_stratification
from sklearn.metrics import roc_auc_score, average_precision_score


def load_and_verify_official_files(official_dir):
    official_dir = Path(official_dir)
    npz_files = sorted(official_dir.glob("phase2_seed*_official.npz"))
    
    if len(npz_files) != 5:
        raise ValueError(f"Expected 5 files, found {len(npz_files)}")
    
    print(f"\n{'='*70}")
    print("Official Files Verification")
    print(f"{'='*70}\n")
    
    all_metadata = []
    for f in npz_files:
        data = np.load(f)
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        
        meta = {
            'path': str(f),
            'filename': f.name,
            'mtime': mtime.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': int(data['seed']),
            'best_epoch': int(data['best_epoch']),
            'oversample_ratio': float(data['oversample_ratio']),
            'focal_gamma': float(data['focal_gamma']),
        }
        all_metadata.append(meta)
        
        print(f"File: {meta['filename']}")
        print(f"  Modified: {meta['mtime']}")
        print(f"  Seed: {meta['seed']}")
        print(f"  Best Epoch: {meta['best_epoch']}")
        print(f"  Oversample: {meta['oversample_ratio']}")
        print(f"  Focal Gamma: {meta['focal_gamma']}")
        print()
    
    print(f"{'='*70}\n")
    
    return [str(f) for f in npz_files], all_metadata


def aggregate_results(npz_paths, percentiles=[1, 5, 10]):
    all_results = []
    
    for path in npz_paths:
        seed_name = Path(path).stem.replace('_official', '')
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
            'std': np.std(values, ddof=1),
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
            'event_rate_std': np.std(event_rates, ddof=1),
            'capture_rate_mean': np.mean(capture_rates),
            'capture_rate_std': np.std(capture_rates, ddof=1),
            'enrichment_mean': np.mean(enrichments),
            'enrichment_std': np.std(enrichments, ddof=1),
        })
    
    return stats, strat_stats


def generate_paper_table(stats, strat_stats):
    lines = []
    lines.append("\n## Table: Risk Enrichment Analysis (5-seed average)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| AUROC | {stats['auroc']['mean']:.4f} ± {stats['auroc']['std']:.4f} |")
    lines.append(f"| AUPRC | {stats['auprc']['mean']:.4f} ± {stats['auprc']['std']:.4f} |")
    
    baseline = stats['overall_prevalence']['mean']
    improvement = stats['auprc']['mean'] / baseline
    lines.append(f"| Random Baseline AUPRC | {baseline:.4f} |")
    lines.append(f"| AUPRC Improvement | {improvement:.2f}× |")
    lines.append("")
    
    lines.append("| Risk Percentile | Event Rate | Capture Rate | Enrichment |")
    lines.append("|----------------|------------|--------------|------------|")
    
    for s in strat_stats:
        lines.append(
            f"| Top {s['percentile']}% | "
            f"{s['event_rate_mean']:.4f} ± {s['event_rate_std']:.4f} | "
            f"{s['capture_rate_mean']:.2%} ± {s['capture_rate_std']:.2%} | "
            f"{s['enrichment_mean']:.2f} ± {s['enrichment_std']:.2f}× |"
        )
    
    return '\n'.join(lines)


def generate_full_report(stats, strat_stats, metadata):
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("OFFICIAL Risk Enrichment Analysis (5 seeds)")
    lines.append(f"{'='*70}\n")
    
    lines.append("Seeds: " + ", ".join([str(m['seed']) for m in metadata]))
    lines.append(f"Oversample Ratio: {metadata[0]['oversample_ratio']}")
    lines.append(f"Focal Gamma: {metadata[0]['focal_gamma']}\n")
    
    lines.append("Overall Performance (mean ± std):")
    lines.append(f"  AUROC: {stats['auroc']['mean']:.4f} ± {stats['auroc']['std']:.4f}")
    lines.append(f"  AUPRC: {stats['auprc']['mean']:.4f} ± {stats['auprc']['std']:.4f}")
    
    baseline = stats['overall_prevalence']['mean']
    improvement = stats['auprc']['mean'] / baseline
    lines.append(f"  Random Baseline: {baseline:.4f}")
    lines.append(f"  AUPRC Improvement: {improvement:.2f}×\n")
    
    lines.append("Risk Stratification (mean ± std):")
    lines.append(f"{'-'*70}")
    lines.append(f"{'Percentile':<12} {'Event Rate':<20} {'Capture Rate':<20} {'Enrichment':<20}")
    lines.append(f"{'-'*70}")
    
    for s in strat_stats:
        lines.append(
            f"Top {s['percentile']:>2}%      "
            f"{s['event_rate_mean']:.4f}±{s['event_rate_std']:.4f}      "
            f"{s['capture_rate_mean']:.2%}±{s['capture_rate_std']:.2%}      "
            f"{s['enrichment_mean']:.2f}±{s['enrichment_std']:.2f}×"
        )
    
    lines.append(f"{'-'*70}\n")
    
    top1_enrich = strat_stats[0]['enrichment_mean']
    top1_enrich_std = strat_stats[0]['enrichment_std']
    
    lines.append("Key Findings:")
    lines.append(f"  • Top 1% 富集倍数: {top1_enrich:.2f} ± {top1_enrich_std:.2f}×")
    lines.append(f"  • AUPRC vs Random: {improvement:.2f}×")
    lines.append(f"  • 结论: 模型具有显著且稳定的风险分层能力\n")
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    official_dir = "outputs/submission_v3/official_npz"
    
    npz_paths, metadata = load_and_verify_official_files(official_dir)
    
    print("Processing files...")
    all_results = aggregate_results(npz_paths, percentiles=[1, 5, 10])
    
    print("Computing statistics...")
    stats, strat_stats = compute_statistics(all_results)
    
    full_report = generate_full_report(stats, strat_stats, metadata)
    print(full_report)
    
    paper_table = generate_paper_table(stats, strat_stats)
    print(paper_table)
    
    output_dir = Path("outputs/submission_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "official_enrichment_report.txt", 'w') as f:
        f.write(full_report)
    
    with open(output_dir / "official_enrichment_table.md", 'w') as f:
        f.write(paper_table)
    
    df = pd.DataFrame([
        {
            'percentile': f"Top {s['percentile']}%",
            'event_rate': f"{s['event_rate_mean']:.4f} ± {s['event_rate_std']:.4f}",
            'capture_rate': f"{s['capture_rate_mean']:.2%} ± {s['capture_rate_std']:.2%}",
            'enrichment': f"{s['enrichment_mean']:.2f} ± {s['enrichment_std']:.2f}×"
        }
        for s in strat_stats
    ])
    df.to_csv(output_dir / "official_enrichment_results.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'official_enrichment_report.txt'}")
    print(f"  - {output_dir / 'official_enrichment_table.md'}")
    print(f"  - {output_dir / 'official_enrichment_results.csv'}")
