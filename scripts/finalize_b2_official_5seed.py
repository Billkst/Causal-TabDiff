#!/usr/bin/env python3
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 52, 62, 72, 82]
ROOT = Path('/home/UserData/ljx/Project_2/Causal-TabDiff')
FORMAL = ROOT / 'outputs/b2_baseline/formal_runs'
SUMMARY = ROOT / 'outputs/b2_baseline/summaries'
SUMMARY.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(values):
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return 'N/A'
    arr = np.array(vals, dtype=float)
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def collect(paths, key):
    vals = []
    for path in paths:
        data = load_json(path)
        if data is not None and key in data and data[key] is not None:
            vals.append(data[key])
    return vals


def make_layer1():
    models = {'CausalForest':'CausalForest','iTransformer':'iTransformer','TSDiff':'TSDiff','STaSy':'STaSy'}
    cols = ['auroc','auprc','f1','precision','recall','specificity','npv','accuracy','balanced_accuracy','mcc','brier_score','calibration_intercept','calibration_slope']
    rows=[]
    for name, stem in models.items():
        files=[FORMAL/'layer1'/f'{stem}_seed{seed}_metrics.json' for seed in SEEDS]
        row={'model':name}
        for col in cols:
            row[col]=fmt(collect(files,col))
        row['seeds_used']=','.join(map(str,SEEDS))
        row['protocol']='val_f1_threshold_then_fixed_test'
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'baseline_layer1_direct.csv', index=False)
    return df


def make_tstr():
    models=['tabsyn','tabdiff','survtraj','sssd','tsdiff','stasy']
    cols=['auroc','auprc','f1','precision','recall','specificity','mcc','brier_score','calibration_intercept','calibration_slope']
    rows=[]
    for model in models:
        files=[FORMAL/'tstr'/f'{model}_seed{seed}_tstr_metrics.json' for seed in SEEDS]
        row={'model':model}
        for col in cols:
            row[col]=fmt(collect(files,col))
        failure_files=[FORMAL/'tstr'/f'{model}_seed{seed}_FAILED.txt' for seed in SEEDS]
        failures=[f.read_text(encoding='utf-8').strip() for f in failure_files if f.exists()]
        row['status']='failed' if failures else 'ok'
        row['failure_reason']=failures[0] if failures else ''
        row['synthetic_sample_size']='1000'
        row['seeds_used']=','.join(map(str,SEEDS))
        row['protocol']='val_f1_threshold_then_fixed_test'
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'baseline_layer1_tstr.csv', index=False)
    return df


def make_layer2():
    models={'iTransformer':'iTransformer','TimeXer':'TimeXer','SSSD':'SSSD','SurvTraj':'SurvTraj'}
    rows=[]
    for name, stem in models.items():
        traj_files=[FORMAL/'layer2'/f'{stem}_seed{seed}_layer2_metrics.json' for seed in SEEDS]
        read_files=[FORMAL/'layer2'/f'{stem}_seed{seed}_layer2_readout_metrics.json' for seed in SEEDS]
        row={
            'model':name,
            'trajectory_mse':fmt(collect(traj_files,'trajectory_mse')),
            'trajectory_mae':fmt(collect(traj_files,'trajectory_mae')),
            'valid_coverage':fmt(collect(traj_files,'valid_coverage')),
            'readout_auroc':fmt(collect(read_files,'auroc')),
            'readout_auprc':fmt(collect(read_files,'auprc')),
            'readout_f1':fmt(collect(read_files,'f1')),
            'readout_precision':fmt(collect(read_files,'precision')),
            'readout_recall':fmt(collect(read_files,'recall')),
            'readout_brier':fmt(collect(read_files,'brier_score')),
            'calibration_intercept':fmt(collect(read_files,'calibration_intercept')),
            'calibration_slope':fmt(collect(read_files,'calibration_slope')),
            'seeds_used':','.join(map(str,SEEDS)),
            'protocol':'val_f1_threshold_then_fixed_test',
        }
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'baseline_layer2.csv', index=False)
    return df


def estimate_cf_params(model_path: Path):
    if not model_path.exists():
        return None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if not hasattr(model, 'estimators_'):
        return None
    return float(sum(est.tree_.node_count for est in model.estimators_))


def make_efficiency():
    sources={
        'CausalForest':('layer1','causal_forest_efficiency_seed{seed}.json'),
        'iTransformer':('layer1','itransformer_efficiency_seed{seed}.json'),
        'TSDiff':('layer1','tsdiff_efficiency_seed{seed}.json'),
        'STaSy':('layer1','stasy_efficiency_seed{seed}.json'),
        'TimeXer':('layer2','timexer_efficiency_seed{seed}.json'),
        'TabSyn':('tstr','tabsyn_efficiency_seed{seed}.json'),
        'TabDiff':('tstr','tabdiff_efficiency_seed{seed}.json'),
        'SurvTraj':('tstr','survtraj_efficiency_seed{seed}.json'),
        'SSSD':('tstr','sssd_efficiency_seed{seed}.json'),
    }
    rows=[]
    for model,(subdir,pattern) in sources.items():
        files=[FORMAL/subdir/pattern.format(seed=seed) for seed in SEEDS]
        metrics=[load_json(f) for f in files if f.exists()]
        row={'model':model}
        for key in ['total_params','trainable_params','inference_latency_ms_per_sample','throughput_samples_per_sec','total_training_wall_clock_sec','peak_gpu_memory_mb']:
            row[key]=fmt([m.get(key) for m in metrics if m is not None and key in m])
        if model=='CausalForest':
            cf_vals=[estimate_cf_params(FORMAL/'layer1'/f'causal_forest_seed{seed}_model.pkl') for seed in SEEDS]
            row['total_params']=fmt(cf_vals)
            row['trainable_params']=fmt(cf_vals)
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'baseline_efficiency.csv', index=False)
    return df


def formal_pred_exists(task, model, seed):
    if task=='layer1_direct':
        if model=='CausalForest':
            pred=FORMAL/'layer1'/f'causal_forest_seed{seed}_predictions.npz'
        elif model=='iTransformer':
            pred=FORMAL/'layer1'/f'itransformer_seed{seed}_predictions.npz'
        elif model=='TSDiff':
            pred=FORMAL/'layer1'/f'tsdiff_seed{seed}_predictions.npz'
        else:
            pred=FORMAL/'layer1'/f'stasy_seed{seed}_predictions.npz'
        metrics=FORMAL/'layer1'/f'{model}_seed{seed}_metrics.json'
    elif task=='layer1_tstr':
        pred=FORMAL/'tstr'/f'{model}_seed{seed}_predictions.npz'
        metrics=FORMAL/'tstr'/f'{model}_seed{seed}_tstr_metrics.json'
    else:
        stem=model.lower() if model in ['iTransformer','TimeXer'] else model.lower()
        pred=FORMAL/'layer2'/f'{stem}_seed{seed}_layer2.npz'
        metrics=FORMAL/'layer2'/f'{model}_seed{seed}_layer2_readout_metrics.json'
    if not pred.exists():
        return False, False, metrics.exists(), None
    data=np.load(pred)
    has_val='val_y_true' in data.files and 'val_y_pred' in data.files
    has_test='test_y_true' in data.files and 'test_y_pred' in data.files
    threshold=None
    m=load_json(metrics)
    if m is not None:
        threshold=m.get('threshold', m.get('optimal_threshold'))
    return has_val, has_test, metrics.exists(), threshold


def make_protocol_check():
    rows=[]
    for model in ['CausalForest','iTransformer','TSDiff','STaSy']:
        for seed in SEEDS:
            has_val, has_test, has_metrics, threshold = formal_pred_exists('layer1_direct', model, seed)
            rows.append({'task':'layer1_direct','model':model,'seed':seed,'has_val_predictions':has_val,'has_test_predictions':has_test,'metrics_file_exists':has_metrics,'threshold':threshold,'formal_protocol':bool(has_val and has_test and has_metrics)})
    for model in ['tabsyn','tabdiff','survtraj','sssd','tsdiff','stasy']:
        for seed in SEEDS:
            has_val, has_test, has_metrics, threshold = formal_pred_exists('layer1_tstr', model, seed)
            rows.append({'task':'layer1_tstr','model':model,'seed':seed,'has_val_predictions':has_val,'has_test_predictions':has_test,'metrics_file_exists':has_metrics,'threshold':threshold,'formal_protocol':bool(has_val and has_test and has_metrics)})
    for model in ['iTransformer','TimeXer','SSSD','SurvTraj']:
        for seed in SEEDS:
            has_val, has_test, has_metrics, threshold = formal_pred_exists('layer2', model, seed)
            rows.append({'task':'layer2','model':model,'seed':seed,'has_val_predictions':has_val,'has_test_predictions':has_test,'metrics_file_exists':has_metrics,'threshold':threshold,'formal_protocol':bool(has_val and has_test and has_metrics)})
    df=pd.DataFrame(rows)
    df.to_csv(SUMMARY/'baseline_protocol_consistency_check.csv', index=False)
    return df


def md_table(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_report(layer1_df, tstr_df, layer2_df, eff_df, check_df):
    lines=[]
    all_pass = bool(check_df['formal_protocol'].all())
    lines.append('# Baseline 对比实验最终报告')
    lines.append('')
    lines.append('## 最终正式协议')
    lines.append('')
    lines.append('- 正式 seeds 固定为：42, 52, 62, 72, 82')
    lines.append('- 正式阈值协议固定为：验证集 F1 最大化选阈值，测试集固定该阈值')
    lines.append('- 所有非正式扩展 seeds 均不纳入正式表')
    lines.append('')
    lines.append('## Layer1 直接预测结果')
    lines.append('')
    lines.append(md_table(layer1_df))
    lines.append('')
    lines.append('## Layer1 TSTR 结果')
    lines.append('')
    lines.append(md_table(tstr_df))
    lines.append('')
    lines.append('## Layer2 结果')
    lines.append('')
    lines.append(md_table(layer2_df))
    lines.append('')
    lines.append('## 效率结果')
    lines.append('')
    lines.append(md_table(eff_df))
    lines.append('')
    lines.append('## 异常复核')
    lines.append('')
    lines.append('- SSSD 的 collapse/退化问题以正式 TSTR 重跑后的结果为准。')
    lines.append('- STaSy 的异常以正式协议重算后的结果为准；若 AUROC 仍偏低，则按性能差而非旧协议污染处理。')
    lines.append('- iTransformer 的旧 0.5 阈值异常已由验证集 F1 选阈值正式口径替代。')
    lines.append('')
    lines.append('## 协议一致性检查')
    lines.append('')
    lines.append(f"- 正式协议通过率：{check_df['formal_protocol'].mean():.4f}")
    lines.append('- 一致性检查文件：`outputs/b2_baseline/summaries/baseline_protocol_consistency_check.csv`')
    lines.append('')
    lines.append('## 最终状态')
    lines.append('')
    if all_pass:
        lines.append('- 当前官方 5-seed baseline 已全部按唯一正式协议通过，并完成正式封版。')
        lines.append('- 现在可以在不再依赖旧协议、旧 seeds 或临时修补表格的前提下进入 Ours 正式实验。')
    else:
        lines.append('- 只有当一致性检查表中所有正式行都通过，且所有正式结果表均已落盘时，baseline 才能视为完全正式封版。')
        lines.append('- 若仍存在失败 baseline，将以显式失败条目保留，而不会再用旧协议数字进行回填。')
    (ROOT/'BASELINE_COMPARISON_REPORT.md').write_text("\n".join(lines), encoding='utf-8')


def main():
    layer1_df=make_layer1()
    tstr_df=make_tstr()
    layer2_df=make_layer2()
    eff_df=make_efficiency()
    check_df=make_protocol_check()
    write_report(layer1_df, tstr_df, layer2_df, eff_df, check_df)
    print('finalized official 5-seed outputs')


if __name__ == '__main__':
    main()
