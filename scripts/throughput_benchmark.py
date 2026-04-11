#!/usr/bin/env python -u
"""
Throughput Benchmark: OOM Cliff Search + Sustained Throughput Measurement

用途: 在不跑完整训练的前提下，快速找到单卡/多卡最大稳定 batch_size 和最优 num_workers。
策略: 从最大 batch_size 开始，OOM 则回退；找到上限后精测吞吐。

用法:
  # Phase 1: OOM cliff search (单卡, 固定 nw=8)
  CUDA_VISIBLE_DEVICES=0 python -u scripts/throughput_benchmark.py --mode oom_search --num_workers 8

  # Phase 2: 吞吐精测 (单卡, 对 top batch_size 扫 num_workers)
  CUDA_VISIBLE_DEVICES=0 python -u scripts/throughput_benchmark.py --mode throughput --batch_sizes 4096,2048,1024 --num_workers_list 4,8,12,16

  # Phase 3: 多卡并发稳定性 (在 launch 脚本中调用)
  CUDA_VISIBLE_DEVICES=0 python -u scripts/throughput_benchmark.py --mode stability --batch_size 2048 --num_workers 8 --tag worker0 &
  CUDA_VISIBLE_DEVICES=1 python -u scripts/throughput_benchmark.py --mode stability --batch_size 2048 --num_workers 8 --tag worker1 &
  ...
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.data_module_landmark import load_and_split_data, collate_fn
from src.data.data_module_landmark_cached import LandmarkDatasetCached
from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory


# ---------- Loss functions (copied from training script, frozen) ----------

def focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = ((1 - pt) ** gamma) * bce_loss
    if pos_weight is not None:
        weight = torch.where(targets > 0.5, pos_weight.reshape(-1, 1), torch.ones_like(targets))
        focal_loss = focal_loss * weight
    return focal_loss.mean()


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, n_tasks=3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigma[i])
            total = total + precision * loss + self.log_sigma[i]
        return total


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------- Helpers ----------

def build_alpha_target(batch, device):
    if 'alpha_target' in batch:
        return batch['alpha_target'].float().to(device)
    return batch['landmark'].float().to(device)


def get_gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def get_cpu_percent():
    if HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1)
    return -1.0


def setup_data(table_path, seed=42):
    """加载数据并缓存，返回 dataset + metadata"""
    train_df, val_df, _, landmark_to_idx = load_and_split_data(table_path, seed)
    train_dataset = LandmarkDatasetCached(train_df, landmark_to_idx)

    train_labels = np.asarray(train_df['y_2year'], dtype=int)
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0

    sample_weights = np.where(train_labels == 1, 5.0 * n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(sample_weights.tolist(), len(train_labels), replacement=True)

    quick_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    sample = next(iter(quick_loader))
    t_steps = sample['x'].shape[1]
    feature_dim = sample['x'].shape[2]
    trajectory_len = sample['trajectory_target'].shape[1]
    del quick_loader

    return {
        'train_dataset': train_dataset,
        'sampler': sampler,
        'pos_weight_val': pos_weight_val,
        't_steps': t_steps,
        'feature_dim': feature_dim,
        'trajectory_len': trajectory_len,
        'n_train': len(train_dataset),
    }


def make_loader(dataset, batch_size, num_workers, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_model_and_opt(t_steps, feature_dim, trajectory_len, device):
    """构建模型 + finetune optimizer (最重的配置)"""
    model = CausalTabDiffTrajectory(t_steps, feature_dim, 100, trajectory_len).to(device)
    # Unfreeze all (finetune mode = worst case memory)
    for p in model.parameters():
        p.requires_grad = True

    uw_loss = UncertaintyWeightedLoss(n_tasks=3).to(device)
    uw_loss.log_sigma.data[2] = -1.0

    optimizer = torch.optim.Adam(
        [
            {'params': model.risk_head.parameters(), 'lr': 1e-3},
            {'params': model.trajectory_head.parameters(), 'lr': 5e-4},
            {'params': model.base_model.parameters(), 'lr': 1e-4},
            {'params': uw_loss.parameters(), 'lr': 1e-2},
        ],
        weight_decay=1e-4,
    )
    return model, uw_loss, optimizer


def run_one_step(model, uw_loss, optimizer, batch, device, pos_weight):
    """执行一个完整的 finetune training step (最重路径)"""
    x = batch['x'].to(device)
    alpha = build_alpha_target(batch, device)
    y_2year = batch['y_2year'].to(device)
    traj_target = batch['trajectory_target'].to(device)
    traj_mask = batch['trajectory_valid_mask'].to(device)
    hl = batch['history_length'].to(device)

    optimizer.zero_grad()
    outputs = model(x, alpha, history_length=hl)

    loss_gen = outputs['diff_loss'] + 0.5 * outputs['disc_loss']
    loss_traj = model.compute_trajectory_loss(outputs['trajectory_logits'], traj_target, traj_mask)
    loss_cls = focal_loss_with_logits(outputs['risk_2year_logit'], y_2year, gamma=2.0, pos_weight=pos_weight)
    loss = uw_loss([loss_gen, loss_traj, loss_cls])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# ---------- Mode: OOM Cliff Search ----------

def _test_one_bs(bs, args, data, device, pos_weight):
    """测试单个 batch_size，返回 result dict 或 None(OOM/error)"""
    if bs > data['n_train']:
        print(f"  SKIP: bs={bs} > n_train={data['n_train']}", flush=True)
        return {'batch_size': bs, 'status': 'skipped', 'reason': 'exceeds dataset'}

    try:
        loader = make_loader(data['train_dataset'], bs, args.num_workers, data['sampler'])
        model, uw_loss, optimizer = build_model_and_opt(
            data['t_steps'], data['feature_dim'], data['trajectory_len'], device
        )

        batch_iter = iter(loader)
        for _ in range(2):
            batch = next(batch_iter)
            _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
        torch.cuda.synchronize()

        step_times = []
        for _ in range(5):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
            t0 = time.perf_counter()
            _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        gpu_peak = get_gpu_mem_mb()
        avg_step = np.mean(step_times)
        std_step = np.std(step_times)
        samples_sec = bs / avg_step

        result = {
            'batch_size': bs,
            'status': 'ok',
            'avg_step_sec': round(avg_step, 4),
            'std_step_sec': round(std_step, 4),
            'samples_sec': round(samples_sec, 1),
            'steps_sec': round(1.0 / avg_step, 2),
            'gpu_peak_mb': round(gpu_peak, 1),
            'gpu_util_pct': round(gpu_peak / (16 * 1024) * 100, 1),
        }

        print(f"  OK | step={avg_step:.4f}s ± {std_step:.4f} | "
              f"samples/s={samples_sec:.0f} | GPU={gpu_peak:.0f}MB ({result['gpu_util_pct']:.1f}%)",
              flush=True)

        del model, uw_loss, optimizer, loader, batch_iter
        reset_gpu()
        return result

    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
            print(f"  OOM at batch_size={bs} | Error: {str(e)[:120]}", flush=True)
            reset_gpu()
            return {'batch_size': bs, 'status': 'oom', 'error': str(e)[:200]}
        print(f"  ERROR at batch_size={bs} | {str(e)[:200]}", flush=True)
        reset_gpu()
        return {'batch_size': bs, 'status': 'error', 'error': str(e)[:200]}
    except Exception as e:
        print(f"  ERROR at batch_size={bs} | {str(e)[:200]}", flush=True)
        reset_gpu()
        return {'batch_size': bs, 'status': 'error', 'error': str(e)[:200]}


def oom_search(args, data):
    """从最大 batch_size 开始，OOM 则回退；最大候选通过则自动 2x 上探直到 OOM 或超数据集"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    candidates = [int(x) for x in args.batch_sizes_oom.split(',')]
    candidates.sort(reverse=True)

    pos_weight = torch.tensor([data['pos_weight_val']], device=device)
    results = []

    # 自动上探：如果最大候选通过，继续 2x 直到 OOM 或超数据集
    max_candidate = candidates[0]
    upward_probes = []
    probe = max_candidate * 2
    while probe <= data['n_train']:
        upward_probes.append(probe)
        probe *= 2
    if upward_probes:
        candidates = sorted(upward_probes, reverse=True) + candidates

    print(f"\n{'='*72}", flush=True)
    print(f"OOM CLIFF SEARCH | GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Candidates (descending): {candidates}", flush=True)
    if upward_probes:
        print(f"Auto-probed upward: {sorted(upward_probes, reverse=True)}", flush=True)
    print(f"num_workers: {args.num_workers} | train_samples: {data['n_train']}", flush=True)
    print(f"{'='*72}\n", flush=True)

    for bs in candidates:
        reset_gpu()
        print(f"--- Testing batch_size={bs} ---", flush=True)
        result = _test_one_bs(bs, args, data, device, pos_weight)
        results.append(result)

    # Summary
    print(f"\n{'='*72}", flush=True)
    print("OOM CLIFF SEARCH SUMMARY", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"{'BS':>8} | {'Status':>8} | {'step(s)':>10} | {'samp/s':>10} | {'GPU MB':>10} | {'GPU%':>6}", flush=True)
    print(f"{'-'*72}", flush=True)
    for r in results:
        if r['status'] == 'ok':
            print(f"{r['batch_size']:>8} | {'OK':>8} | {r['avg_step_sec']:>10.4f} | "
                  f"{r['samples_sec']:>10.0f} | {r['gpu_peak_mb']:>10.0f} | {r['gpu_util_pct']:>5.1f}%",
                  flush=True)
        else:
            print(f"{r['batch_size']:>8} | {r['status'].upper():>8} |", flush=True)

    # Find optimal
    ok_results = [r for r in results if r['status'] == 'ok']
    if ok_results:
        best = max(ok_results, key=lambda r: r['samples_sec'])
        print(f"\nBest throughput: bs={best['batch_size']} → {best['samples_sec']:.0f} samples/s", flush=True)

    # Save
    out_path = os.path.join(args.output_dir, f"oom_search_nw{args.num_workers}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


# ---------- Mode: Throughput Sweep ----------

def throughput_sweep(args, data):
    """对指定 batch_sizes × num_workers 做精确吞吐测量"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    nw_list = [int(x) for x in args.num_workers_list.split(',')]
    pos_weight = torch.tensor([data['pos_weight_val']], device=device)

    WARMUP_STEPS = 5
    TIMED_STEPS = 20

    print(f"\n{'='*72}", flush=True)
    print(f"THROUGHPUT SWEEP | GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"batch_sizes: {batch_sizes} | num_workers: {nw_list}", flush=True)
    print(f"warmup={WARMUP_STEPS} steps, timed={TIMED_STEPS} steps", flush=True)
    print(f"{'='*72}\n", flush=True)

    results = []

    for bs in batch_sizes:
        for nw in nw_list:
            reset_gpu()
            tag = f"bs={bs},nw={nw}"
            print(f"--- {tag} ---", flush=True)

            try:
                loader = make_loader(data['train_dataset'], bs, nw, data['sampler'])
                model, uw_loss, optimizer = build_model_and_opt(
                    data['t_steps'], data['feature_dim'], data['trajectory_len'], device
                )

                batch_iter = iter(loader)

                # Warmup
                for _ in range(WARMUP_STEPS):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(loader)
                        batch = next(batch_iter)
                    _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
                torch.cuda.synchronize()

                # Timed
                cpu_before = get_cpu_percent()
                step_times = []
                for _ in range(TIMED_STEPS):
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        batch_iter = iter(loader)
                        batch = next(batch_iter)
                    t0 = time.perf_counter()
                    _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
                    torch.cuda.synchronize()
                    step_times.append(time.perf_counter() - t0)

                cpu_after = get_cpu_percent()
                gpu_peak = get_gpu_mem_mb()

                avg_step = np.mean(step_times)
                std_step = np.std(step_times)
                p50 = np.percentile(step_times, 50)
                p95 = np.percentile(step_times, 95)
                jitter = (p95 - p50) / p50 * 100 if p50 > 0 else 0

                result = {
                    'batch_size': bs,
                    'num_workers': nw,
                    'avg_step_sec': round(avg_step, 5),
                    'std_step_sec': round(std_step, 5),
                    'p50_step_sec': round(p50, 5),
                    'p95_step_sec': round(p95, 5),
                    'jitter_pct': round(jitter, 1),
                    'samples_sec': round(bs / avg_step, 1),
                    'steps_sec': round(1.0 / avg_step, 2),
                    'gpu_peak_mb': round(gpu_peak, 1),
                    'cpu_pct': round(cpu_after, 1),
                }
                results.append(result)

                print(f"  step={avg_step:.5f}s ± {std_step:.5f} | samp/s={result['samples_sec']:.0f} | "
                      f"jitter={jitter:.1f}% | GPU={gpu_peak:.0f}MB | CPU={cpu_after:.0f}%",
                      flush=True)

                del model, uw_loss, optimizer, loader, batch_iter
                reset_gpu()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  OOM at {tag}", flush=True)
                    results.append({'batch_size': bs, 'num_workers': nw, 'status': 'oom'})
                    reset_gpu()
                else:
                    print(f"  ERROR at {tag}: {str(e)[:150]}", flush=True)
                    results.append({'batch_size': bs, 'num_workers': nw, 'status': 'error'})
                    reset_gpu()

    # Summary table
    print(f"\n{'='*90}", flush=True)
    print("THROUGHPUT SWEEP SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'BS':>8} | {'NW':>4} | {'step(s)':>10} | {'samp/s':>10} | {'jitter%':>8} | {'GPU MB':>8} | {'CPU%':>6}", flush=True)
    print(f"{'-'*90}", flush=True)
    for r in results:
        if 'status' in r:
            print(f"{r['batch_size']:>8} | {r['num_workers']:>4} | {r.get('status','?').upper():>10} |", flush=True)
        else:
            print(f"{r['batch_size']:>8} | {r['num_workers']:>4} | {r['avg_step_sec']:>10.5f} | "
                  f"{r['samples_sec']:>10.0f} | {r['jitter_pct']:>7.1f}% | {r['gpu_peak_mb']:>8.0f} | "
                  f"{r['cpu_pct']:>5.0f}%", flush=True)

    ok_results = [r for r in results if 'status' not in r]
    if ok_results:
        best = max(ok_results, key=lambda r: r['samples_sec'])
        print(f"\nBest throughput: bs={best['batch_size']}, nw={best['num_workers']} → "
              f"{best['samples_sec']:.0f} samples/s (jitter={best['jitter_pct']:.1f}%)", flush=True)

    out_path = os.path.join(args.output_dir, "throughput_sweep.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return results


# ---------- Mode: Stability Test ----------

def stability_test(args, data):
    """持续跑 N 个 step，测量吞吐稳定性 (用于多卡并发场景)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_weight = torch.tensor([data['pos_weight_val']], device=device)

    WARMUP_STEPS = 10
    TIMED_STEPS = args.stability_steps

    tag = args.tag or f"gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"

    print(f"\n{'='*72}", flush=True)
    print(f"STABILITY TEST [{tag}] | GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"bs={args.batch_size}, nw={args.num_workers}, steps={TIMED_STEPS}", flush=True)
    print(f"{'='*72}\n", flush=True)

    loader = make_loader(data['train_dataset'], args.batch_size, args.num_workers, data['sampler'])
    model, uw_loss, optimizer = build_model_and_opt(
        data['t_steps'], data['feature_dim'], data['trajectory_len'], device
    )

    batch_iter = iter(loader)

    # Warmup
    print(f"[{tag}] Warmup {WARMUP_STEPS} steps...", flush=True)
    for _ in range(WARMUP_STEPS):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)
        _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
    torch.cuda.synchronize()

    # Timed
    print(f"[{tag}] Timed {TIMED_STEPS} steps...", flush=True)
    step_times = []
    for s in range(TIMED_STEPS):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)
        t0 = time.perf_counter()
        _ = run_one_step(model, uw_loss, optimizer, batch, device, pos_weight)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_times.append(elapsed)

        if (s + 1) % 20 == 0:
            recent = step_times[-20:]
            print(f"  [{tag}] step {s+1}/{TIMED_STEPS} | "
                  f"last20_avg={np.mean(recent):.5f}s | samp/s={args.batch_size/np.mean(recent):.0f} | "
                  f"GPU={get_gpu_mem_mb():.0f}MB | CPU={get_cpu_percent():.0f}%",
                  flush=True)

    gpu_peak = get_gpu_mem_mb()

    # Analysis
    avg_step = np.mean(step_times)
    std_step = np.std(step_times)
    p50 = np.percentile(step_times, 50)
    p95 = np.percentile(step_times, 95)
    p99 = np.percentile(step_times, 99)
    jitter = (p95 - p50) / p50 * 100 if p50 > 0 else 0

    # Detect drift: compare first half vs second half
    first_half = np.mean(step_times[:len(step_times)//2])
    second_half = np.mean(step_times[len(step_times)//2:])
    drift_pct = abs(second_half - first_half) / first_half * 100

    samples_sec = round(args.batch_size / avg_step, 1)

    degradation_pct = 0.0
    degradation_ok = True
    if args.single_card_sps > 0:
        degradation_pct = round((1.0 - samples_sec / args.single_card_sps) * 100, 1)
        degradation_ok = degradation_pct <= 25.0

    result = {
        'tag': tag,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'total_steps': TIMED_STEPS,
        'avg_step_sec': round(avg_step, 5),
        'std_step_sec': round(std_step, 5),
        'p50_step_sec': round(p50, 5),
        'p95_step_sec': round(p95, 5),
        'p99_step_sec': round(p99, 5),
        'jitter_pct': round(jitter, 1),
        'drift_pct': round(drift_pct, 1),
        'samples_sec': samples_sec,
        'gpu_peak_mb': round(gpu_peak, 1),
        'single_card_sps_baseline': args.single_card_sps,
        'degradation_pct': degradation_pct,
        'degradation_ok': degradation_ok,
        'stable': jitter < 20 and drift_pct < 10 and degradation_ok,
    }

    print(f"\n{'='*72}", flush=True)
    print(f"STABILITY RESULT [{tag}]", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  avg_step:   {avg_step:.5f}s ± {std_step:.5f}", flush=True)
    print(f"  p50/p95/p99: {p50:.5f} / {p95:.5f} / {p99:.5f}", flush=True)
    print(f"  jitter:     {jitter:.1f}%", flush=True)
    print(f"  drift:      {drift_pct:.1f}% (first_half={first_half:.5f} vs second_half={second_half:.5f})", flush=True)
    print(f"  samples/s:  {samples_sec:.0f}", flush=True)
    print(f"  GPU peak:   {gpu_peak:.0f} MB", flush=True)
    if args.single_card_sps > 0:
        print(f"  degradation: {degradation_pct:.1f}% vs single-card {args.single_card_sps:.0f} samp/s "
              f"({'OK ✓' if degradation_ok else 'FAIL ✗ (>25%)'})", flush=True)
    print(f"  STABLE:     {'YES ✓' if result['stable'] else 'NO ✗'}", flush=True)

    out_path = os.path.join(args.output_dir, f"stability_{tag}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(_json_safe(result), f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    return result


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Throughput Benchmark for Causal-TabDiff')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['oom_search', 'throughput', 'stability'],
                        help='Benchmark mode')
    parser.add_argument('--table_path', type=str,
                        default='data/landmark_tables/unified_person_landmark_table.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='logs/benchmark')

    # OOM search
    parser.add_argument('--batch_sizes_oom', type=str,
                        default='32768,16384,8192,4096,2048,1024,512,256',
                        help='Comma-separated batch sizes for OOM search (descending)')

    # Throughput sweep
    parser.add_argument('--batch_sizes', type=str, default='4096,2048,1024',
                        help='Comma-separated batch sizes for throughput sweep')
    parser.add_argument('--num_workers_list', type=str, default='4,8,12,16',
                        help='Comma-separated num_workers for throughput sweep')

    # Stability / shared
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--stability_steps', type=int, default=100)
    parser.add_argument('--single_card_sps', type=float, default=0.0,
                        help='Single-card samples/sec baseline for degradation check (0=skip)')
    parser.add_argument('--tag', type=str, default=None, help='Worker tag for stability test')

    args = parser.parse_args()

    print(f"Loading data from {args.table_path} ...", flush=True)
    data = setup_data(args.table_path, args.seed)
    print(f"Data loaded: {data['n_train']} train samples, "
          f"t_steps={data['t_steps']}, feature_dim={data['feature_dim']}, "
          f"trajectory_len={data['trajectory_len']}", flush=True)

    if args.mode == 'oom_search':
        oom_search(args, data)
    elif args.mode == 'throughput':
        throughput_sweep(args, data)
    elif args.mode == 'stability':
        stability_test(args, data)


if __name__ == '__main__':
    main()
