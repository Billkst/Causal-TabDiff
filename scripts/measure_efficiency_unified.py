#!/usr/bin/env python3
"""
统一效率测量脚本 — 对所有有 checkpoint 的模型用相同方法测量推理延迟。

测量协议：
  1. 加载模型到 GPU
  2. Warmup 20 次推理（排除 JIT 编译等冷启动开销）
  3. cuda.synchronize()
  4. 计时 1000 次推理
  5. cuda.synchronize()
  6. 计算 per-sample 延迟和吞吐量

输出: outputs/fulldata_baselines/formal_runs/efficiency_unified/{model}_seed{seed}.json
"""
import sys, os, json, time, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "baselines" / "stasy"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "baselines" / "tabsyn"))

import numpy as np
import torch

WARMUP_SAMPLES = 20
TIMED_SAMPLES = 1000
OUTPUT_DIR = PROJECT_ROOT / "outputs/fulldata_baselines/formal_runs/efficiency_unified"


def measure_latency(inference_fn, n_warmup, n_timed, device):
    """统一测量函数：warmup + sync + timed + sync"""
    # Warmup
    for _ in range(n_warmup):
        inference_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Timed
    start = time.perf_counter()
    for _ in range(n_timed):
        inference_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    latency_ms = (elapsed / n_timed) * 1000
    throughput = n_timed / elapsed

    return {
        "inference_latency_ms_per_sample": latency_ms,
        "throughput_samples_per_sec": throughput,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "timed_samples": n_timed,
        "warmup_samples": n_warmup,
    }


def measure_batch_latency(inference_fn, batch_size, n_warmup_batches, n_timed_batches, device):
    """统一测量函数（批量版）：warmup + sync + timed + sync，per-sample = total / (n_batches * batch_size)"""
    for _ in range(n_warmup_batches):
        inference_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(n_timed_batches):
        inference_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    total_samples = n_timed_batches * batch_size
    latency_ms = (elapsed / total_samples) * 1000
    throughput = total_samples / elapsed

    return {
        "inference_latency_ms_per_sample": latency_ms,
        "throughput_samples_per_sec": throughput,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "timed_samples": total_samples,
        "warmup_samples": n_warmup_batches * batch_size,
    }


def save_result(result, model_name, seed):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{model_name}_seed{seed}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [Saved] {path}", flush=True)


# ============================================================
# CausalTabDiff — 从 checkpoint 加载，测 forward 推理
# ============================================================
def measure_causal_tabdiff(seed, device):
    print(f"\n=== CausalTabDiff seed={seed} ===", flush=True)
    from src.data.data_module_landmark import load_and_split_data, LandmarkDataset, collate_fn
    from src.models.causal_tabdiff_trajectory import CausalTabDiffTrajectory
    from torch.utils.data import DataLoader

    ckpt_path = PROJECT_ROOT / f"checkpoints/landmark/best_model_seed{seed}.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}", flush=True)
        return

    table_path = str(PROJECT_ROOT / "data/landmark_tables/unified_person_landmark_table.pkl")
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(table_path, seed=seed)
    test_dataset = LandmarkDataset(test_df, landmark_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    sample_batch = next(iter(test_loader))
    t_steps = sample_batch["x"].shape[1]
    feature_dim = sample_batch["x"].shape[2]

    model = CausalTabDiffTrajectory(
        t_steps=t_steps, feature_dim=feature_dim,
        diffusion_steps=100, trajectory_len=7,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # 获取模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 准备一个固定 batch 用于测量
    test_batch = next(iter(test_loader))
    x = test_batch["x"].to(device)
    alpha = test_batch["alpha_target"].float().to(device) if "alpha_target" in test_batch else test_batch["landmark"].float().to(device)
    history_length = test_batch["history_length"].to(device)
    batch_size = x.shape[0]

    def inference_fn():
        with torch.no_grad():
            model(x, alpha, history_length=history_length)

    result = measure_batch_latency(inference_fn, batch_size, n_warmup_batches=5, n_timed_batches=50, device=device)
    result["total_params"] = total_params
    result["trainable_params"] = trainable_params
    result["nfe"] = 100
    result["type"] = "Diffusion"

    print(f"  Latency: {result['inference_latency_ms_per_sample']:.4f} ms/sample", flush=True)
    print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec", flush=True)
    print(f"  Params: {total_params}", flush=True)
    save_result(result, "CausalTabDiff", seed)


def measure_causal_tabdiff_l1(seed, device):
    """CausalTabDiff Layer1: 完整扩散采样 (sample_with_guidance)"""
    print(f"\n=== CausalTabDiff_L1 seed={seed} ===", flush=True)
    from src.models.causal_tabdiff import CausalTabDiff

    ckpt_path = PROJECT_ROOT / f"checkpoints/landmark/best_model_seed{seed}.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}", flush=True)
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_sd = {k.replace('base_model.', ''): v for k, v in checkpoint['model_state_dict'].items() if k.startswith('base_model.')}

    model = CausalTabDiff(t_steps=3, feature_dim=15, diffusion_steps=100).to(device)
    model.load_state_dict(base_sd)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    sample_batch_size = 100
    alpha_target = torch.rand(sample_batch_size, 1, device=device)

    def inference_fn():
        with torch.no_grad():
            model.sample_with_guidance(
                batch_size=sample_batch_size,
                alpha_target=alpha_target,
                guidance_scale=1.0,
                guidance_schedule='constant',
                guidance_power=2.0,
            )

    result = measure_batch_latency(inference_fn, sample_batch_size, n_warmup_batches=2, n_timed_batches=10, device=device)
    result["total_params"] = total_params
    result["nfe"] = 100
    result["type"] = "Diffusion"

    print(f"  Latency: {result['inference_latency_ms_per_sample']:.4f} ms/sample", flush=True)
    print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec", flush=True)
    print(f"  Params: {total_params}", flush=True)
    save_result(result, "CausalTabDiff_L1", seed)


# ============================================================
# TSTR 生成模型 — 复用已有 generative JSON（已有 warmup，sync 差异可忽略）
# ============================================================
def measure_tstr_generative(model_name, seed, device):
    print(f"\n=== TSTR Generative: {model_name} seed={seed} (reuse existing JSON) ===", flush=True)

    src_path = PROJECT_ROOT / f"outputs/fulldata_baselines/formal_runs/efficiency/generative_{model_name}_seed{seed}.json"
    if not src_path.exists():
        print(f"  [SKIP] Source JSON not found: {src_path}", flush=True)
        return

    with open(src_path) as f:
        src = json.load(f)

    nfe_map = {"tsdiff": 1000, "stasy": 50, "tabsyn": 50, "tabdiff": 50, "sssd": 100, "survtraj": 100}

    result = {
        "inference_latency_ms_per_sample": src["generative_inference_latency_ms_per_sample"],
        "throughput_samples_per_sec": src["generative_throughput_samples_per_sec"],
        "peak_gpu_memory_mb": src.get("peak_gpu_memory_mb", 0.0),
        "total_params": src.get("total_params", 0),
        "type": "TSTR",
        "nfe": nfe_map.get(model_name),
        "source": "generative_json_reuse",
    }

    print(f"  Latency: {result['inference_latency_ms_per_sample']:.4f} ms/sample", flush=True)
    print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec", flush=True)
    print(f"  Params: {result['total_params']}", flush=True)
    save_result(result, f"generative_{model_name}", seed)


# ============================================================
# CausalForest — 从 pkl 加载，测 predict_proba
# ============================================================
def measure_causal_forest(seed, device):
    print(f"\n=== CausalForest seed={seed} ===", flush=True)
    import pickle

    pkl_path = PROJECT_ROOT / f"outputs/fulldata_baselines/layer1/causal_forest_seed{seed}_model.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] Model not found: {pkl_path}", flush=True)
        return

    with open(pkl_path, "rb") as f:
        clf = pickle.load(f)

    # 生成与训练数据维度一致的假数据
    n_features = clf.n_features_in_
    X_fake = np.random.randn(TIMED_SAMPLES, n_features).astype(np.float32)

    def inference_fn():
        clf.predict_proba(X_fake)

    result = measure_latency(inference_fn, n_warmup=WARMUP_SAMPLES, n_timed=50, device=device)
    # 修正：每次调用处理 TIMED_SAMPLES 个样本
    result["inference_latency_ms_per_sample"] = result["inference_latency_ms_per_sample"] / TIMED_SAMPLES
    result["throughput_samples_per_sec"] = result["throughput_samples_per_sec"] * TIMED_SAMPLES
    result["total_params"] = "N/A"
    result["type"] = "Tree"
    result["nfe"] = None

    print(f"  Latency: {result['inference_latency_ms_per_sample']:.6f} ms/sample", flush=True)
    print(f"  Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec", flush=True)
    save_result(result, "CausalForest", seed)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Unified efficiency measurement")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "causal_tabdiff", "causal_tabdiff_l1", "tstr_generative", "causal_forest"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}", flush=True)

    if args.model in ("all", "causal_tabdiff"):
        for seed in [42, 52, 62, 72, 82]:
            measure_causal_tabdiff(seed, device)

    if args.model in ("all", "causal_tabdiff_l1"):
        for seed in [42, 52, 62, 72, 82]:
            measure_causal_tabdiff_l1(seed, device)

    if args.model in ("all", "tstr_generative"):
        for model_name in ["sssd", "tabsyn", "tabdiff", "survtraj", "tsdiff", "stasy"]:
            measure_tstr_generative(model_name, args.seed, device)

    if args.model in ("all", "causal_forest"):
        for seed in [42, 52, 62, 72, 82]:
            measure_causal_forest(seed, device)

    print("\n[All Done] Unified efficiency measurement complete.", flush=True)


if __name__ == "__main__":
    main()
