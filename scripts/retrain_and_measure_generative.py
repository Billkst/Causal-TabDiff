import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.data_module_landmark import create_dataloaders, load_and_split_data  # pyright: ignore[reportMissingImports]
from evaluation.efficiency import EfficiencyTracker  # pyright: ignore[reportMissingImports]


MODEL_CHOICES = ["tabdiff", "tabsyn", "sssd", "survtraj", "stasy", "tsdiff"]


def build_wrapper(model_name: str, seq_len: int, feature_dim: int):
    if model_name == "tabsyn":
        from baselines.tabsyn_landmark_strict import TabSynLandmarkStrictWrapper  # pyright: ignore[reportMissingImports]

        return TabSynLandmarkStrictWrapper(seq_len=seq_len, feature_dim=feature_dim)
    if model_name == "tabdiff":
        from baselines.tabdiff_landmark_strict import TabDiffLandmarkStrictWrapper  # pyright: ignore[reportMissingImports]

        return TabDiffLandmarkStrictWrapper(seq_len=seq_len, feature_dim=feature_dim)
    if model_name == "survtraj":
        from baselines.survtraj_landmark_strict import SurvTrajLandmarkWrapper  # pyright: ignore[reportMissingImports]

        return SurvTrajLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)
    if model_name == "sssd":
        from baselines.sssd_landmark_strict import SSSDLandmarkWrapper  # pyright: ignore[reportMissingImports]

        return SSSDLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)
    if model_name == "stasy":
        from baselines.stasy_landmark_v2 import STaSyLandmarkWrapper  # pyright: ignore[reportMissingImports]

        return STaSyLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)
    if model_name == "tsdiff":
        from baselines.tsdiff_landmark_wrapper import TSDiffLandmarkWrapper  # pyright: ignore[reportMissingImports]

        return TSDiffLandmarkWrapper(seq_len=seq_len, feature_dim=feature_dim)

    raise ValueError(f"Unsupported model: {model_name}")


def _module_param_count(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def collect_total_params(wrapper) -> tuple[int, int]:
    module_attrs = ["model", "vae_model", "diffusion_model", "vae"]
    seen = set()
    total_params = 0
    trainable_params = 0

    for attr in module_attrs:
        module = getattr(wrapper, attr, None)
        if isinstance(module, torch.nn.Module) and id(module) not in seen:
            seen.add(id(module))
            t, tr = _module_param_count(module)
            total_params += int(t)
            trainable_params += int(tr)

    return total_params, trainable_params


def _call_sample(wrapper, n_samples: int, device: torch.device):
    sample_out = wrapper.sample(n_samples, device)
    if not isinstance(sample_out, tuple) or len(sample_out) < 2:
        raise ValueError("sample() must return at least (X_syn, Y_syn)")

    x_syn, y_syn = sample_out[0], sample_out[1]
    if x_syn.shape[0] != n_samples or y_syn.shape[0] != n_samples:
        raise ValueError(
            f"sample() output batch mismatch: expected {n_samples}, got X={x_syn.shape[0]}, Y={y_syn.shape[0]}"
        )
    return sample_out


def measure_sampling(wrapper, tracker: EfficiencyTracker, device: torch.device) -> dict[str, float]:
    warmup_samples = 20
    timed_samples = 1000

    print(f"[Sampling] Warmup: {warmup_samples} samples", flush=True)
    with torch.no_grad():
        _call_sample(wrapper, warmup_samples, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    effective_timed_samples = timed_samples

    with torch.no_grad():
        try:
            print(f"[Sampling] Timed generation: {timed_samples} samples (batched)", flush=True)
            with tracker.track_inference(timed_samples):
                _call_sample(wrapper, timed_samples, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as err:
            effective_timed_samples = 100
            print(
                f"[Sampling] Batched generation failed ({err}). Fallback to one-by-one with {effective_timed_samples} samples.",
                flush=True,
            )
            with tracker.track_inference(effective_timed_samples):
                for _ in range(effective_timed_samples):
                    _call_sample(wrapper, 1, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0.0
    )

    latency = float(tracker.metrics.get("inference_latency_ms_per_sample", 0.0))
    throughput = float(tracker.metrics.get("throughput_samples_per_sec", 0.0))

    return {
        "generative_inference_latency_ms_per_sample": latency,
        "generative_throughput_samples_per_sec": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "timed_samples": int(effective_timed_samples),
    }


def save_checkpoint(wrapper, model_name: str, seed: int, epochs: int, ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    state_dicts: dict[str, object] = {}
    wrapper_scalars: dict[str, object] = {}
    payload: dict[str, object] = {
        "model_name": model_name,
        "seed": int(seed),
        "epochs": int(epochs),
        "wrapper_class": wrapper.__class__.__name__,
        "state_dicts": state_dicts,
        "wrapper_scalars": wrapper_scalars,
    }

    for attr in ["model", "vae_model", "diffusion_model", "vae"]:
        module = getattr(wrapper, attr, None)
        if isinstance(module, torch.nn.Module):
            state_dicts[attr] = module.state_dict()

    ema = getattr(wrapper, "ema", None)
    if ema is not None and hasattr(ema, "state_dict"):
        state_dicts["ema"] = ema.state_dict()

    for scalar_name in [
        "fitted",
        "seq_len",
        "feature_dim",
        "total_dim",
        "trajectory_len",
        "train_pos_rate",
    ]:
        if hasattr(wrapper, scalar_name):
            value = getattr(wrapper, scalar_name)
            if isinstance(value, (int, float, bool)):
                wrapper_scalars[scalar_name] = value

    torch.save(payload, str(ckpt_path))
    print(f"[Checkpoint] Saved: {ckpt_path}", flush=True)


def run_single_model(model_name: str, seed: int, epochs: int, device: torch.device):
    print("=" * 80, flush=True)
    print(f"[Start] model={model_name} | seed={seed} | epochs={epochs} | device={device}", flush=True)

    table_path = PROJECT_ROOT / "data/landmark_tables/unified_person_landmark_table.pkl"
    train_df, val_df, test_df, landmark_to_idx = load_and_split_data(str(table_path), seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, landmark_to_idx, batch_size=64, num_workers=4
    )
    del val_loader, test_loader

    first_batch = next(iter(train_loader))
    seq_len = int(first_batch["x"].shape[1])
    feature_dim = int(first_batch["x"].shape[2])

    wrapper = build_wrapper(model_name, seq_len=seq_len, feature_dim=feature_dim)
    tracker = EfficiencyTracker()

    print("[Train] Begin generative model training", flush=True)
    with tracker.track_training():
        wrapper.fit(train_loader, epochs, device)

    total_params, trainable_params = collect_total_params(wrapper)
    tracker.metrics["total_params"] = int(total_params)
    tracker.metrics["trainable_params"] = int(trainable_params)
    print(
        f"[Params] total_params={tracker.metrics['total_params']}, trainable_params={tracker.metrics['trainable_params']}",
        flush=True,
    )

    ckpt_path = PROJECT_ROOT / f"checkpoints/generative/{model_name}_seed{seed}.pt"
    save_checkpoint(wrapper, model_name=model_name, seed=seed, epochs=epochs, ckpt_path=ckpt_path)

    metrics = measure_sampling(wrapper, tracker, device)

    efficiency_payload = {
        "total_params": int(tracker.metrics.get("total_params", 0)),
        "generative_inference_latency_ms_per_sample": float(metrics["generative_inference_latency_ms_per_sample"]),
        "generative_throughput_samples_per_sec": float(metrics["generative_throughput_samples_per_sec"]),
        "peak_gpu_memory_mb": float(metrics["peak_gpu_memory_mb"]),
    }

    output_path = PROJECT_ROOT / f"outputs/fulldata_baselines/formal_runs/efficiency/generative_{model_name}_seed{seed}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_payload, f, indent=2)
    print(f"[Efficiency] Saved: {output_path}", flush=True)
    print(f"[Done] model={model_name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain TSTR generative model and measure generation latency")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES + ["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed != 42:
        raise ValueError("This script is restricted to seed=42 for generative latency measurement consistency.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] Using device={device}", flush=True)

    models: list[str] = MODEL_CHOICES if args.model == "all" else [args.model]
    for model_name in models:
        run_single_model(model_name=model_name, seed=args.seed, epochs=args.epochs, device=device)

    print("[All Done] retrain_and_measure_generative finished.", flush=True)


if __name__ == "__main__":
    main()
