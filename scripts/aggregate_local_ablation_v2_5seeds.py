#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_ranking_metrics, compute_threshold_metrics

SEEDS = [42, 52, 62, 72, 82]
DEFAULT_VARIANTS = [
    "Full_Model_v2",
    "wo_Discriminator_v2",
    "wo_Oversampling_v2",
    "wo_Time_Attention_v2",
]
METRICS = [
    "best_epoch",
    "best_val_auprc",
    "val_auprc",
    "val_auroc",
    "test_auprc",
    "test_auroc",
    "optimal_threshold",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_balanced_accuracy",
    "test_mcc",
    "total_params",
    "train_wall_clock_sec",
    "peak_gpu_memory_mb",
    "inference_latency_ms",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/local_ablation_v2_5seeds")
    p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    p.add_argument("--old-summary", type=Path, default=PROJECT_ROOT / "outputs/submission_v3_5seeds/ablation_layer2/ablation_5seeds_summary.csv")
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data.files:
        return None
    value = data[key]
    return value.item() if getattr(value, "shape", None) == () else value.tolist()


def eff_value(eff: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in eff and eff[key] is not None:
            return eff[key]
    return None


def collect(output_root: Path, variants: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for variant in variants:
        for seed in SEEDS:
            npz_path = output_root / variant / f"{variant}_seed{seed}.npz"
            eff_path = output_root / variant / f"{variant}_seed{seed}_efficiency.json"
            if not npz_path.exists() or not eff_path.exists():
                if not npz_path.exists():
                    missing.append(rel(npz_path))
                if not eff_path.exists():
                    missing.append(rel(eff_path))
                continue
            data = np.load(npz_path)
            eff = load_json(eff_path)
            test_y_true = np.asarray(data["test_y_true"]).ravel()
            test_y_pred = np.asarray(data["test_y_pred"]).ravel()
            val_y_true = np.asarray(data["val_y_true"]).ravel()
            val_y_pred = np.asarray(data["val_y_pred"]).ravel()
            threshold = float(npz_scalar(data, "optimal_threshold") or 0.5)
            test_rank = compute_ranking_metrics(test_y_true, test_y_pred)
            val_rank = compute_ranking_metrics(val_y_true, val_y_pred)
            test_thr = compute_threshold_metrics(test_y_true, (test_y_pred >= threshold).astype(int))
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "npz_path": rel(npz_path),
                    "efficiency_path": rel(eff_path),
                    "command": eff.get("command", ""),
                    "best_epoch": finite_float(npz_scalar(data, "best_epoch")),
                    "best_val_auprc": finite_float(npz_scalar(data, "best_val_auprc")),
                    "val_auprc": finite_float(val_rank.get("auprc")),
                    "val_auroc": finite_float(val_rank.get("auroc")),
                    "test_auprc": finite_float(test_rank.get("auprc")),
                    "test_auroc": finite_float(test_rank.get("auroc")),
                    "optimal_threshold": threshold,
                    "test_f1": finite_float(test_thr.get("f1")),
                    "test_precision": finite_float(test_thr.get("precision")),
                    "test_recall": finite_float(test_thr.get("recall")),
                    "test_specificity": finite_float(test_thr.get("specificity")),
                    "test_balanced_accuracy": finite_float(test_thr.get("balanced_accuracy")),
                    "test_mcc": finite_float(test_thr.get("mcc")),
                    "total_params": finite_float(eff_value(eff, "total_params", "num_parameters_total")),
                    "train_wall_clock_sec": finite_float(eff_value(eff, "train_wall_clock_sec", "total_training_wall_clock_sec")),
                    "peak_gpu_memory_mb": finite_float(eff_value(eff, "peak_gpu_memory_mb", "gpu_memory_mb_peak")),
                    "inference_latency_ms": finite_float(eff_value(eff, "inference_latency_ms_per_sample")),
                }
            )
    columns = [
        "variant", "seed", "npz_path", "efficiency_path", "command",
        *METRICS,
    ]
    return pd.DataFrame(rows, columns=columns), missing


def summarize(detail: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    full = detail[detail["variant"] == "Full_Model_v2"]
    full_mean = float(full["test_auprc"].mean()) if not full.empty else float("nan")
    rows: list[dict[str, Any]] = []
    for variant in variants:
        sub = detail[detail["variant"] == variant]
        row: dict[str, Any] = {
            "variant": variant,
            "n_seeds": int(sub["seed"].nunique()) if not sub.empty else 0,
            "seeds": ",".join(str(x) for x in sorted(sub["seed"].astype(int).tolist())) if not sub.empty else "",
        }
        for metric in METRICS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna() if metric in sub else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else None
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        row["delta_test_auprc_vs_full_v2_mean"] = row["test_auprc_mean"] - full_mean if full_mean == full_mean and row["test_auprc_mean"] is not None else None
        row["relative_delta_test_auprc_vs_full_v2_pct"] = 100.0 * row["delta_test_auprc_vs_full_v2_mean"] / full_mean if full_mean and row["delta_test_auprc_vs_full_v2_mean"] is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(mean: Any, std: Any) -> str:
    if mean is None or pd.isna(mean):
        return "NA"
    return f"{float(mean):.4f} +/- {float(0.0 if std is None or pd.isna(std) else std):.4f}"


def write_md(summary: pd.DataFrame, output_root: Path, old_summary: Path) -> None:
    lines = [
        "# Local Ablation v2 5-Seed Summary",
        "",
        "Protocol: batch_size=94739, num_workers=0, use_cached_dataset=true, local single GPU serial execution.",
        "",
        "| Variant | Seeds | Test AUPRC | Test AUROC | Test F1 | Test MCC | Delta AUPRC vs Full_Model_v2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        delta = r["delta_test_auprc_vs_full_v2_mean"]
        delta_s = "NA" if delta is None or pd.isna(delta) else f"{float(delta):.4f}"
        lines.append(
            f"| {r['variant']} | {int(r['n_seeds'])} | {fmt(r['test_auprc_mean'], r['test_auprc_std'])} | "
            f"{fmt(r['test_auroc_mean'], r['test_auroc_std'])} | {fmt(r['test_f1_mean'], r['test_f1_std'])} | "
            f"{fmt(r['test_mcc_mean'], r['test_mcc_std'])} | {delta_s} |"
        )
    if old_summary.exists():
        lines.extend(["", f"Old ablation summary for comparison: `{rel(old_summary)}`"])
    (output_root / "local_ablation_v2_5seeds_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(summary: pd.DataFrame, old_summary: Path, output_root: Path) -> None:
    if not old_summary.exists():
        return
    old = pd.read_csv(old_summary)
    full_v2 = summary.loc[summary["variant"] == "Full_Model_v2", "test_auprc_mean"]
    full_old = old.loc[old["variant"] == "Full_Model", "test_auprc_mean"] if "variant" in old else pd.Series(dtype=float)
    rows = []
    if len(full_v2) and len(full_old) and full_v2.iloc[0] is not None and not pd.isna(full_v2.iloc[0]):
        rows.append(
            {
                "comparison": "Full_Model_v2_vs_current_Full_Model",
                "new_test_auprc_mean": float(full_v2.iloc[0]),
                "old_test_auprc_mean": float(full_old.iloc[0]),
                "delta_new_minus_old": float(full_v2.iloc[0] - full_old.iloc[0]),
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "local_ablation_v2_vs_current_comparison.csv", index=False)


def main() -> int:
    args = parse_args()
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    detail, missing = collect(args.output_root, variants)
    if missing and not args.allow_incomplete:
        print(f"Incomplete local ablation v2 outputs: {len(missing)} missing files", flush=True)
        for item in missing[:50]:
            print(f"  - {item}", flush=True)
        return 2
    detail_path = args.output_root / "local_ablation_v2_5seeds_detail.csv"
    summary_path = args.output_root / "local_ablation_v2_5seeds_summary.csv"
    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = summarize(detail, variants)
    summary.to_csv(summary_path, index=False)
    write_md(summary, args.output_root, args.old_summary)
    write_comparison(summary, args.old_summary, args.output_root)
    print(f"Wrote {rel(detail_path)}", flush=True)
    print(f"Wrote {rel(summary_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
