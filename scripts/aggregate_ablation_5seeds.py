#!/usr/bin/env python3
"""Aggregate the official 4-variant Layer2 ablation without rerunning Full_Model."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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
ABLATION_VARIANTS = ["wo_Discriminator", "wo_Dual_Attention", "wo_Oversampling"]
SUMMARY_VARIANTS = ["Full_Model", *ABLATION_VARIANTS]
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
FINAL_OUTPUTS = [
    "ablation_5seeds_detail.csv",
    "ablation_5seeds_summary.csv",
    "ablation_5seeds_summary.md",
    "ablation_5seeds_table_source.csv",
    "ablation_5seeds_table_source.tex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/submission_v3_5seeds/ablation_layer2",
    )
    parser.add_argument(
        "--full-detail-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs/submission_v3/layer2_ours_5seed_detail.csv",
    )
    parser.add_argument(
        "--full-efficiency-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/efficiency",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def md5(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def collect_full_model(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not args.full_detail_csv.exists():
        manifest["missing"].append(rel(args.full_detail_csv))
        return []

    df = pd.read_csv(args.full_detail_csv)
    rows = []
    seen = set()
    for _, row in df.iterrows():
        seed = int(row["seed"])
        if seed not in SEEDS:
            continue
        seen.add(seed)
        eff_path = args.full_efficiency_root / f"CausalTabDiff_layer2_seed{seed}_efficiency.json"
        eff = load_json(eff_path) if eff_path.exists() else {}
        if not eff_path.exists():
            manifest["missing"].append(rel(eff_path))
        rows.append(
            {
                "variant": "Full_Model",
                "seed": seed,
                "source": "reused_official_layer2_ours_5seed",
                "source_path": rel(args.full_detail_csv),
                "npz_path": eff.get("output_file", ""),
                "efficiency_path": rel(eff_path),
                "best_epoch": finite_float(row.get("best_epoch")),
                "best_val_auprc": finite_float(row.get("best_val_auprc")),
                "val_auprc": finite_float(row.get("best_val_auprc")),
                "val_auroc": None,
                "test_auprc": finite_float(row.get("test_auprc")),
                "test_auroc": finite_float(row.get("test_auroc")),
                "optimal_threshold": finite_float(row.get("optimal_threshold")),
                "test_f1": finite_float(row.get("test_f1")),
                "test_precision": finite_float(row.get("test_precision")),
                "test_recall": finite_float(row.get("test_recall")),
                "test_specificity": finite_float(row.get("test_specificity")),
                "test_balanced_accuracy": finite_float(row.get("test_balanced_accuracy")),
                "test_mcc": finite_float(row.get("test_mcc")),
                "total_params": finite_float(eff_value(eff, "total_params", "num_parameters_total")),
                "train_wall_clock_sec": finite_float(
                    eff_value(eff, "train_wall_clock_sec", "total_training_wall_clock_sec")
                ),
                "peak_gpu_memory_mb": finite_float(eff_value(eff, "peak_gpu_memory_mb", "gpu_memory_mb_peak")),
                "inference_latency_ms": finite_float(eff_value(eff, "inference_latency_ms_per_sample")),
                "md5": md5(eff_path),
            }
        )
    for seed in SEEDS:
        if seed not in seen:
            manifest["missing"].append(f"Full_Model official detail seed {seed}")
    return rows


def collect_ablation_variants(args: argparse.Namespace, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant in ABLATION_VARIANTS:
        for seed in SEEDS:
            npz_path = args.output_root / variant / f"{variant}_seed{seed}.npz"
            eff_path = args.output_root / variant / f"{variant}_seed{seed}_efficiency.json"
            if not npz_path.exists() or not eff_path.exists():
                if not npz_path.exists():
                    manifest["missing"].append(rel(npz_path))
                if not eff_path.exists():
                    manifest["missing"].append(rel(eff_path))
                continue
            try:
                data = np.load(npz_path)
                eff = load_json(eff_path)
                test_y_true = np.asarray(data["test_y_true"]).ravel()
                test_y_pred = np.asarray(data["test_y_pred"]).ravel()
                val_y_true = np.asarray(data["val_y_true"]).ravel()
                val_y_pred = np.asarray(data["val_y_pred"]).ravel()
            except Exception as exc:
                manifest["invalid"].append({"path": rel(npz_path), "error": repr(exc)})
                continue

            threshold = float(npz_scalar(data, "optimal_threshold") or 0.5)
            test_rank = compute_ranking_metrics(test_y_true, test_y_pred)
            val_rank = compute_ranking_metrics(val_y_true, val_y_pred)
            test_thr = compute_threshold_metrics(test_y_true, (test_y_pred >= threshold).astype(int))
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "source": "new_ablation_5seed_supplement",
                    "source_path": rel(npz_path),
                    "npz_path": rel(npz_path),
                    "efficiency_path": rel(eff_path),
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
                    "train_wall_clock_sec": finite_float(
                        eff_value(eff, "train_wall_clock_sec", "total_training_wall_clock_sec")
                    ),
                    "peak_gpu_memory_mb": finite_float(eff_value(eff, "peak_gpu_memory_mb", "gpu_memory_mb_peak")),
                    "inference_latency_ms": finite_float(eff_value(eff, "inference_latency_ms_per_sample")),
                    "md5": md5(npz_path),
                }
            )
    return rows


def summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    full_mean = float(detail_df.loc[detail_df["variant"] == "Full_Model", "test_auprc"].mean())
    out = []
    for variant in SUMMARY_VARIANTS:
        sub = detail_df[detail_df["variant"] == variant]
        row: dict[str, Any] = {
            "variant": variant,
            "n_seeds": int(sub["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(sub["seed"].astype(int).tolist())),
        }
        for metric in METRICS:
            values = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["delta_test_auprc_vs_full_mean"] = row["test_auprc_mean"] - full_mean
        row["relative_delta_test_auprc_vs_full_pct"] = (
            100.0 * row["delta_test_auprc_vs_full_mean"] / full_mean if full_mean else None
        )
        out.append(row)
    return pd.DataFrame(out)


def fmt(mean: Any, std: Any, digits: int = 4) -> str:
    if mean is None or pd.isna(mean):
        return "NA"
    std_val = 0.0 if std is None or pd.isna(std) else float(std)
    return f"{float(mean):.{digits}f} +/- {std_val:.{digits}f}"


def write_summary_md(summary_df: pd.DataFrame, output_root: Path, complete_new_runs: int) -> None:
    lines = [
        "# Layer2 Official 4-Variant Ablation 5-Seed Summary",
        "",
        "Full_Model uses the existing official Layer2 Ours 5-seed result and is not rerun.",
        f"Supplemental ablation runs complete: {complete_new_runs}/15.",
        "",
        "| Variant | Seeds | Test AUPRC | Test AUROC | Test F1 | Test MCC | Delta AUPRC vs Full | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_df.iterrows():
        source = "reused official" if row["variant"] == "Full_Model" else "new supplement"
        lines.append(
            "| {variant} | {n_seeds} | {auprc} | {auroc} | {f1} | {mcc} | {delta:.4f} | {source} |".format(
                variant=row["variant"],
                n_seeds=int(row["n_seeds"]),
                auprc=fmt(row["test_auprc_mean"], row["test_auprc_std"]),
                auroc=fmt(row["test_auroc_mean"], row["test_auroc_std"]),
                f1=fmt(row["test_f1_mean"], row["test_f1_std"]),
                mcc=fmt(row["test_mcc_mean"], row["test_mcc_std"]),
                delta=float(row["delta_test_auprc_vs_full_mean"]),
                source=source,
            )
        )
    (output_root / "ablation_5seeds_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_sources(summary_df: pd.DataFrame, output_root: Path) -> None:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "variant": row["variant"],
                "test_auprc": fmt(row["test_auprc_mean"], row["test_auprc_std"]),
                "test_auroc": fmt(row["test_auroc_mean"], row["test_auroc_std"]),
                "test_f1": fmt(row["test_f1_mean"], row["test_f1_std"]),
                "test_mcc": fmt(row["test_mcc_mean"], row["test_mcc_std"]),
                "delta_test_auprc_vs_full_mean": f"{float(row['delta_test_auprc_vs_full_mean']):.4f}",
                "source": "reused official Full_Model" if row["variant"] == "Full_Model" else "new 5-seed supplement",
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "ablation_5seeds_table_source.csv", index=False)

    tex = [
        "% Source only. Do not include automatically in main.tex.",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & Test AUPRC & Test AUROC & Test F1 & Test MCC & $\Delta$ AUPRC \\",
        r"\midrule",
    ]
    for item in rows:
        tex.append(
            r"{variant} & {auprc} & {auroc} & {f1} & {mcc} & {delta} \\".format(
                variant=item["variant"].replace("_", r"\_"),
                auprc=item["test_auprc"],
                auroc=item["test_auroc"],
                f1=item["test_f1"],
                mcc=item["test_mcc"],
                delta=item["delta_test_auprc_vs_full_mean"],
            )
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "ablation_5seeds_table_source.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def remove_final_outputs(output_root: Path) -> None:
    for name in FINAL_OUTPUTS:
        path = output_root / name
        if path.exists():
            path.unlink()


def write_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    registry = {
        "name": "official_4variant_ablation_5seed_supplement",
        "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "output_root": rel(output_root),
        "full_model_policy": "reuse outputs/submission_v3/layer2_ours_5seed_detail.csv; do not rerun Full_Model",
        "supplement_variants": ABLATION_VARIANTS,
        "seeds": SEEDS,
        "complete_new_runs": manifest["complete_new_runs"],
        "expected_new_runs": 15,
        "is_complete": manifest["is_complete"],
        "files": manifest["files"],
        "missing": manifest["missing"],
        "invalid": manifest["invalid"],
    }
    with (output_root / "REGISTRY.json").open("w", encoding="utf-8") as fp:
        json.dump(registry, fp, indent=2, ensure_ascii=False)

    lines = [
        "# Ablation 5-Seed Manifest",
        "",
        f"- Complete supplemental runs: {manifest['complete_new_runs']}/15",
        "- Full_Model policy: reuse existing official 5-seed result; do not rerun",
        f"- Complete final outputs generated: {manifest['is_complete']}",
        "",
        "## Expected Supplemental Runs",
        "",
    ]
    completed = set(manifest["completed_keys"])
    for variant in ABLATION_VARIANTS:
        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            status = "ok" if key in completed else "missing"
            lines.append(f"- {status}: {key}")
    if manifest["missing"]:
        lines.extend(["", "## Missing", ""])
        lines.extend(f"- {item}" for item in manifest["missing"])
    if manifest["invalid"]:
        lines.extend(["", "## Invalid", ""])
        lines.extend(f"- {item}" for item in manifest["invalid"])
    (output_root / "MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "complete_new_runs": 0,
        "completed_keys": [],
        "is_complete": False,
        "missing": [],
        "invalid": [],
        "files": {},
    }

    full_rows = collect_full_model(args, manifest)
    ablation_rows = collect_ablation_variants(args, manifest)
    manifest["complete_new_runs"] = len(ablation_rows)
    manifest["completed_keys"] = [f"{row['variant']}_seed{row['seed']}" for row in ablation_rows]
    manifest["is_complete"] = bool(len(full_rows) == 5 and len(ablation_rows) == 15 and not manifest["invalid"])
    write_manifest(args.output_root, manifest)

    if args.check_only:
        print(f"Completeness: supplemental {len(ablation_rows)}/15, Full_Model official {len(full_rows)}/5", flush=True)
        if manifest["missing"]:
            print("Missing:", flush=True)
            for item in manifest["missing"]:
                print(f"  - {item}", flush=True)
        return 0 if manifest["is_complete"] else 2

    if not manifest["is_complete"]:
        remove_final_outputs(args.output_root)
        print(
            "Incomplete ablation supplement; final ablation tables were not written. "
            f"Supplemental runs complete: {len(ablation_rows)}/15.",
            flush=True,
        )
        return 2

    detail_df = pd.DataFrame(full_rows + ablation_rows)
    detail_df["variant"] = pd.Categorical(detail_df["variant"], SUMMARY_VARIANTS, ordered=True)
    detail_df = detail_df.sort_values(["variant", "seed"]).reset_index(drop=True)
    detail_path = args.output_root / "ablation_5seeds_detail.csv"
    detail_df.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary_df = summarize(detail_df)
    summary_path = args.output_root / "ablation_5seeds_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    write_summary_md(summary_df, args.output_root, len(ablation_rows))
    write_table_sources(summary_df, args.output_root)

    for name in FINAL_OUTPUTS + ["REGISTRY.json", "MANIFEST.md"]:
        out_path = args.output_root / name
        manifest["files"][name] = {"path": rel(out_path), "md5": md5(out_path)}
    write_manifest(args.output_root, manifest)

    print(f"Wrote {rel(detail_path)}", flush=True)
    print(f"Wrote {rel(summary_path)}", flush=True)
    print(f"Wrote {rel(args.output_root / 'ablation_5seeds_summary.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
