import math
import statistics
from pathlib import Path


LOWER_BETTER = {"ATE_Bias", "Wasserstein", "CMD", "AvgInfer(ms/sample)"}
HIGHER_BETTER = {"TSTR_AUC", "TSTR_F1"}


def parse_value(cell: str):
    cell = cell.strip()
    if cell == "N/A" or cell == "":
        return float("nan"), float("nan")
    if "±" in cell:
        left, right = cell.split("±", 1)
        return float(left.strip()), float(right.strip())
    val = float(cell)
    return val, float("nan")


def read_markdown_table(md_path: Path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    header = [x.strip() for x in table_lines[0].strip("|").split("|")]

    rows = []
    for ln in table_lines[2:]:
        cols = [x.strip() for x in ln.strip("|").split("|")]
        if len(cols) != len(header):
            continue
        row = dict(zip(header, cols))
        rows.append(row)
    return rows


def median_without_nan(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return statistics.median(vals)


def main():
    md_path = Path("markdown_report.md")
    if not md_path.exists():
        raise FileNotFoundError("markdown_report.md not found")

    rows = read_markdown_table(md_path)
    metrics = [k for k in rows[0].keys() if k != "Model"]

    parsed = []
    for row in rows:
        model = row["Model"]
        item = {"Model": model}
        for m in metrics:
            mean_v, std_v = parse_value(row[m])
            item[f"{m}__mean"] = mean_v
            item[f"{m}__std"] = std_v
        parsed.append(item)

    metric_medians = {}
    for m in metrics:
        metric_medians[m] = median_without_nan([r[f"{m}__mean"] for r in parsed])

    alerts = []
    for r in parsed:
        model = r["Model"]
        for m in metrics:
            mean_v = r[f"{m}__mean"]
            std_v = r[f"{m}__std"]
            med = metric_medians[m]
            if not math.isfinite(mean_v) or not math.isfinite(med):
                continue

            if m in LOWER_BETTER:
                if med > 0 and mean_v > med * 3 and (mean_v - med) > 0.05:
                    alerts.append(("high", model, m, mean_v, med, "明显偏高"))
            elif m in HIGHER_BETTER:
                if mean_v < med * 0.7 and (med - mean_v) > 0.05:
                    alerts.append(("low", model, m, mean_v, med, "明显偏低"))

            if math.isfinite(std_v) and mean_v != 0 and abs(std_v / mean_v) > 0.6:
                alerts.append(("unstable", model, m, mean_v, std_v, "方差占比偏大"))

        auc = r.get("TSTR_AUC__mean", float("nan"))
        f1 = r.get("TSTR_F1__mean", float("nan"))
        if math.isfinite(auc) and math.isfinite(f1):
            if auc <= 0.55 and f1 <= 0.08:
                alerts.append(("low", model, "TSTR_combo", auc, f1, "分类效能接近随机或极弱"))
            if auc >= 0.98 and f1 >= 0.85:
                alerts.append(("suspicious", model, "TSTR_combo", auc, f1, "分类效能异常高，建议排查泄漏"))

    print("=== Baseline Audit Summary ===")
    print(f"models={len(parsed)}, metrics={len(metrics)}")
    print("\n=== Metric Medians ===")
    for m in metrics:
        print(f"- {m}: {metric_medians[m]:.6f}")

    print("\n=== Alerts ===")
    if not alerts:
        print("No strong anomalies detected.")
    else:
        for a in alerts:
            kind, model, metric, v1, v2, reason = a
            print(f"[{kind}] {model} | {metric} | v1={v1:.6f}, ref={v2:.6f} | {reason}")

    needs_rerun = any(a[0] in {"high", "low", "suspicious"} for a in alerts)
    print("\n=== Rerun Recommendation ===")
    if needs_rerun:
        print("建议重跑至少异常模型（或先做定向复核后重跑）。")
    else:
        print("当前不需要重跑。")


if __name__ == "__main__":
    main()
