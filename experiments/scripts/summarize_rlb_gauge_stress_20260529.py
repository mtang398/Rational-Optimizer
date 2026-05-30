#!/usr/bin/env python3
"""Summarize the May 29 RLB gauge-stress runs.

Raw JSONL files stay under experiments/runs/. This script writes compact
CSV/Markdown/PNG artifacts under experiments/results/ so README claims can be
reproduced without committing raw Slurm outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TASKS = [
    ("synthetic/code", "synthetic_code", "Code"),
    ("synthetic/reasoning_mix", "synthetic_reasoning_mix", "Reasoning mix"),
]

METHODS = [
    ("RLB+AdamW", "rlb_adamw"),
    ("RLB+Muon", "rlb_muon"),
    ("RLB MatrixPolicy", "matrix_policy"),
    ("RLB MatrixPolicy group-stat", "matrix_policy_groupstat"),
]

GAUGES = [0.0, 2.0]
ACTIVATION = "rlb_fused_fixed_strong_ffn"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def events(records: list[dict], name: str) -> list[dict]:
    return [row for row in records if row.get("event") == name]


def last(records: list[dict], name: str) -> dict | None:
    selected = events(records, name)
    return selected[-1] if selected else None


def mean_auc(points: list[tuple[int, float]], horizon: int) -> float | str:
    selected = [(int(step), float(value)) for step, value in points if int(step) <= horizon]
    if len(selected) < 2:
        return ""
    area = 0.0
    for (s0, y0), (s1, y1) in zip(selected, selected[1:]):
        area += (s1 - s0) * (y0 + y1) / 2.0
    denom = selected[-1][0] - selected[0][0]
    return area / denom if denom > 0 else ""


def first_below(points: list[tuple[int, float]], threshold: float) -> int | str:
    for step, value in points:
        if float(value) <= threshold:
            return int(step)
    return ""


def run_path(root: Path, safe_name: str, tag: str, gauge: float) -> Path:
    gauge_name = str(gauge).replace(".", "p")
    run_name = f"{safe_name}_{tag}_gauge{gauge_name}_20260529"
    return root / safe_name / run_name / f"{ACTIVATION}.jsonl"


def label_color(method: str) -> str:
    return {
        "RLB+AdamW": "#2f855a",
        "RLB+Muon": "#b7791f",
        "RLB MatrixPolicy": "#c53030",
        "RLB MatrixPolicy group-stat": "#1a202c",
    }.get(method, "#4a5568")


def plot_task(result_dir: Path, curve_rows: list[dict], task_safe: str, task_label: str, metric: str, ylabel: str, filename: str, logy: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, gauge in zip(axes, GAUGES):
        for method, _ in METHODS:
            selected = [
                row for row in curve_rows
                if row["task_safe"] == task_safe and row["method"] == method and float(row["gauge_log_scale"]) == gauge and row[metric] != ""
            ]
            selected.sort(key=lambda row: int(row["step"]))
            if not selected:
                continue
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                markersize=3.0,
                linewidth=1.8,
                label=method,
                color=label_color(method),
            )
        ax.set_title(f"gauge log scale {gauge:g}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{task_label} gauge stress")
    fig.tight_layout()
    fig.savefig(result_dir / filename, dpi=160)
    plt.close(fig)


def table(rows: list[dict], fields: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for title, _ in fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for _, key in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="experiments/runs/rlb_gauge_stress_20260529")
    parser.add_argument("--result-dir", default="experiments/results/rlb_gauge_stress_2026_05_29")
    args = parser.parse_args()

    root = Path(args.run_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for dataset_name, safe_name, task_label in TASKS:
        for gauge in GAUGES:
            for method, tag in METHODS:
                path = run_path(root, safe_name, tag, gauge)
                records = read_jsonl(path)
                config = last(records, "config") or {}
                final_eval = last(records, "eval") or {}
                summary = last(records, "summary") or {}
                eval_points = [(int(row["step"]), float(row["val_loss"])) for row in events(records, "eval") if "val_loss" in row]
                train_points = [(int(row["step"]), float(row["loss"])) for row in events(records, "train") if "loss" in row]
                complete = bool(final_eval and summary and int(summary.get("steps", -1)) == int(final_eval.get("step", -2)))
                row = {
                    "task": task_label,
                    "task_safe": safe_name,
                    "dataset_name": dataset_name,
                    "method": method,
                    "gauge_log_scale": gauge,
                    "complete": complete,
                    "last_eval_step": final_eval.get("step", ""),
                    "val_loss": final_eval.get("val_loss", ""),
                    "val_ppl": final_eval.get("val_ppl", ""),
                    "val_auc_mean_200": mean_auc(eval_points, 200),
                    "val_auc_mean_400": mean_auc(eval_points, 400),
                    "val_auc_mean_750": mean_auc(eval_points, 750),
                    "train_auc_mean_200": mean_auc(train_points, 200),
                    "train_auc_mean_400": mean_auc(train_points, 400),
                    "first_val_loss_le_0p2": first_below(eval_points, 0.2),
                    "first_val_loss_le_0p15": first_below(eval_points, 0.15),
                    "optimizer": config.get("optimizer", ""),
                    "activation": config.get("activation", ""),
                    "path": str(path),
                }
                summary_rows.append(row)
                for event in events(records, "eval"):
                    loss = event.get("val_loss", "")
                    eval_rows.append({
                        "task": task_label,
                        "task_safe": safe_name,
                        "method": method,
                        "gauge_log_scale": gauge,
                        "step": event.get("step", ""),
                        "val_loss": loss,
                        "val_ppl": event.get("val_ppl", ""),
                        "path": str(path),
                    })
                for event in events(records, "train"):
                    loss = event.get("loss", "")
                    train_rows.append({
                        "task": task_label,
                        "task_safe": safe_name,
                        "method": method,
                        "gauge_log_scale": gauge,
                        "step": event.get("step", ""),
                        "loss": loss,
                        "train_ppl": "" if loss == "" else math.exp(min(20.0, float(loss))),
                        "path": str(path),
                    })

    by_key = {(row["task"], row["method"], row["gauge_log_scale"]): row for row in summary_rows}
    degradation_rows: list[dict] = []
    for _, _, task_label in TASKS:
        for method, _ in METHODS:
            g0 = by_key.get((task_label, method, 0.0), {})
            g2 = by_key.get((task_label, method, 2.0), {})
            row = {"task": task_label, "method": method}
            for key in [
                "val_loss",
                "val_ppl",
                "val_auc_mean_200",
                "val_auc_mean_400",
                "val_auc_mean_750",
                "train_auc_mean_200",
                "train_auc_mean_400",
            ]:
                v0 = g0.get(key, "")
                v2 = g2.get(key, "")
                row[f"{key}_gauge0"] = v0
                row[f"{key}_gauge2"] = v2
                row[f"delta_{key}_gauge2_minus_gauge0"] = "" if v0 == "" or v2 == "" else float(v2) - float(v0)
            for key in ["first_val_loss_le_0p2", "first_val_loss_le_0p15"]:
                v0 = g0.get(key, "")
                v2 = g2.get(key, "")
                row[f"{key}_gauge0"] = v0
                row[f"{key}_gauge2"] = v2
                row[f"delta_{key}_gauge2_minus_gauge0"] = "" if v0 == "" or v2 == "" else int(v2) - int(v0)
            degradation_rows.append(row)

    summary_fields = [
        "task", "task_safe", "dataset_name", "method", "gauge_log_scale", "complete", "last_eval_step",
        "val_loss", "val_ppl", "val_auc_mean_200", "val_auc_mean_400", "val_auc_mean_750",
        "train_auc_mean_200", "train_auc_mean_400", "first_val_loss_le_0p2", "first_val_loss_le_0p15",
        "optimizer", "activation", "path",
    ]
    write_csv(result_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(result_dir / "eval_curves.csv", eval_rows, ["task", "task_safe", "method", "gauge_log_scale", "step", "val_loss", "val_ppl", "path"])
    write_csv(result_dir / "train_curves.csv", train_rows, ["task", "task_safe", "method", "gauge_log_scale", "step", "loss", "train_ppl", "path"])
    degradation_fields = list(degradation_rows[0].keys()) if degradation_rows else []
    write_csv(result_dir / "gauge_degradation.csv", degradation_rows, degradation_fields)

    for _, safe_name, task_label in TASKS:
        plot_task(result_dir, eval_rows, safe_name, task_label, "val_loss", "validation loss", f"{safe_name}_validation_loss_by_gauge.png")
        plot_task(result_dir, eval_rows, safe_name, task_label, "val_ppl", "validation PPL", f"{safe_name}_validation_ppl_by_gauge.png", logy=True)
        plot_task(result_dir, train_rows, safe_name, task_label, "loss", "training loss", f"{safe_name}_training_loss_by_gauge.png")
        plot_task(result_dir, train_rows, safe_name, task_label, "train_ppl", "training PPL", f"{safe_name}_training_ppl_by_gauge.png", logy=True)

    summary_sorted = sorted(summary_rows, key=lambda row: (row["task"], row["gauge_log_scale"], float(row["val_auc_mean_200"] or 1e9)))
    degradation_sorted = sorted(degradation_rows, key=lambda row: (row["task"], float(row.get("delta_val_auc_mean_200_gauge2_minus_gauge0", 1e9) or 1e9)))
    summary_md = [
        "# RLB Gauge Stress Summary",
        "",
        "Gauge stress applies an equivalent-function positive group rescaling at initialization.",
        "Lower AUC mean loss is better. Negative gauge deltas mean the stressed parameterization trained faster on that metric, so this run should be read as gauge sensitivity rather than a pure degradation test.",
        "",
        "## Curve Metrics",
        "",
        table(summary_sorted, [
            ("task", "task"),
            ("gauge", "gauge_log_scale"),
            ("method", "method"),
            ("val AUC mean 200", "val_auc_mean_200"),
            ("val AUC mean 400", "val_auc_mean_400"),
            ("train AUC mean 200", "train_auc_mean_200"),
            ("first val <= 0.2", "first_val_loss_le_0p2"),
            ("final loss", "val_loss"),
            ("final PPL", "val_ppl"),
        ]),
        "",
        "## Gauge Sensitivity",
        "",
        table(degradation_sorted, [
            ("task", "task"),
            ("method", "method"),
            ("delta val AUC 200", "delta_val_auc_mean_200_gauge2_minus_gauge0"),
            ("delta val AUC 400", "delta_val_auc_mean_400_gauge2_minus_gauge0"),
            ("delta train AUC 200", "delta_train_auc_mean_200_gauge2_minus_gauge0"),
            ("delta final loss", "delta_val_loss_gauge2_minus_gauge0"),
            ("delta first <= 0.2", "delta_first_val_loss_le_0p2_gauge2_minus_gauge0"),
        ]),
        "",
        "## Figures",
        "",
        "![Code validation loss](synthetic_code_validation_loss_by_gauge.png)",
        "",
        "![Code validation PPL](synthetic_code_validation_ppl_by_gauge.png)",
        "",
        "![Code training loss](synthetic_code_training_loss_by_gauge.png)",
        "",
        "![Reasoning mix validation loss](synthetic_reasoning_mix_validation_loss_by_gauge.png)",
        "",
        "![Reasoning mix validation PPL](synthetic_reasoning_mix_validation_ppl_by_gauge.png)",
        "",
        "![Reasoning mix training loss](synthetic_reasoning_mix_training_loss_by_gauge.png)",
        "",
        "Generated by `experiments/scripts/summarize_rlb_gauge_stress_20260529.py`.",
        "Raw JSONL files stay under `experiments/runs/` and are not committed.",
    ]
    (result_dir / "summary.md").write_text("\n".join(summary_md) + "\n")


if __name__ == "__main__":
    main()
