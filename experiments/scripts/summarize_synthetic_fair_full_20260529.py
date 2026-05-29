#!/usr/bin/env python3
"""Summarize the May 29 synthetic fair rerun.

The final artifact combines two clean Slurm runs:

* Code and Symbolic from experiments/runs/synthetic_fair_full_20260529
* Reasoning mix from experiments/runs/synthetic_fair_reasoning_mix_20260529

Raw JSONL files stay under experiments/runs/. This script writes compact
CSV/Markdown/PNG artifacts under experiments/results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TASKS = [
    ("synthetic/code", "synthetic_code", "Code"),
    ("synthetic/symbolic", "synthetic_symbolic", "Symbolic"),
    ("synthetic/reasoning_mix", "synthetic_reasoning_mix", "Reasoning mix"),
]

METHODS = [
    ("SiLU/SwiGLU+AdamW", "adamw_controls", "silu"),
    ("RLB+AdamW", "adamw_controls", "rlb_fused_fixed_strong_ffn"),
    ("SiLU/SwiGLU+Muon", "muon_controls", "silu"),
    ("RLB+Muon", "muon_controls", "rlb_fused_fixed_strong_ffn"),
    ("RLB MatrixPolicy", "matrix_policy", "rlb_fused_fixed_strong_ffn"),
    ("RLB MatrixPolicy group-stat", "matrix_policy_groupstat", "rlb_fused_fixed_strong_ffn"),
]

DEFAULT_TASK_LAYOUT = {
    "synthetic_code": ("experiments/runs/synthetic_fair_full_20260529", "20260529_fair_full"),
    "synthetic_symbolic": ("experiments/runs/synthetic_fair_full_20260529", "20260529_fair_full"),
    "synthetic_reasoning_mix": (
        "experiments/runs/synthetic_fair_reasoning_mix_20260529",
        "20260529_reasoning_rerun",
    ),
}


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


def last_event(rows: list[dict], event: str) -> dict | None:
    selected = [row for row in rows if row.get("event") == event]
    return selected[-1] if selected else None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(result_dir: Path, curve_rows: list[dict], y_key: str, ylabel: str, suffix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional on headless nodes.
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    colors = {
        "SiLU/SwiGLU+AdamW": "#2b6cb0",
        "RLB+AdamW": "#2f855a",
        "SiLU/SwiGLU+Muon": "#805ad5",
        "RLB+Muon": "#b7791f",
        "RLB MatrixPolicy": "#c53030",
        "RLB MatrixPolicy group-stat": "#1a202c",
    }
    for _, safe_name, task_label in TASKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        has_curve = False
        for method, _, _ in METHODS:
            selected = [
                row for row in curve_rows
                if row["task_safe"] == safe_name and row["method"] == method and row[y_key] != ""
            ]
            if not selected:
                continue
            has_curve = True
            selected.sort(key=lambda row: int(row["step"]))
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row[y_key]) for row in selected],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=method,
                color=colors.get(method),
            )
        if not has_curve:
            plt.close(fig)
            continue
        ax.set_title(f"{task_label} synthetic fair rerun")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(result_dir / f"{safe_name}_{suffix}.png", dpi=160)
        plt.close(fig)


def plot_final_bars(result_dir: Path, summary_rows: list[dict], y_key: str, ylabel: str, filename: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - plotting is optional on headless nodes.
        print(f"Skipping final bar plot because matplotlib/numpy is unavailable: {exc}")
        return

    task_labels = [task_label for _, _, task_label in TASKS]
    method_labels = [method for method, _, _ in METHODS]
    values_by_method = []
    for method in method_labels:
        values = []
        for task_label in task_labels:
            row = next(
                (
                    item for item in summary_rows
                    if item["task"] == task_label and item["method"] == method and item[y_key] != ""
                ),
                None,
            )
            values.append(float(row[y_key]) if row else np.nan)
        values_by_method.append(values)

    x = np.arange(len(task_labels))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2b6cb0", "#2f855a", "#805ad5", "#b7791f", "#c53030", "#1a202c"]
    offsets = (np.arange(len(method_labels)) - (len(method_labels) - 1) / 2) * width
    for method, values, offset, color in zip(method_labels, values_by_method, offsets, colors):
        ax.bar(x + offset, values, width, label=method, color=color)

    ax.set_title("Synthetic fair rerun final validation metrics")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(result_dir / filename, dpi=160)
    plt.close(fig)


def markdown_table(rows: list[dict]) -> str:
    headers = ["task", "method", "step", "loss", "PPL", "delta loss vs SiLU+AdamW"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        loss = "" if row["val_loss"] == "" else f"{float(row['val_loss']):.6f}"
        ppl = "" if row["val_ppl"] == "" else f"{float(row['val_ppl']):.4f}"
        delta = "" if row["delta_loss_vs_silu_adamw"] == "" else f"{float(row['delta_loss_vs_silu_adamw']):+.6f}"
        lines.append(f"| {row['task']} | {row['method']} | {row['last_eval_step']} | {loss} | {ppl} | {delta} |")
    return "\n".join(lines)



def build_step_diagnostics(curve_rows: list[dict], loss_key: str, ppl_key: str) -> tuple[list[dict], list[dict]]:
    by_key = {
        (row["task"], row["method"], int(row["step"])): row
        for row in curve_rows
        if row.get(loss_key, "") != "" and row.get(ppl_key, "") != ""
    }
    diagnostic_rows: list[dict] = []
    auc_rows: list[dict] = []
    for _, _, task_label in TASKS:
        steps = sorted({step for task, _, step in by_key if task == task_label and step > 1})
        for step in steps:
            baseline = by_key.get((task_label, "SiLU/SwiGLU+AdamW", step))
            candidates = [
                by_key[(task_label, method, step)]
                for method, _, _ in METHODS
                if (task_label, method, step) in by_key
            ]
            if baseline is None or not candidates:
                continue
            best = min(candidates, key=lambda row: float(row[loss_key]))
            diagnostic_rows.append({
                "task": task_label,
                "step": step,
                "best_method": best["method"],
                "best_loss": best[loss_key],
                "best_ppl": best[ppl_key],
                "silu_adamw_loss": baseline[loss_key],
                "silu_adamw_ppl": baseline[ppl_key],
                "delta_loss_vs_silu_adamw": float(best[loss_key]) - float(baseline[loss_key]),
                "delta_ppl_vs_silu_adamw": float(best[ppl_key]) - float(baseline[ppl_key]),
            })

        for method, _, _ in METHODS:
            points = [
                (step, float(by_key[(task_label, method, step)][loss_key]))
                for step in steps
                if (task_label, method, step) in by_key
            ]
            if len(points) < 2:
                continue
            auc = sum(
                (points[i + 1][0] - points[i][0]) * (points[i][1] + points[i + 1][1]) / 2
                for i in range(len(points) - 1)
            )
            auc_rows.append({
                "task": task_label,
                "method": method,
                "start_step": points[0][0],
                "end_step": points[-1][0],
                "auc_loss": auc,
            })
    auc_rows.sort(key=lambda row: (row["task"], float(row["auc_loss"])))
    return diagnostic_rows, auc_rows


def build_curve_diagnostics(eval_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    return build_step_diagnostics(eval_rows, "val_loss", "val_ppl")


def build_training_diagnostics(train_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    return build_step_diagnostics(train_rows, "loss", "train_ppl")


def select_report_rows(rows: list[dict]) -> list[dict]:
    preferred_steps = {10, 25, 50, 100, 200, 250, 500, 750, 1000, 1250}
    selected: list[dict] = []
    for _, _, task_label in TASKS:
        task_rows = sorted([row for row in rows if row["task"] == task_label], key=lambda row: int(row["step"]))
        preferred = [row for row in task_rows if int(row["step"]) in preferred_steps]
        if preferred:
            selected.extend(preferred)
        else:
            selected.extend(task_rows[:6])
            selected.extend(row for row in task_rows[-3:] if row not in task_rows[:6])
    return selected


def markdown_curve_table(rows: list[dict]) -> str:
    headers = ["task", "step", "best curve row", "loss", "PPL", "delta loss vs SiLU+AdamW", "delta PPL"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["task"],
                str(row["step"]),
                row["best_method"],
                f"{float(row['best_loss']):.6f}",
                f"{float(row['best_ppl']):.4f}",
                f"{float(row['delta_loss_vs_silu_adamw']):+.6f}",
                f"{float(row['delta_ppl_vs_silu_adamw']):+.4f}",
            ])
            + " |"
        )
    return "\n".join(lines)


def markdown_auc_table(rows: list[dict]) -> str:
    headers = ["task", "best AUC row", "step range", "AUC loss"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: |"]
    for _, _, task_label in TASKS:
        task_rows = [row for row in rows if row["task"] == task_label]
        if not task_rows:
            continue
        best = min(task_rows, key=lambda row: float(row["auc_loss"]))
        lines.append(
            f"| {task_label} | {best['method']} | {best['start_step']}-{best['end_step']} | "
            f"{float(best['auc_loss']):.2f} |"
        )
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default=None,
        help="Use one run root for every task. Must be supplied together with --suffix.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Use one run suffix for every task. Must be supplied together with --run-root.",
    )
    parser.add_argument(
        "--task-source",
        action="append",
        default=[],
        metavar="TASK_SAFE,RUN_ROOT,SUFFIX",
        help=(
            "Override one task source. TASK_SAFE is synthetic_code, synthetic_symbolic, "
            "or synthetic_reasoning_mix. Can be repeated."
        ),
    )
    parser.add_argument("--result-dir", default="experiments/results/synthetic_fair_full_2026_05_29")
    args = parser.parse_args()

    if (args.run_root is None) != (args.suffix is None):
        parser.error("--run-root and --suffix must be supplied together")

    if args.run_root is not None:
        task_layout = {safe_name: (args.run_root, args.suffix) for _, safe_name, _ in TASKS}
    else:
        task_layout = dict(DEFAULT_TASK_LAYOUT)

    for item in args.task_source:
        try:
            safe_name, run_root, suffix = item.split(",", 2)
        except ValueError as exc:
            raise SystemExit("--task-source must have the form TASK_SAFE,RUN_ROOT,SUFFIX") from exc
        if safe_name not in task_layout:
            raise SystemExit(f"unknown task safe name for --task-source: {safe_name}")
        task_layout[safe_name] = (run_root, suffix)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for dataset_name, safe_name, task_label in TASKS:
        task_run_root, task_suffix = task_layout[safe_name]
        run_root = Path(task_run_root)
        baselines: dict[str, float] = {}
        task_rows: list[dict] = []
        for method, run_tag, activation in METHODS:
            run_name = f"{safe_name}_{run_tag}_{task_suffix}"
            path = run_root / safe_name / run_name / f"{activation}.jsonl"
            records = read_jsonl(path)
            config = last_event(records, "config") or {}
            final_eval = last_event(records, "eval")
            summary = last_event(records, "summary") or {}
            complete = bool(final_eval and summary and int(summary.get("steps", -1)) == int(final_eval.get("step", -2)))
            val_loss = "" if final_eval is None else final_eval.get("val_loss", "")
            val_ppl = "" if final_eval is None else final_eval.get("val_ppl", "")
            row = {
                "task": task_label,
                "task_safe": safe_name,
                "dataset_name": dataset_name,
                "method": method,
                "run_name": run_name,
                "activation": activation,
                "optimizer": config.get("optimizer", ""),
                "complete": complete,
                "last_eval_step": "" if final_eval is None else final_eval.get("step", ""),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "delta_loss_vs_silu_adamw": "",
                "delta_ppl_vs_silu_adamw": "",
                "path": str(path),
            }
            if method == "SiLU/SwiGLU+AdamW" and val_loss != "":
                baselines["loss"] = float(val_loss)
                baselines["ppl"] = float(val_ppl)
            for event in records:
                if event.get("event") == "eval":
                    eval_rows.append({
                        "task": task_label,
                        "task_safe": safe_name,
                        "method": method,
                        "step": event.get("step", ""),
                        "val_loss": event.get("val_loss", ""),
                        "val_ppl": event.get("val_ppl", ""),
                        "path": str(path),
                    })
                elif event.get("event") == "train":
                    loss = event.get("loss", "")
                    train_rows.append({
                        "task": task_label,
                        "task_safe": safe_name,
                        "method": method,
                        "step": event.get("step", ""),
                        "loss": loss,
                        "train_ppl": "" if loss == "" else math.exp(min(20.0, float(loss))),
                        "path": str(path),
                    })
            task_rows.append(row)
        for row in task_rows:
            if baselines and row["val_loss"] != "":
                row["delta_loss_vs_silu_adamw"] = float(row["val_loss"]) - baselines["loss"]
                row["delta_ppl_vs_silu_adamw"] = float(row["val_ppl"]) - baselines["ppl"]
            summary_rows.append(row)

    summary_fields = [
        "task", "task_safe", "dataset_name", "method", "complete", "last_eval_step", "val_loss", "val_ppl",
        "delta_loss_vs_silu_adamw", "delta_ppl_vs_silu_adamw", "optimizer", "activation", "run_name", "path",
    ]
    write_csv(result_dir / "summary.csv", summary_rows, summary_fields)
    write_csv(result_dir / "eval_curves.csv", eval_rows, ["task", "task_safe", "method", "step", "val_loss", "val_ppl", "path"])
    write_csv(result_dir / "train_curves.csv", train_rows, ["task", "task_safe", "method", "step", "loss", "train_ppl", "path"])
    plot_curves(result_dir, eval_rows, "val_loss", "validation loss", "validation_loss")
    plot_curves(result_dir, eval_rows, "val_ppl", "validation PPL", "validation_ppl")
    plot_curves(result_dir, train_rows, "loss", "training loss", "training_loss")
    plot_curves(result_dir, train_rows, "train_ppl", "training PPL", "training_ppl")
    plot_final_bars(result_dir, summary_rows, "val_loss", "final validation loss", "final_loss_by_task.png")
    plot_final_bars(result_dir, summary_rows, "val_ppl", "final validation PPL", "final_ppl_by_task.png")

    diagnostic_fields = [
        "task", "step", "best_method", "best_loss", "best_ppl", "silu_adamw_loss", "silu_adamw_ppl",
        "delta_loss_vs_silu_adamw", "delta_ppl_vs_silu_adamw",
    ]
    auc_fields = ["task", "method", "start_step", "end_step", "auc_loss"]
    curve_diagnostic_rows, auc_rows = build_curve_diagnostics(eval_rows)
    train_diagnostic_rows, train_auc_rows = build_training_diagnostics(train_rows)
    report_curve_rows = select_report_rows(curve_diagnostic_rows)
    report_train_rows = select_report_rows(train_diagnostic_rows)
    write_csv(result_dir / "curve_diagnostics.csv", curve_diagnostic_rows, diagnostic_fields)
    write_csv(result_dir / "eval_auc_loss.csv", auc_rows, auc_fields)
    write_csv(result_dir / "auc250_loss.csv", auc_rows, auc_fields)
    write_csv(result_dir / "train_curve_diagnostics.csv", train_diagnostic_rows, diagnostic_fields)
    write_csv(result_dir / "train_auc_loss.csv", train_auc_rows, auc_fields)
    curve_md = [
        "# Synthetic Curve Diagnostics",
        "",
        "The sparse historical run is useful only as a provisional curve smoke test.",
        "Dense reruns should use frequent training logs and validation evals before making optimizer claims.",
        "",
        "## Validation Curve",
        "",
        markdown_curve_table(report_curve_rows),
        "",
        markdown_auc_table(auc_rows),
        "",
        "## Training Curve",
        "",
        markdown_curve_table(report_train_rows),
        "",
        markdown_auc_table(train_auc_rows),
    ]
    (result_dir / "curve_diagnostics.md").write_text("\n".join(curve_md) + "\n")

    complete_count = sum(1 for row in summary_rows if row["complete"])
    total_count = len(summary_rows)
    source_lines = [
        f"* `{safe_name}`: `{task_layout[safe_name][0]}`, suffix `{task_layout[safe_name][1]}`"
        for _, safe_name, _ in TASKS
    ]
    summary_md = [
        "# Synthetic Fair Full Rerun",
        "",
        f"Completed rows: {complete_count}/{total_count}",
        "",
        "## Curve Signal",
        "",
        "This artifact includes both validation and training curve diagnostics.",
        "Sparse runs are provisional; dense runs should use frequent `LOG_INTERVAL` and `EVAL_INTERVAL` settings.",
        "",
        "### Validation Curve",
        "",
        markdown_curve_table(report_curve_rows),
        "",
        markdown_auc_table(auc_rows),
        "",
        "### Training Curve",
        "",
        markdown_curve_table(report_train_rows),
        "",
        markdown_auc_table(train_auc_rows),
        "",
        "## Final Rows",
        "",
        "These final rows are secondary for the synthetic tasks because the losses are near the floor.",
        "",
        markdown_table(summary_rows),
        "",
        "Sources:",
        "",
        *source_lines,
        "",
        "Generated by `experiments/scripts/summarize_synthetic_fair_full_20260529.py`.",
        "Raw JSONL files stay under `experiments/runs/` and are not committed.",
    ]
    (result_dir / "summary.md").write_text("\n".join(summary_md) + "\n")


if __name__ == "__main__":
    main()
