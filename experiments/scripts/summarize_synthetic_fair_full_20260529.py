#!/usr/bin/env python3
"""Summarize the May 29 synthetic fair rerun.

This parser expects the layout produced by
experiments/scripts/run_synthetic_fair_full_20260529.sh and writes compact
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def markdown_table(rows: list[dict]) -> str:
    headers = ["task", "method", "step", "loss", "PPL", "delta loss vs SiLU+AdamW"]
    lines = ["| " + " | ".join(headers) + " |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        loss = "" if row["val_loss"] == "" else f"{float(row['val_loss']):.6f}"
        ppl = "" if row["val_ppl"] == "" else f"{float(row['val_ppl']):.4f}"
        delta = "" if row["delta_loss_vs_silu_adamw"] == "" else f"{float(row['delta_loss_vs_silu_adamw']):+.6f}"
        lines.append(f"| {row['task']} | {row['method']} | {row['last_eval_step']} | {loss} | {ppl} | {delta} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="experiments/runs/synthetic_fair_full_20260529")
    parser.add_argument("--suffix", default="20260529_fair_full")
    parser.add_argument("--result-dir", default="experiments/results/synthetic_fair_full_2026_05_29")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for dataset_name, safe_name, task_label in TASKS:
        baselines: dict[str, float] = {}
        task_rows: list[dict] = []
        for method, run_tag, activation in METHODS:
            run_name = f"{safe_name}_{run_tag}_{args.suffix}"
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

    complete_count = sum(1 for row in summary_rows if row["complete"])
    total_count = len(summary_rows)
    summary_md = [
        "# Synthetic Fair Full Rerun",
        "",
        f"Completed rows: {complete_count}/{total_count}",
        "",
        markdown_table(summary_rows),
        "",
        "Generated from `experiments/scripts/run_synthetic_fair_full_20260529.sh`.",
        "Raw JSONL files stay under `experiments/runs/` and are not committed.",
    ]
    (result_dir / "summary.md").write_text("\n".join(summary_md) + "\n")


if __name__ == "__main__":
    main()
