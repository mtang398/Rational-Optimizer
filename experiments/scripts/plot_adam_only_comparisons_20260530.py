#!/usr/bin/env python3
"""Plot AdamW-control comparisons without generic Muon rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TASKS = [
    ("synthetic_code", "Code"),
    ("synthetic_symbolic", "Symbolic"),
    ("synthetic_reasoning_mix", "Reasoning mix"),
]
METHODS = [
    "SiLU/SwiGLU+AdamW",
    "RLB+AdamW",
    "RLB MatrixPolicy",
    "RLB MatrixPolicy group-stat",
]
COLORS = {
    "SiLU/SwiGLU+AdamW": "#2b6cb0",
    "RLB+AdamW": "#2f855a",
    "RLB MatrixPolicy": "#c53030",
    "RLB MatrixPolicy group-stat": "#1a202c",
}
LINESTYLES = {
    "SiLU/SwiGLU+AdamW": "-",
    "RLB+AdamW": "-",
    "RLB MatrixPolicy": "-",
    "RLB MatrixPolicy group-stat": "--",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_grid(rows: list[dict], result_dir: Path, y_key: str, ylabel: str, filename: str, title: str, logy: bool = False) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    fig, axes = plt.subplots(1, len(TASKS), figsize=(15, 4.6), sharey=False)
    for ax, (task_safe, task_label) in zip(axes, TASKS):
        for method in METHODS:
            selected = [
                row for row in rows
                if row.get("task_safe") == task_safe and row.get("method") == method and row.get(y_key, "") != ""
            ]
            selected.sort(key=lambda row: int(row["step"]))
            if not selected:
                continue
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row[y_key]) for row in selected],
                marker="o",
                markersize=2.6,
                linewidth=1.9,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                label=method,
            )
        ax.set_title(task_label)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(result_dir / filename, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="experiments/results/synthetic_dense_curves_2026_05_29")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    eval_rows = read_csv(result_dir / "eval_curves.csv")
    train_rows = read_csv(result_dir / "train_curves.csv")

    plot_grid(
        eval_rows,
        result_dir,
        "val_loss",
        "validation loss",
        "adam_only_validation_loss.png",
        "AdamW controls vs MatrixPolicy: validation loss",
    )
    plot_grid(
        eval_rows,
        result_dir,
        "val_ppl",
        "validation PPL",
        "adam_only_validation_ppl.png",
        "AdamW controls vs MatrixPolicy: validation PPL",
        logy=True,
    )
    plot_grid(
        train_rows,
        result_dir,
        "loss",
        "training loss",
        "adam_only_training_loss.png",
        "AdamW controls vs MatrixPolicy: training loss",
    )
    plot_grid(
        train_rows,
        result_dir,
        "train_ppl",
        "training PPL",
        "adam_only_training_ppl.png",
        "AdamW controls vs MatrixPolicy: training PPL",
        logy=True,
    )


if __name__ == "__main__":
    main()
