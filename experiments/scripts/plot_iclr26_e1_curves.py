#!/usr/bin/env python3
"""Generate dense E1 mean/std curve figures from ICLR26 manifest logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROW_RE = re.compile(
    r"^=== row=(?P<row>\d+); id=(?P<row_id>[^;]+); phase=(?P<phase>[^;]+); "
    r"dataset=(?P<dataset>[^;]+); method=(?P<method>[^;]+); seed=(?P<seed>\d+);"
)

DATASETS = [
    ("dclm", "DCLM"),
    ("fineweb_edu", "FineWeb-Edu"),
    ("fineweb", "FineWeb"),
    ("dolma_sample", "Dolma-sample"),
]

METHODS = [
    ("rlb_matrixpolicy_original", "MatrixPolicy", "#111111", "-", 2.3, 0.16),
    ("rlb_adamw", "RLB+AdamW", "#1f77b4", "-", 1.45, 0.10),
    ("silu_adamw", "SiLU+AdamW", "#1f77b4", "--", 1.45, 0.08),
    ("rlb_lion", "RLB+Lion", "#2ca02c", "-", 1.45, 0.10),
    ("silu_lion", "SiLU+Lion", "#2ca02c", "--", 1.45, 0.08),
    ("rlb_soap", "RLB+SOAP", "#9467bd", "-", 1.35, 0.08),
    ("silu_soap", "SiLU+SOAP", "#9467bd", "--", 1.35, 0.08),
    ("rlb_muon", "RLB+Muon", "#ff7f0e", "-", 1.35, 0.08),
    ("silu_muon", "SiLU+Muon", "#ff7f0e", "--", 1.35, 0.08),
    ("rlb_schedulefree", "RLB+ScheduleFree", "#8c564b", "-", 1.25, 0.06),
    ("silu_schedulefree", "SiLU+ScheduleFree", "#8c564b", "--", 1.25, 0.06),
    ("rlb_came", "RLB+CAME", "#17becf", "-", 1.25, 0.06),
    ("silu_came", "SiLU+CAME", "#17becf", "--", 1.25, 0.06),
]

START_STEP = 500
END_STEP = 3050
XTICKS = [500, 1000, 1500, 2000, 2500, 3000]


def parse_logs(log_dir: Path):
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"train": {}, "eval": {}})))
    for log_path in sorted(log_dir.glob("iclr26-main-*.out")):
        current = None
        with log_path.open("r", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                match = ROW_RE.match(line)
                if match:
                    current = {
                        "dataset": match.group("dataset"),
                        "method": match.group("method"),
                        "seed": int(match.group("seed")),
                    }
                    continue
                if current is None or not line.startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = record.get("event")
                step = record.get("step")
                if event not in {"train", "eval"} or not isinstance(step, int):
                    continue
                if step < START_STEP or step > END_STEP:
                    continue
                dataset = current["dataset"]
                method = current["method"]
                seed = current["seed"]
                if event == "train":
                    value = record.get("loss")
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        curves[dataset][method][seed]["train"][step] = float(value)
                elif event == "eval":
                    loss = record.get("val_loss")
                    ppl = record.get("val_ppl")
                    if isinstance(loss, (int, float)) and math.isfinite(loss):
                        curves[dataset][method][seed]["eval"].setdefault(step, {})["val_loss"] = float(loss)
                    if isinstance(ppl, (int, float)) and math.isfinite(ppl):
                        curves[dataset][method][seed]["eval"].setdefault(step, {})["val_ppl"] = float(ppl)
    return curves


def aggregate(curves, dataset: str, method: str, metric: str):
    seed_data = curves.get(dataset, {}).get(method, {})
    by_step = defaultdict(list)
    for seed, events in seed_data.items():
        if metric == "train_loss":
            source = events["train"]
            for step, value in source.items():
                by_step[step].append(value)
        else:
            key = "val_loss" if metric == "val_loss" else "val_ppl"
            for step, values in events["eval"].items():
                value = values.get(key)
                if value is not None and math.isfinite(value):
                    by_step[step].append(value)
    steps, means, stds = [], [], []
    for step in sorted(by_step):
        vals = np.asarray(by_step[step], dtype=float)
        # Plot only true multi-seed aggregates for finished cells.
        if vals.size < 3:
            continue
        steps.append(step)
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)))
    return np.asarray(steps), np.asarray(means), np.asarray(stds)


def plot_dataset(curves, dataset: str, dataset_label: str, metric: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=140)
    plotted = 0
    for method, label, color, linestyle, linewidth, alpha in METHODS:
        steps, means, stds = aggregate(curves, dataset, method, metric)
        if steps.size == 0:
            continue
        ax.plot(
            steps,
            means,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
        )
        ax.fill_between(steps, means - stds, means + stds, color=color, alpha=alpha, linewidth=0)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False

    metric_label = {
        "val_loss": "validation loss",
        "val_ppl": "validation PPL",
        "train_loss": "training loss",
    }[metric]
    ax.set_title(f"{dataset_label} E1 {metric_label}, mean +/- std", fontsize=12, pad=9)
    ax.set_xlabel("step")
    ax.set_ylabel(metric_label)
    ax.set_xlim(START_STEP, END_STEP)
    ax.set_xticks(XTICKS)
    ax.grid(True, color="#d8d8d8", linewidth=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(
        title="Curve",
        loc="upper right",
        frameon=True,
        framealpha=0.90,
        fontsize=6.8,
        title_fontsize=7.4,
        ncol=2,
        borderpad=0.6,
        labelspacing=0.35,
        columnspacing=0.8,
        handlelength=2.5,
    )
    legend.get_frame().set_edgecolor("#bbbbbb")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("experiments/runs/logs"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/iclr26_e1_figures"))
    args = parser.parse_args()

    curves = parse_logs(args.log_dir)
    suffixes = {
        "val_loss": "core_validation_loss_mean_std.svg",
        "val_ppl": "core_validation_ppl_mean_std.svg",
        "train_loss": "core_training_loss_mean_std.svg",
    }
    made = []
    for dataset, label in DATASETS:
        for metric, suffix in suffixes.items():
            out_path = args.out_dir / f"{dataset}_{suffix}"
            if plot_dataset(curves, dataset, label, metric, out_path):
                made.append(out_path)
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
