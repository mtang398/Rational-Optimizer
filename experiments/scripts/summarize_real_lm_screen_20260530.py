#!/usr/bin/env python3
"""Summarize the May 30 real-corpus LM screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

TASKS = {
    "fineweb": "FineWeb",
    "fineweb_edu": "FineWeb-Edu",
}

LABELS = {
    ("adamw", "silu"): "SiLU+AdamW",
    ("adamw", "rlb_fused_fixed_strong_ffn"): "RLB+AdamW",
    ("muon", "silu"): "SiLU+Muon",
    ("muon", "rlb_fused_fixed_strong_ffn"): "RLB+Muon",
    ("rational_matrix_policy_onpolicy", "rlb_fused_fixed_strong_ffn"): "RLB+MatrixPolicy (group-stat)",
}

ORDER = [
    "SiLU+AdamW",
    "RLB+AdamW",
    "SiLU+Muon",
    "RLB+Muon",
    "RLB+MatrixPolicy (group-stat)",
]

COLORS = {
    "SiLU+AdamW": "#3b6fb6",
    "RLB+AdamW": "#4b9b5f",
    "SiLU+Muon": "#8f8f8f",
    "RLB+Muon": "#c06f3c",
    "RLB+MatrixPolicy (group-stat)": "#b9332f",
}

STYLES = {
    "SiLU+AdamW": "-",
    "RLB+AdamW": "-",
    "SiLU+Muon": "--",
    "RLB+Muon": "--",
    "RLB+MatrixPolicy (group-stat)": "-",
}


def is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def read_jsonl(path: Path) -> dict:
    config = {}
    train = []
    evals = []
    summary = None
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "config":
                config = record
            elif event == "train":
                train.append(record)
            elif event == "eval":
                evals.append(record)
            elif event == "summary":
                summary = record
    label = LABELS.get((config.get("optimizer"), config.get("activation")))
    if label is None:
        label = f"{config.get('activation')}+{config.get('optimizer')}"
    return {"path": path, "config": config, "train": train, "eval": evals, "summary": summary, "label": label}


def collect_runs(run_root: Path) -> dict[str, list[dict]]:
    tasks = {}
    for task in TASKS:
        task_root = run_root / task
        runs = []
        if not task_root.exists():
            continue
        for path in sorted(task_root.glob("*/*.jsonl")):
            if ".incomplete_" in str(path):
                continue
            run = read_jsonl(path)
            if run["summary"] is not None:
                runs.append(run)
        tasks[task] = sorted(runs, key=lambda item: ORDER.index(item["label"]) if item["label"] in ORDER else 99)
    return tasks


def finite_series(
    records: list[dict],
    key: str,
    min_step: int = 1,
    require_eval_loss_finite: bool = False,
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for record in records:
        step = int(record["step"])
        if step < min_step:
            continue
        if require_eval_loss_finite and not is_finite(record.get("val_loss")):
            continue
        value = record.get(key)
        if is_finite(value):
            xs.append(step)
            ys.append(float(value))
    return xs, ys


def first_nonfinite_step(records: list[dict], key: str) -> int | None:
    for record in records:
        value = record.get(key)
        if not is_finite(value):
            return int(record["step"])
    return None


def trapezoid_auc(records: list[dict], key: str, max_step: int | None = None) -> float | None:
    points = []
    for record in records:
        step = int(record["step"])
        if max_step is not None and step > max_step:
            continue
        value = record.get(key)
        if is_finite(value):
            points.append((step, float(value)))
    if len(points) < 2:
        return None
    area = 0.0
    span = points[-1][0] - points[0][0]
    if span <= 0:
        return None
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / span


def final_finite(records: list[dict], key: str) -> tuple[int | None, float | None]:
    for record in reversed(records):
        value = record.get(key)
        if is_finite(value):
            return int(record["step"]), float(value)
    return None, None


def final_finite_eval(records: list[dict]) -> tuple[int | None, float | None, float | None]:
    for record in reversed(records):
        loss = record.get("val_loss")
        if is_finite(loss):
            loss_value = float(loss)
            ppl = record.get("val_ppl")
            ppl_value = float(ppl) if is_finite(ppl) else math.exp(min(loss_value, 20.0))
            return int(record["step"]), loss_value, ppl_value
    return None, None, None


def divergence_note(train_step: int | None, eval_step: int | None) -> str:
    parts = []
    if train_step is not None:
        parts.append(f"train nonfinite at step {train_step}")
    if eval_step is not None:
        parts.append(f"validation nonfinite at step {eval_step}")
    return "; ".join(parts)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_diverged(run: dict) -> bool:
    return bool(first_nonfinite_step(run["train"], "loss") or first_nonfinite_step(run["eval"], "val_loss"))


def plot_metric(
    task: str,
    runs: list[dict],
    event: str,
    key: str,
    ylabel: str,
    out_path: Path,
    min_step: int = 1,
    require_eval_loss_finite: bool = False,
    exclude_diverged_runs: bool = False,
) -> None:
    plt.figure(figsize=(8.0, 4.8))
    plotted = False
    for run in runs:
        if exclude_diverged_runs and run_diverged(run):
            continue
        records = run[event]
        xs, ys = finite_series(records, key, min_step=min_step, require_eval_loss_finite=require_eval_loss_finite)
        if not xs:
            continue
        plotted = True
        label = run["label"]
        plt.plot(
            xs,
            ys,
            label=label,
            color=COLORS.get(label),
            linestyle=STYLES.get(label, "-"),
            linewidth=2.0 if label == "RLB+MatrixPolicy (group-stat)" else 1.6,
        )
        bad_step = first_nonfinite_step(records, key)
        if bad_step is not None and min_step <= bad_step and not exclude_diverged_runs:
            plt.scatter([xs[-1]], [ys[-1]], color=COLORS.get(label), marker="x", s=48, zorder=4)
            plt.text(xs[-1], ys[-1], f"  diverged @ {bad_step}", fontsize=8, va="center")
    title = f"{TASKS[task]} {ylabel}"
    if min_step > 1:
        title += f" (step >= {min_step})"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="experiments/runs/real_lm_screen_20260530")
    parser.add_argument("--result-dir", default="experiments/results/real_lm_screen_2026_05_30")
    parser.add_argument("--zoom-min-step", type=int, default=1000)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    tasks = collect_runs(run_root)

    summary_rows = []
    train_rows = []
    eval_rows = []

    for task, runs in tasks.items():
        for run in runs:
            config = run["config"]
            label = run["label"]
            final_step, final_loss, final_ppl = final_finite_eval(run["eval"])
            train_nan = first_nonfinite_step(run["train"], "loss")
            eval_nan = first_nonfinite_step(run["eval"], "val_loss")
            completed_steps = bool(run["summary"] and int(run["summary"].get("steps", -1)) == 3050)
            status = "diverged" if train_nan or eval_nan else ("complete" if completed_steps else "incomplete")
            summary_rows.append(
                {
                    "task": task,
                    "method": label,
                    "activation": config.get("activation"),
                    "optimizer": config.get("optimizer"),
                    "complete": completed_steps,
                    "status": status,
                    "train_diverged_step": train_nan or "",
                    "eval_diverged_step": eval_nan or "",
                    "final_finite_step": final_step or "",
                    "final_val_loss": f"{final_loss:.6f}" if final_loss is not None else "",
                    "final_val_ppl": f"{final_ppl:.6f}" if final_ppl is not None else "",
                    "val_loss_auc_1000": f"{trapezoid_auc(run['eval'], 'val_loss', 1000):.6f}" if trapezoid_auc(run['eval'], 'val_loss', 1000) is not None else "",
                    "val_loss_auc_2000": f"{trapezoid_auc(run['eval'], 'val_loss', 2000):.6f}" if trapezoid_auc(run['eval'], 'val_loss', 2000) is not None else "",
                    "val_loss_auc_full": f"{trapezoid_auc(run['eval'], 'val_loss'):.6f}" if trapezoid_auc(run['eval'], 'val_loss') is not None else "",
                    "mean_seconds_per_step": f"{float(run['summary'].get('mean_seconds_per_step', 0.0)):.6f}" if run["summary"] else "",
                }
            )
            for record in run["train"]:
                train_rows.append(
                    {
                        "task": task,
                        "method": label,
                        "step": record.get("step"),
                        "loss": record.get("loss"),
                        "lr": record.get("lr"),
                        "seconds_per_step": record.get("seconds_per_step"),
                    }
                )
            for record in run["eval"]:
                eval_rows.append(
                    {
                        "task": task,
                        "method": label,
                        "step": record.get("step"),
                        "val_loss": record.get("val_loss"),
                        "val_ppl": record.get("val_ppl"),
                    }
                )

        plot_metric(task, runs, "eval", "val_loss", "validation loss", result_dir / f"{task}_validation_loss.png")
        plot_metric(
            task,
            runs,
            "eval",
            "val_ppl",
            "validation PPL",
            result_dir / f"{task}_validation_ppl.png",
            require_eval_loss_finite=True,
            exclude_diverged_runs=True,
        )
        plot_metric(
            task,
            runs,
            "eval",
            "val_loss",
            "validation loss",
            result_dir / f"{task}_validation_loss_zoom_step{args.zoom_min_step}.png",
            min_step=args.zoom_min_step,
        )
        plot_metric(
            task,
            runs,
            "eval",
            "val_ppl",
            "validation PPL",
            result_dir / f"{task}_validation_ppl_zoom_step{args.zoom_min_step}.png",
            min_step=args.zoom_min_step,
            require_eval_loss_finite=True,
            exclude_diverged_runs=True,
        )
        plot_metric(task, runs, "train", "loss", "training loss", result_dir / f"{task}_training_loss.png")

    write_csv(
        result_dir / "summary.csv",
        summary_rows,
        [
            "task",
            "method",
            "activation",
            "optimizer",
            "complete",
            "status",
            "train_diverged_step",
            "eval_diverged_step",
            "final_finite_step",
            "final_val_loss",
            "final_val_ppl",
            "val_loss_auc_1000",
            "val_loss_auc_2000",
            "val_loss_auc_full",
            "mean_seconds_per_step",
        ],
    )
    write_csv(result_dir / "train_curves.csv", train_rows, ["task", "method", "step", "loss", "lr", "seconds_per_step"])
    write_csv(result_dir / "eval_curves.csv", eval_rows, ["task", "method", "step", "val_loss", "val_ppl"])

    lines = ["# Real LM Screen, 2026-05-30", ""]
    lines.append("All rows use the same 100M-token training budget, 4M-token heldout slice after a 110M-token stream offset, and the same base LR schedule.")
    lines.append("PPL plots omit divergent/nonfinite runs; zoomed validation plots start at step 1000.")
    lines.append("")
    for task in TASKS:
        lines.append(f"## {TASKS[task]}")
        lines.append("")
        lines.append("| method | last finite validation loss | last finite PPL | AUC <= 1000 | AUC <= 2000 | note |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in [row for row in summary_rows if row["task"] == task]:
            note = divergence_note(
                int(row["train_diverged_step"]) if row["train_diverged_step"] else None,
                int(row["eval_diverged_step"]) if row["eval_diverged_step"] else None,
            )
            if not note:
                note = row["status"]
            ppl = f"{float(row['final_val_ppl']):.2f}" if row["final_val_ppl"] else ""
            lines.append(
                f"| {row['method']} | {row['final_val_loss']} | {ppl} | "
                f"{row['val_loss_auc_1000']} | {row['val_loss_auc_2000']} | {note} |"
            )
        lines.append("")
    (result_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n")


if __name__ == "__main__":
    main()
