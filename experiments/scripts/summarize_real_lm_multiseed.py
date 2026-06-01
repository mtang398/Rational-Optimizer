#!/usr/bin/env python3
"""Summarize real-LM MatrixPolicy runs across seeds.

This script accepts a mix of raw JSONL run roots and prior summary CSV files.
The latter is useful for older runs where only paper-ready summaries were kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TASK_NAMES = {
    "fineweb": "FineWeb",
    "fineweb_edu": "FineWeb-Edu",
    "dclm": "DCLM",
    "dolma_sample": "Dolma sample",
}

CONTROL_METHODS = {
    "SiLU+AdamW",
    "RLB+AdamW",
    "SiLU+Muon",
    "RLB+Muon",
}

METHOD_ORDER = {
    "SiLU+AdamW": 0,
    "RLB+AdamW": 1,
    "SiLU+Muon": 2,
    "RLB+Muon": 3,
    "RLB+MatrixPolicy": 4,
    "RLB+MatrixPolicy (group-stat)": 5,
}


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def fmt(value: float | None, places: int = 6) -> str:
    return "" if value is None else f"{value:.{places}f}"


def is_groupstat_matrix_policy(config: dict[str, Any], path: Path) -> bool:
    if "groupstat" in str(path):
        return True
    group_keys = (
        "rational_matrix_policy_group_gain_strength",
        "rational_matrix_policy_group_pressure_strength",
        "rational_matrix_policy_group_activity_damping",
    )
    return any((finite_float(config.get(key)) or 0.0) != 0.0 for key in group_keys)


def label_run(config: dict[str, Any], path: Path) -> str:
    activation = config.get("activation")
    optimizer = config.get("optimizer")
    if activation == "silu" and optimizer == "adamw":
        return "SiLU+AdamW"
    if activation == "silu" and optimizer == "muon":
        return "SiLU+Muon"
    if activation == "rlb_fused_fixed_strong_ffn" and optimizer == "adamw":
        return "RLB+AdamW"
    if activation == "rlb_fused_fixed_strong_ffn" and optimizer == "muon":
        return "RLB+Muon"
    if activation == "rlb_fused_fixed_strong_ffn" and optimizer == "rational_matrix_policy_onpolicy":
        if is_groupstat_matrix_policy(config, path):
            return "RLB+MatrixPolicy (group-stat)"
        return "RLB+MatrixPolicy"
    return f"{activation}+{optimizer}"


def final_finite_eval(records: list[dict[str, Any]]) -> tuple[int | None, float | None, float | None]:
    for record in reversed(records):
        loss = finite_float(record.get("val_loss"))
        if loss is None:
            continue
        ppl = finite_float(record.get("val_ppl"))
        if ppl is None:
            ppl = math.exp(min(loss, 20.0))
        return finite_int(record.get("step")), loss, ppl
    return None, None, None


def first_nonfinite_step(records: list[dict[str, Any]], key: str) -> int | None:
    for record in records:
        if finite_float(record.get(key)) is None:
            return finite_int(record.get("step"))
    return None


def trapezoid_auc(records: list[dict[str, Any]], key: str, max_step: int | None = None) -> float | None:
    points: list[tuple[int, float]] = []
    for record in records:
        step = finite_int(record.get("step"))
        value = finite_float(record.get(key))
        if step is None or value is None:
            continue
        if max_step is not None and step > max_step:
            continue
        points.append((step, value))
    if len(points) < 2:
        return None
    span = points[-1][0] - points[0][0]
    if span <= 0:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += 0.5 * (y0 + y1) * (x1 - x0)
    return area / span


def read_jsonl(path: Path) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    try:
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
    except FileNotFoundError:
        return None
    if not config:
        return None
    task = infer_task(path, config)
    seed = finite_int(config.get("seed"))
    final_step, final_loss, final_ppl = final_finite_eval(evals)
    train_nan = first_nonfinite_step(train, "loss")
    eval_nan = first_nonfinite_step(evals, "val_loss")
    configured_steps = finite_int(config.get("steps"))
    summary_steps = finite_int(summary.get("steps")) if summary else None
    complete = configured_steps is not None and summary_steps == configured_steps
    status = "diverged" if train_nan or eval_nan else ("complete" if complete else "running")
    return {
        "task": task,
        "seed": seed,
        "method": label_run(config, path),
        "activation": config.get("activation"),
        "optimizer": config.get("optimizer"),
        "complete": complete,
        "status": status,
        "train_diverged_step": train_nan,
        "eval_diverged_step": eval_nan,
        "final_finite_step": final_step,
        "final_val_loss": final_loss,
        "final_val_ppl": final_ppl,
        "val_loss_auc_1000": trapezoid_auc(evals, "val_loss", 1000),
        "val_loss_auc_2000": trapezoid_auc(evals, "val_loss", 2000),
        "val_loss_auc_full": trapezoid_auc(evals, "val_loss"),
        "mean_seconds_per_step": finite_float(summary.get("mean_seconds_per_step")) if summary else None,
        "source": str(path),
    }


def infer_task(path: Path, config: dict[str, Any]) -> str:
    dataset = str(config.get("dataset", "")).lower()
    if "fineweb-edu" in dataset:
        return "fineweb_edu"
    if "fineweb" in dataset:
        return "fineweb"
    if "dclm" in dataset:
        return "dclm"
    if "dolma" in dataset:
        return "dolma_sample"
    for part in path.parts:
        if part in TASK_NAMES:
            return part
    return "unknown"


def read_jsonl_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.glob("*/*/*.jsonl*")):
            row = read_jsonl(path)
            if row is not None:
                rows.append(row)
    return rows


def read_baseline_summary(path: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "task": row.get("task"),
                    "seed": seed,
                    "method": row.get("method"),
                    "activation": row.get("activation"),
                    "optimizer": row.get("optimizer"),
                    "complete": str(row.get("complete", "")).lower() == "true",
                    "status": row.get("status") or "unknown",
                    "train_diverged_step": finite_int(row.get("train_diverged_step")),
                    "eval_diverged_step": finite_int(row.get("eval_diverged_step")),
                    "final_finite_step": finite_int(row.get("final_finite_step")),
                    "final_val_loss": finite_float(row.get("final_val_loss")),
                    "final_val_ppl": finite_float(row.get("final_val_ppl")),
                    "val_loss_auc_1000": finite_float(row.get("val_loss_auc_1000")),
                    "val_loss_auc_2000": finite_float(row.get("val_loss_auc_2000")),
                    "val_loss_auc_full": finite_float(row.get("val_loss_auc_full")),
                    "mean_seconds_per_step": finite_float(row.get("mean_seconds_per_step")),
                    "source": str(path),
                }
            )
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the most informative row for each task/seed/method."""
    best: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        if seed is None:
            continue
        key = (str(row.get("task")), int(seed), str(row.get("method")))
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        row_score = int(row.get("complete", False)), row.get("final_finite_step") or -1
        current_score = int(current.get("complete", False)), current.get("final_finite_step") or -1
        if row_score >= current_score:
            best[key] = row
    return sorted(best.values(), key=row_sort_key)


def row_sort_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("task")),
        int(row.get("seed") or -1),
        METHOD_ORDER.get(str(row.get("method")), 99),
        str(row.get("method")),
    )


def add_paired_gaps(rows: list[dict[str, Any]]) -> None:
    by_task_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("seed") is None:
            continue
        by_task_seed[(str(row.get("task")), int(row["seed"]))].append(row)

    for group_rows in by_task_seed.values():
        by_method = {row["method"]: row for row in group_rows}
        silu = by_method.get("SiLU+AdamW")
        silu_loss = finite_float(silu.get("final_val_loss")) if silu else None
        silu_ppl = finite_float(silu.get("final_val_ppl")) if silu else None
        control_losses = [
            finite_float(row.get("final_val_loss"))
            for row in group_rows
            if row.get("method") in CONTROL_METHODS and row.get("status") != "running"
        ]
        control_losses = [value for value in control_losses if value is not None]
        best_control_loss = min(control_losses) if control_losses else None
        for row in group_rows:
            if row.get("status") == "running":
                row["gap_loss_vs_silu_adamw"] = None
                row["gap_ppl_vs_silu_adamw"] = None
                row["gap_loss_vs_best_control"] = None
                continue
            loss = finite_float(row.get("final_val_loss"))
            ppl = finite_float(row.get("final_val_ppl"))
            row["gap_loss_vs_silu_adamw"] = silu_loss - loss if silu_loss is not None and loss is not None else None
            row["gap_ppl_vs_silu_adamw"] = silu_ppl - ppl if silu_ppl is not None and ppl is not None else None
            row["gap_loss_vs_best_control"] = (
                best_control_loss - loss if best_control_loss is not None and loss is not None else None
            )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "running":
            continue
        grouped[(str(row.get("task")), str(row.get("method")))].append(row)

    out: list[dict[str, Any]] = []
    for (task, method), group_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], METHOD_ORDER.get(item[0][1], 99), item[0][1])
    ):
        def values(key: str) -> list[float]:
            return [value for value in (finite_float(row.get(key)) for row in group_rows) if value is not None]

        loss_mean, loss_std = mean_std(values("final_val_loss"))
        ppl_mean, ppl_std = mean_std(values("final_val_ppl"))
        auc1_mean, auc1_std = mean_std(values("val_loss_auc_1000"))
        auc2_mean, auc2_std = mean_std(values("val_loss_auc_2000"))
        aucf_mean, aucf_std = mean_std(values("val_loss_auc_full"))
        silu_gap_mean, silu_gap_std = mean_std(values("gap_loss_vs_silu_adamw"))
        best_gap_mean, best_gap_std = mean_std(values("gap_loss_vs_best_control"))
        out.append(
            {
                "task": task,
                "method": method,
                "n": len(group_rows),
                "complete_n": sum(1 for row in group_rows if row.get("complete")),
                "diverged_n": sum(1 for row in group_rows if row.get("status") == "diverged"),
                "mean_final_val_loss": loss_mean,
                "std_final_val_loss": loss_std,
                "mean_final_val_ppl": ppl_mean,
                "std_final_val_ppl": ppl_std,
                "mean_val_loss_auc_1000": auc1_mean,
                "std_val_loss_auc_1000": auc1_std,
                "mean_val_loss_auc_2000": auc2_mean,
                "std_val_loss_auc_2000": auc2_std,
                "mean_val_loss_auc_full": aucf_mean,
                "std_val_loss_auc_full": aucf_std,
                "mean_gap_loss_vs_silu_adamw": silu_gap_mean,
                "std_gap_loss_vs_silu_adamw": silu_gap_std,
                "mean_gap_loss_vs_best_control": best_gap_mean,
                "std_gap_loss_vs_best_control": best_gap_std,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    formatted[field] = fmt(value)
                elif value is None:
                    formatted[field] = ""
                else:
                    formatted[field] = value
            writer.writerow(formatted)


def write_markdown(path: Path, per_seed: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    lines = ["# Real-LM Multi-Seed Summary", ""]
    lines.append("Positive gaps mean the method has lower validation loss than the comparison row.")
    lines.append("")
    for task in sorted({str(row.get("task")) for row in aggregate_rows}):
        lines.append(f"## {TASK_NAMES.get(task, task)}")
        lines.append("")
        lines.append(
            "| method | n | div | mean loss | std | mean PPL | gap vs SiLU+AdamW | gap vs best control |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in [item for item in aggregate_rows if item["task"] == task]:
            lines.append(
                f"| {row['method']} | {row['n']} | {row['diverged_n']} | "
                f"{fmt(row['mean_final_val_loss'])} | {fmt(row['std_final_val_loss'])} | "
                f"{fmt(row['mean_final_val_ppl'], 2)} | "
                f"{fmt(row['mean_gap_loss_vs_silu_adamw'])} | "
                f"{fmt(row['mean_gap_loss_vs_best_control'])} |"
            )
        lines.append("")
        lines.append("| seed | method | status | final step | val loss | PPL | gap vs SiLU+AdamW |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for row in [item for item in per_seed if item["task"] == task]:
            lines.append(
                f"| {row['seed']} | {row['method']} | {row['status']} | "
                f"{row.get('final_finite_step') or ''} | {fmt(row.get('final_val_loss'))} | "
                f"{fmt(row.get('final_val_ppl'), 2)} | {fmt(row.get('gap_loss_vs_silu_adamw'))} |"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=Path, default=[])
    parser.add_argument("--baseline-summary-csv", action="append", type=Path, default=[])
    parser.add_argument("--baseline-seed", action="append", type=int, default=[])
    parser.add_argument("--result-dir", type=Path, default=Path("experiments/results/real_lm_multiseed_2026_05_31"))
    args = parser.parse_args()

    if args.baseline_summary_csv and len(args.baseline_summary_csv) != len(args.baseline_seed):
        raise SystemExit("--baseline-summary-csv and --baseline-seed must be provided in matching counts")
    if not args.run_root and not args.baseline_summary_csv:
        raise SystemExit("provide at least one --run-root or --baseline-summary-csv input")

    rows = read_jsonl_roots(args.run_root)
    for path, seed in zip(args.baseline_summary_csv, args.baseline_seed):
        rows.extend(read_baseline_summary(path, seed))

    per_seed = dedupe_rows(rows)
    add_paired_gaps(per_seed)
    aggregate_rows = aggregate(per_seed)

    per_seed_fields = [
        "task",
        "seed",
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
        "gap_loss_vs_silu_adamw",
        "gap_ppl_vs_silu_adamw",
        "gap_loss_vs_best_control",
        "mean_seconds_per_step",
        "source",
    ]
    aggregate_fields = [
        "task",
        "method",
        "n",
        "complete_n",
        "diverged_n",
        "mean_final_val_loss",
        "std_final_val_loss",
        "mean_final_val_ppl",
        "std_final_val_ppl",
        "mean_val_loss_auc_1000",
        "std_val_loss_auc_1000",
        "mean_val_loss_auc_2000",
        "std_val_loss_auc_2000",
        "mean_val_loss_auc_full",
        "std_val_loss_auc_full",
        "mean_gap_loss_vs_silu_adamw",
        "std_gap_loss_vs_silu_adamw",
        "mean_gap_loss_vs_best_control",
        "std_gap_loss_vs_best_control",
    ]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.result_dir / "per_seed_summary.csv", per_seed, per_seed_fields)
    write_csv(args.result_dir / "aggregate_summary.csv", aggregate_rows, aggregate_fields)
    write_markdown(args.result_dir / "summary.md", per_seed, aggregate_rows)


if __name__ == "__main__":
    main()
