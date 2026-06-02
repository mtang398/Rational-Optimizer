#!/usr/bin/env python3
"""Summarize Phase A optimizer HPO JSONL traces."""

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
    "unknown": "unknown",
}

HYPER_KEYS = (
    "optimizer_lr",
    "optimizer_min_lr",
    "optimizer_weight_decay",
    "optimizer_beta1",
    "optimizer_beta2",
    "factored_min_dim",
    "factored_clip_threshold",
    "ademamix_alpha",
    "ademamix_beta3",
    "schedule_free_beta1",
    "schedule_free_warmup_steps",
    "came_beta3",
    "came_confidence_scale",
    "soap_precondition_frequency",
    "soap_large_side_identity_threshold",
    "soap_one_sided",
    "muon_momentum",
    "muon_ns_steps",
    "rational_matrix_policy_adam_lr_scale",
    "rational_matrix_policy_group_gain_strength",
    "rational_matrix_policy_group_pressure_strength",
    "rational_matrix_policy_group_activity_damping",
)


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


def final_eval(evals: list[dict[str, Any]]) -> tuple[int | None, float | None, float | None]:
    for record in reversed(evals):
        loss = finite_float(record.get("val_loss"))
        if loss is None:
            continue
        ppl = finite_float(record.get("val_ppl"))
        if ppl is None:
            ppl = math.exp(min(20.0, loss))
        return finite_int(record.get("step")), loss, ppl
    return None, None, None


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


def first_nonfinite_step(records: list[dict[str, Any]], key: str) -> int | None:
    for record in records:
        value = finite_float(record.get(key))
        if value is None:
            return finite_int(record.get("step"))
    return None


def mean_key(records: list[dict[str, Any]], key: str) -> float | None:
    values = [finite_float(record.get(key)) for record in records]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def read_jsonl(path: Path) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    stopped_record: dict[str, Any] | None = None
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
            elif event == "stopped_early":
                stopped_record = record
    if not config:
        return None
    final_step, final_loss, final_ppl = final_eval(evals)
    configured_steps = finite_int(config.get("steps"))
    summary_steps = finite_int(summary.get("steps")) if summary else None
    completed_steps = finite_int(summary.get("completed_steps")) if summary else None
    early_stopped = bool(stopped_record) or (bool(summary.get("stopped_early")) if summary else False)
    stop_reason = (stopped_record or {}).get("reason") or ((summary or {}).get("stop_reason"))
    complete = configured_steps is not None and summary_steps == configured_steps and completed_steps in (None, configured_steps)
    train_nan = first_nonfinite_step(train, "loss")
    eval_nan = first_nonfinite_step(evals, "val_loss")
    status = "diverged" if train_nan or eval_nan else ("stopped_early" if early_stopped else ("complete" if complete else "running"))
    row: dict[str, Any] = {
        "task": infer_task(path, config),
        "source": str(path),
        "run_name": path.parent.name,
        "activation": config.get("activation"),
        "optimizer": config.get("optimizer"),
        "seed": finite_int(config.get("seed")),
        "status": status,
        "complete": complete,
        "stopped_early": early_stopped,
        "stop_reason": stop_reason,
        "steps": configured_steps,
        "completed_steps": completed_steps,
        "final_step": final_step,
        "final_val_loss": final_loss,
        "final_val_ppl": final_ppl,
        "val_loss_auc_250": trapezoid_auc(evals, "val_loss", 250),
        "val_loss_auc_500": trapezoid_auc(evals, "val_loss", 500),
        "val_loss_auc_1000": trapezoid_auc(evals, "val_loss", 1000),
        "val_loss_auc_full": trapezoid_auc(evals, "val_loss"),
        "train_diverged_step": train_nan,
        "eval_diverged_step": eval_nan,
        "mean_seconds_per_step": finite_float(summary.get("mean_seconds_per_step")) if summary else None,
        "tokens_per_second": finite_float(summary.get("tokens_per_second")) if summary else None,
        "mean_optimizer_step_seconds": mean_key(train, "optimizer_step_seconds"),
        "mean_forward_backward_seconds": mean_key(train, "forward_backward_seconds"),
        "grad_clip_rate": None,
    }
    clip_values = [record.get("grad_clip_triggered") for record in train if "grad_clip_triggered" in record]
    if clip_values:
        row["grad_clip_rate"] = sum(1 for value in clip_values if bool(value)) / len(clip_values)
    for key in HYPER_KEYS:
        row[key] = config.get(key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task"],
        row["activation"],
        row["optimizer"],
        row.get("optimizer_lr"),
        row.get("optimizer_weight_decay"),
        row.get("optimizer_beta1"),
        row.get("optimizer_beta2"),
        row.get("muon_momentum"),
        row.get("ademamix_alpha"),
        row.get("ademamix_beta3"),
        row.get("schedule_free_beta1"),
        row.get("came_confidence_scale"),
        row.get("soap_precondition_frequency"),
        row.get("soap_one_sided"),
        row.get("rational_matrix_policy_adam_lr_scale"),
        row.get("rational_matrix_policy_group_gain_strength"),
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for _, items in buckets.items():
        first = items[0]
        losses = [item["final_val_loss"] for item in items if item.get("final_val_loss") is not None]
        aucs = [item["val_loss_auc_full"] for item in items if item.get("val_loss_auc_full") is not None]
        seconds = [item["mean_seconds_per_step"] for item in items if item.get("mean_seconds_per_step") is not None]
        row = {key: first.get(key) for key in ["task", "activation", "optimizer", *HYPER_KEYS]}
        row.update(
            {
                "n": len(items),
                "complete_n": sum(1 for item in items if item.get("complete")),
                "stopped_n": sum(1 for item in items if item.get("status") == "stopped_early"),
                "diverged_n": sum(1 for item in items if item.get("status") in {"diverged", "stopped_early"}),
                "mean_final_val_loss": statistics.fmean(losses) if losses else None,
                "std_final_val_loss": statistics.stdev(losses) if len(losses) > 1 else (0.0 if losses else None),
                "mean_val_loss_auc_full": statistics.fmean(aucs) if aucs else None,
                "mean_seconds_per_step": statistics.fmean(seconds) if seconds else None,
                "sources": ";".join(item["source"] for item in items),
            }
        )
        out.append(row)
    return sorted(out, key=lambda row: (row["task"], row["mean_final_val_loss"] is None, row["mean_final_val_loss"] or 1e9))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase A HPO Summary\n\n")
        handle.write("Rows are grouped by task, activation, optimizer, LR/WD, and family-specific knobs. Lower validation loss is better.\n\n")
        for task in sorted({row["task"] for row in rows}):
            handle.write(f"## {TASK_NAMES.get(task, task)}\n\n")
            task_rows = [row for row in rows if row["task"] == task]
            handle.write("| rank | optimizer | activation | lr | wd | n | div | mean loss | auc | sec/step | key knobs |\n")
            handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
            for rank, row in enumerate(task_rows[:50], start=1):
                loss = row.get("mean_final_val_loss")
                auc = row.get("mean_val_loss_auc_full")
                sec = row.get("mean_seconds_per_step")
                knobs = []
                for key in HYPER_KEYS[5:]:
                    value = row.get(key)
                    if value not in (None, ""):
                        knobs.append(f"{key}={value}")
                handle.write(
                    "| {rank} | {opt} | {act} | {lr} | {wd} | {n} | {div} | {loss} | {auc} | {sec} | {knobs} |\n".format(
                        rank=rank,
                        opt=row.get("optimizer"),
                        act=row.get("activation"),
                        lr=row.get("optimizer_lr"),
                        wd=row.get("optimizer_weight_decay"),
                        n=row.get("n"),
                        div=row.get("diverged_n"),
                        loss="" if loss is None else f"{loss:.6f}",
                        auc="" if auc is None else f"{auc:.6f}",
                        sec="" if sec is None else f"{sec:.4f}",
                        knobs=", ".join(knobs[:6]),
                    )
                )
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.run_root.glob("**/*.jsonl")):
        if ".incomplete_" in str(path):
            continue
        row = read_jsonl(path)
        if row is not None:
            rows.append(row)

    fieldnames = [
        "task",
        "source",
        "run_name",
        "activation",
        "optimizer",
        "seed",
        "status",
        "complete",
        "stopped_early",
        "stop_reason",
        "steps",
        "completed_steps",
        "final_step",
        "final_val_loss",
        "final_val_ppl",
        "val_loss_auc_250",
        "val_loss_auc_500",
        "val_loss_auc_1000",
        "val_loss_auc_full",
        "train_diverged_step",
        "eval_diverged_step",
        "mean_seconds_per_step",
        "tokens_per_second",
        "mean_optimizer_step_seconds",
        "mean_forward_backward_seconds",
        "grad_clip_rate",
        *HYPER_KEYS,
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase_a_hpo_runs.csv", rows, fieldnames)
    agg = aggregate(rows)
    aggregate_fields = ["task", "activation", "optimizer", *HYPER_KEYS, "n", "complete_n", "stopped_n", "diverged_n", "mean_final_val_loss", "std_final_val_loss", "mean_val_loss_auc_full", "mean_seconds_per_step", "sources"]
    write_csv(args.output_dir / "phase_a_hpo_rankings.csv", agg, aggregate_fields)
    write_markdown(args.output_dir / "phase_a_hpo_summary.md", agg)
    print(json.dumps({"runs": len(rows), "groups": len(agg), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
