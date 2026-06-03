#!/usr/bin/env python3
"""Summarize Phase 1 protocol-lock JSONL traces with curve-first outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

CURVE_FIELDNAMES = [
    "task",
    "run_name",
    "label",
    "activation",
    "optimizer",
    "seed",
    "step",
    "tokens",
    "gpu_hours_approx",
]

COLORS = {
    "adamw/silu": "#3366aa",
    "adamw/rlb": "#4a9955",
    "rational_matrix_policy_onpolicy/rlb": "#b9332f",
    "muon/silu": "#888888",
    "muon/rlb": "#c06f3c",
    "soap_adamw/silu": "#8a67c7",
    "soap_adamw/rlb": "#6f4ba8",
    "lion/silu": "#d99a2b",
    "lion/rlb": "#b5791f",
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


def fmt_float(value: Any, places: int = 6) -> str:
    number = finite_float(value)
    return "" if number is None else f"{number:.{places}f}"


def fmt_compact(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:g}"


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


def method_name(config: dict[str, Any]) -> str:
    activation = str(config.get("activation") or "")
    optimizer = str(config.get("optimizer") or "")
    is_rlb = activation.startswith("rlb")
    if optimizer == "adamw" and activation == "silu":
        return "SiLU+AdamW"
    if optimizer == "adamw" and is_rlb:
        return "RLB+AdamW"
    if optimizer == "muon" and activation == "silu":
        return "SiLU+Muon"
    if optimizer == "muon" and is_rlb:
        return "RLB+Muon"
    if optimizer == "soap_adamw" and activation == "silu":
        return "SiLU+SOAP"
    if optimizer == "soap_adamw" and is_rlb:
        return "RLB+SOAP"
    if optimizer == "rational_matrix_policy_onpolicy" and is_rlb:
        return "RLB+MatrixPolicy"
    if optimizer == "lion" and activation == "silu":
        return "SiLU+Lion"
    if optimizer == "lion" and is_rlb:
        return "RLB+Lion"
    prefix = "RLB" if is_rlb else ("SiLU" if activation == "silu" else activation)
    return f"{prefix}+{optimizer}"


def curve_label(config: dict[str, Any]) -> str:
    parts = [method_name(config)]
    lr = fmt_compact(config.get("optimizer_lr"))
    wd = fmt_compact(config.get("optimizer_weight_decay"))
    if lr:
        parts.append(f"lr={lr}")
    if wd:
        parts.append(f"wd={wd}")
    optimizer = config.get("optimizer")
    if optimizer == "rational_matrix_policy_onpolicy":
        scale = fmt_compact(config.get("rational_matrix_policy_adam_lr_scale"))
        gain = fmt_compact(config.get("rational_matrix_policy_group_gain_strength"))
        if scale:
            parts.append(f"as={scale}")
        if gain:
            parts.append(f"gg={gain}")
    elif optimizer == "muon":
        momentum = fmt_compact(config.get("muon_momentum"))
        if momentum:
            parts.append(f"mom={momentum}")
    elif optimizer == "soap_adamw":
        freq = fmt_compact(config.get("soap_precondition_frequency"))
        one = config.get("soap_one_sided")
        if freq:
            parts.append(f"f={freq}")
        if one is not None:
            parts.append(f"one={one}")
    return " ".join(parts)


def plot_color(config: dict[str, Any]) -> str | None:
    activation = str(config.get("activation") or "")
    role = "rlb" if activation.startswith("rlb") else activation
    return COLORS.get(f"{config.get('optimizer')}/{role}")


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


def best_eval(evals: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    best_step: int | None = None
    best_loss: float | None = None
    for record in evals:
        step = finite_int(record.get("step"))
        loss = finite_float(record.get("val_loss"))
        if step is None or loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_step = step
            best_loss = loss
    return best_step, best_loss


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


def eval_density(evals: list[dict[str, Any]]) -> tuple[int | None, int | None, int]:
    steps = sorted(step for record in evals if (step := finite_int(record.get("step"))) is not None)
    if len(steps) < 2:
        return None, None, len(steps)
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    return min(gaps), max(gaps), len(steps)


def read_trace(path: Path) -> dict[str, Any] | None:
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
    return {
        "path": path,
        "task": infer_task(path, config),
        "config": config,
        "train": train,
        "eval": evals,
        "summary": summary,
        "stopped_record": stopped_record,
    }


def summarize_trace(trace: dict[str, Any], dense_eval_max_interval: int) -> dict[str, Any]:
    path: Path = trace["path"]
    config = trace["config"]
    train = trace["train"]
    evals = trace["eval"]
    summary = trace["summary"]
    stopped_record = trace["stopped_record"]
    final_step, final_loss, final_ppl = final_eval(evals)
    best_step, best_loss = best_eval(evals)
    configured_steps = finite_int(config.get("steps"))
    summary_steps = finite_int(summary.get("steps")) if summary else None
    completed_steps = finite_int(summary.get("completed_steps")) if summary else None
    early_stopped = bool(stopped_record) or (bool(summary.get("stopped_early")) if summary else False)
    stop_reason = (stopped_record or {}).get("reason") or ((summary or {}).get("stop_reason"))
    complete = configured_steps is not None and summary_steps == configured_steps and completed_steps in (None, configured_steps)
    train_nan = first_nonfinite_step(train, "loss")
    eval_nan = first_nonfinite_step(evals, "val_loss")
    status = "diverged" if train_nan or eval_nan else ("stopped_early" if early_stopped else ("complete" if complete else "running"))
    min_gap, max_gap, eval_points = eval_density(evals)
    configured_eval_interval = finite_int(config.get("eval_interval"))
    curve_density_ok = (
        configured_eval_interval is not None
        and configured_eval_interval <= dense_eval_max_interval
        and (max_gap is None or max_gap <= dense_eval_max_interval)
    )
    row: dict[str, Any] = {
        "task": trace["task"],
        "source": str(path),
        "run_name": path.parent.name,
        "label": curve_label(config),
        "method": method_name(config),
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
        "best_val_step": best_step,
        "best_val_loss": best_loss,
        "val_loss_auc_250": trapezoid_auc(evals, "val_loss", 250),
        "val_loss_auc_500": trapezoid_auc(evals, "val_loss", 500),
        "val_loss_auc_1000": trapezoid_auc(evals, "val_loss", 1000),
        "val_loss_auc_full": trapezoid_auc(evals, "val_loss"),
        "train_diverged_step": train_nan,
        "eval_diverged_step": eval_nan,
        "eval_points": eval_points,
        "eval_interval_configured": configured_eval_interval,
        "eval_interval_observed_min": min_gap,
        "eval_interval_observed_max": max_gap,
        "curve_density_ok": curve_density_ok,
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


def approx_gpu_hours(step: int | None, seconds_per_step: float | None, world_size: int | None) -> float | None:
    if step is None or seconds_per_step is None or world_size is None:
        return None
    return step * seconds_per_step * world_size / 3600.0


def nearest_seconds_per_step(train_records: list[dict[str, Any]], step: int | None) -> float | None:
    if step is None:
        return None
    best_step: int | None = None
    best_value: float | None = None
    for record in train_records:
        train_step = finite_int(record.get("step"))
        seconds = finite_float(record.get("seconds_per_step"))
        if train_step is None or seconds is None or train_step > step:
            continue
        if best_step is None or train_step >= best_step:
            best_step = train_step
            best_value = seconds
    return best_value


def curve_rows(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = trace["config"]
    task = trace["task"]
    run_name = trace["path"].parent.name
    label = curve_label(config)
    seed = finite_int(config.get("seed"))
    world_size = finite_int(config.get("world_size"))
    tokens_per_step = finite_int(config.get("global_tokens_per_step"))
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for record in trace["train"]:
        step = finite_int(record.get("step"))
        seconds = finite_float(record.get("seconds_per_step"))
        train_rows.append(
            {
                "task": task,
                "run_name": run_name,
                "label": label,
                "activation": config.get("activation"),
                "optimizer": config.get("optimizer"),
                "seed": seed,
                "step": step,
                "tokens": None if step is None or tokens_per_step is None else step * tokens_per_step,
                "gpu_hours_approx": approx_gpu_hours(step, seconds, world_size),
                "loss": finite_float(record.get("loss")),
                "lr": finite_float(record.get("lr")),
                "seconds_per_step": seconds,
                "grad_clip_triggered": bool(record.get("grad_clip_triggered")) if "grad_clip_triggered" in record else None,
                "grad_global_norm_before_clip": finite_float(record.get("grad_global_norm_before_clip")),
            }
        )
    for record in trace["eval"]:
        step = finite_int(record.get("step"))
        seconds = nearest_seconds_per_step(trace["train"], step)
        eval_rows.append(
            {
                "task": task,
                "run_name": run_name,
                "label": label,
                "activation": config.get("activation"),
                "optimizer": config.get("optimizer"),
                "seed": seed,
                "step": step,
                "tokens": None if step is None or tokens_per_step is None else step * tokens_per_step,
                "gpu_hours_approx": approx_gpu_hours(step, seconds, world_size),
                "val_loss": finite_float(record.get("val_loss")),
                "val_ppl": finite_float(record.get("val_ppl")),
            }
        )
    return train_rows, eval_rows


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


def mean_or_none(values: list[Any]) -> float | None:
    finite = [value for value in (finite_float(v) for v in values) if value is not None]
    return statistics.fmean(finite) if finite else None


def std_or_none(values: list[Any]) -> float | None:
    finite = [value for value in (finite_float(v) for v in values) if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row)].append(row)
    out: list[dict[str, Any]] = []
    for _, items in buckets.items():
        first = items[0]
        row = {key: first.get(key) for key in ["task", "method", "label", "activation", "optimizer", *HYPER_KEYS]}
        row.update(
            {
                "n": len(items),
                "complete_n": sum(1 for item in items if item.get("complete")),
                "running_n": sum(1 for item in items if item.get("status") == "running"),
                "stopped_n": sum(1 for item in items if item.get("status") == "stopped_early"),
                "diverged_n": sum(1 for item in items if item.get("status") in {"diverged", "stopped_early"}),
                "curve_density_ok_n": sum(1 for item in items if item.get("curve_density_ok")),
                "eval_points_min": min((item.get("eval_points") or 0) for item in items),
                "mean_final_val_loss": mean_or_none([item.get("final_val_loss") for item in items]),
                "std_final_val_loss": std_or_none([item.get("final_val_loss") for item in items]),
                "mean_best_val_loss": mean_or_none([item.get("best_val_loss") for item in items]),
                "mean_val_loss_auc_250": mean_or_none([item.get("val_loss_auc_250") for item in items]),
                "mean_val_loss_auc_500": mean_or_none([item.get("val_loss_auc_500") for item in items]),
                "mean_val_loss_auc_1000": mean_or_none([item.get("val_loss_auc_1000") for item in items]),
                "mean_val_loss_auc_full": mean_or_none([item.get("val_loss_auc_full") for item in items]),
                "mean_seconds_per_step": mean_or_none([item.get("mean_seconds_per_step") for item in items]),
                "sources": ";".join(item["source"] for item in items),
            }
        )
        out.append(row)
    return sorted(out, key=lambda row: (row["task"], row["mean_val_loss_auc_full"] is None, row["mean_val_loss_auc_full"] or 1e9, row["mean_final_val_loss"] or 1e9))


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower()).strip("_") or "plot"


def finite_curve_points(rows: list[dict[str, Any]], y_key: str, x_key: str = "step") -> tuple[list[float], list[float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        x = finite_float(row.get(x_key))
        y = finite_float(row.get(y_key))
        if x is not None and y is not None:
            pairs.append((x, y))
    pairs.sort(key=lambda pair: pair[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_curve_set(rows: list[dict[str, Any]], y_key: str, ylabel: str, out_path: Path, title: str, x_key: str = "step", min_step: int | None = None) -> bool:
    plot_rows = [row for row in rows if min_step is None or (finite_int(row.get("step")) or -1) >= min_step]
    labels = sorted({str(row.get("label")) for row in plot_rows if row.get("label")})
    if not labels:
        return False
    plt.figure(figsize=(10.5, 6.0))
    plotted = False
    for label in labels:
        series = [row for row in plot_rows if row.get("label") == label]
        xs, ys = finite_curve_points(series, y_key, x_key=x_key)
        if not xs:
            continue
        first = series[0]
        color = plot_color(first)
        plt.plot(xs, ys, label=label, color=color, linewidth=2.2 if "MatrixPolicy" in label else 1.5, alpha=0.95)
        plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("step" if x_key == "step" else x_key.replace("_", " "))
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=6, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def write_plots(output_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], zoom_min_step: int) -> list[Path]:
    paths: list[Path] = []
    for task in sorted({str(row.get("task")) for row in train_rows + eval_rows if row.get("task")}):
        task_label = TASK_NAMES.get(task, task)
        task_train = [row for row in train_rows if row.get("task") == task]
        task_eval = [row for row in eval_rows if row.get("task") == task]
        candidates = [
            (task_eval, "val_loss", "validation loss", output_dir / f"{task}_validation_loss_curves.png", f"{task_label} validation loss"),
            (task_eval, "val_ppl", "validation PPL", output_dir / f"{task}_validation_ppl_curves.png", f"{task_label} validation PPL"),
            (task_train, "loss", "training loss", output_dir / f"{task}_training_loss_curves.png", f"{task_label} training loss"),
            (task_eval, "val_loss", "validation loss", output_dir / f"{task}_validation_loss_curves_zoom_step{zoom_min_step}.png", f"{task_label} validation loss, step >= {zoom_min_step}", zoom_min_step),
        ]
        for item in candidates:
            rows, y_key, ylabel, path, title, *rest = item
            min_step = rest[0] if rest else None
            if plot_curve_set(rows, y_key, ylabel, path, title, min_step=min_step):
                paths.append(path)
    return paths


def write_markdown(path: Path, rows: list[dict[str, Any]], plot_paths: list[Path], output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Phase 1 Protocol-Lock Summary\n\n")
        handle.write("Curve evidence is primary. Use the dense eval/train curve CSVs and plots before interpreting final validation loss. Lower AUC and lower validation loss are better.\n\n")
        handle.write("Generated curve artifacts:\n\n")
        handle.write("```text\n")
        handle.write("eval_curves.csv\ntrain_curves.csv\n")
        for plot_path in plot_paths:
            handle.write(f"{plot_path.relative_to(output_dir)}\n")
        handle.write("```\n\n")
        for task in sorted({row["task"] for row in rows}):
            handle.write(f"## {TASK_NAMES.get(task, task)}\n\n")
            task_rows = [row for row in rows if row["task"] == task]
            handle.write("| rank | optimizer | activation | lr | wd | n | running | div | dense | eval pts | final loss | best loss | auc full | auc 500 | sec/step | key knobs |\n")
            handle.write("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
            for rank, row in enumerate(task_rows[:80], start=1):
                knobs = []
                for key in HYPER_KEYS[5:]:
                    value = row.get(key)
                    if value not in (None, ""):
                        knobs.append(f"{key}={value}")
                handle.write(
                    "| {rank} | {opt} | {act} | {lr} | {wd} | {n} | {running} | {div} | {dense} | {pts} | {loss} | {best} | {auc} | {auc500} | {sec} | {knobs} |\n".format(
                        rank=rank,
                        opt=row.get("optimizer"),
                        act=row.get("activation"),
                        lr=fmt_compact(row.get("optimizer_lr")),
                        wd=fmt_compact(row.get("optimizer_weight_decay")),
                        n=row.get("n"),
                        running=row.get("running_n"),
                        div=row.get("diverged_n"),
                        dense=row.get("curve_density_ok_n"),
                        pts=row.get("eval_points_min"),
                        loss=fmt_float(row.get("mean_final_val_loss")),
                        best=fmt_float(row.get("mean_best_val_loss")),
                        auc=fmt_float(row.get("mean_val_loss_auc_full")),
                        auc500=fmt_float(row.get("mean_val_loss_auc_500")),
                        sec=fmt_float(row.get("mean_seconds_per_step"), places=4),
                        knobs=", ".join(knobs[:6]),
                    )
                )
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-eval-max-interval", type=int, default=50)
    parser.add_argument("--zoom-min-step", type=int, default=250)
    args = parser.parse_args()

    traces: list[dict[str, Any]] = []
    for path in sorted(args.run_root.glob("**/*.jsonl")):
        if ".incomplete_" in str(path):
            continue
        trace = read_trace(path)
        if trace is not None:
            traces.append(trace)

    rows = [summarize_trace(trace, args.dense_eval_max_interval) for trace in traces]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for trace in traces:
        trace_train, trace_eval = curve_rows(trace)
        train_rows.extend(trace_train)
        eval_rows.extend(trace_eval)

    run_fields = [
        "task",
        "source",
        "run_name",
        "label",
        "method",
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
        "best_val_step",
        "best_val_loss",
        "val_loss_auc_250",
        "val_loss_auc_500",
        "val_loss_auc_1000",
        "val_loss_auc_full",
        "train_diverged_step",
        "eval_diverged_step",
        "eval_points",
        "eval_interval_configured",
        "eval_interval_observed_min",
        "eval_interval_observed_max",
        "curve_density_ok",
        "mean_seconds_per_step",
        "tokens_per_second",
        "mean_optimizer_step_seconds",
        "mean_forward_backward_seconds",
        "grad_clip_rate",
        *HYPER_KEYS,
    ]
    train_fields = [*CURVE_FIELDNAMES, "loss", "lr", "seconds_per_step", "grad_clip_triggered", "grad_global_norm_before_clip"]
    eval_fields = [*CURVE_FIELDNAMES, "val_loss", "val_ppl"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase1_protocol_lock_runs.csv", rows, run_fields)
    write_csv(args.output_dir / "train_curves.csv", train_rows, train_fields)
    write_csv(args.output_dir / "eval_curves.csv", eval_rows, eval_fields)

    agg = aggregate(rows)
    aggregate_fields = [
        "task",
        "method",
        "label",
        "activation",
        "optimizer",
        *HYPER_KEYS,
        "n",
        "complete_n",
        "running_n",
        "stopped_n",
        "diverged_n",
        "curve_density_ok_n",
        "eval_points_min",
        "mean_final_val_loss",
        "std_final_val_loss",
        "mean_best_val_loss",
        "mean_val_loss_auc_250",
        "mean_val_loss_auc_500",
        "mean_val_loss_auc_1000",
        "mean_val_loss_auc_full",
        "mean_seconds_per_step",
        "sources",
    ]
    write_csv(args.output_dir / "phase1_protocol_lock_rankings.csv", agg, aggregate_fields)
    plot_paths = write_plots(args.output_dir, train_rows, eval_rows, args.zoom_min_step)
    write_markdown(args.output_dir / "phase1_protocol_lock_summary.md", agg, plot_paths, args.output_dir)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "groups": len(agg),
                "train_curve_points": len(train_rows),
                "eval_curve_points": len(eval_rows),
                "plots": len(plot_paths),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
