#!/usr/bin/env python3
"""Summarize CPU-readable phase timing from existing ICLR JSONL logs.

This does not import torch and does not touch GPUs. It only reads manifest rows and
train events that already contain timing telemetry from completed/in-progress runs.

Important: the phase timing fields are logged-step diagnostics. For MatrixPolicy,
logged optimizer timing includes telemetry capture on log steps, so it is not the
average non-logging optimizer-step overhead. Use runtime summary CSVs for average
seconds/step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("experiments/manifests/iclr26_main_manifest.csv")
DEFAULT_RUN_ROOT = Path("experiments/runs/iclr26_main")
DEFAULT_OUTPUT = Path("experiments/rlb_acceleration/phase_timing_summary.csv")

METHOD_LABELS = {
    "rlb_matrixpolicy_original": "RLB+MatrixPolicy",
    "silu_adamw": "SiLU+AdamW",
    "rlb_adamw": "RLB+AdamW",
    "silu_muon": "SiLU+Muon",
    "rlb_muon": "RLB+Muon",
    "silu_lion": "SiLU+Lion",
    "rlb_lion": "RLB+Lion",
    "silu_soap": "SiLU+SOAP",
    "rlb_soap": "RLB+SOAP",
    "silu_ademamix": "SiLU+ADeMaMix",
    "rlb_ademamix": "RLB+ADeMaMix",
    "silu_came": "SiLU+CAME",
    "rlb_came": "RLB+CAME",
    "silu_schedulefree": "SiLU+ScheduleFree",
    "rlb_schedulefree": "RLB+ScheduleFree",
}


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def read_manifest(path: Path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def jsonl_path(run_root: Path, row: dict[str, str]) -> Path:
    return run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"


def read_train_events(path: Path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "train":
                yield record


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)

    output = []
    for key_tuple, group_rows in sorted(buckets.items()):
        first = group_rows[0]
        out = {key: value for key, value in zip(keys, key_tuple)}
        out.update(
            {
                "rows": len({row["row_index"] for row in group_rows}),
                "train_events": len(group_rows),
                "logged_forward_backward_seconds_mean": mean([row["forward_backward_seconds"] for row in group_rows]),
                "logged_forward_backward_seconds_std": sample_std([row["forward_backward_seconds"] for row in group_rows]),
                "logged_optimizer_step_seconds_mean": mean([row["optimizer_step_seconds"] for row in group_rows]),
                "logged_optimizer_step_seconds_std": sample_std([row["optimizer_step_seconds"] for row in group_rows]),
                "seconds_per_step_mean": mean([row["seconds_per_step"] for row in group_rows]),
                "tokens_per_second_mean": mean([row["tokens_per_second"] for row in group_rows]),
                "method_label": first["method_label"],
                "activation": first["activation"],
                "optimizer": first["optimizer"],
            }
        )
        fb = out["logged_forward_backward_seconds_mean"]
        opt = out["logged_optimizer_step_seconds_mean"]
        step = out["seconds_per_step_mean"]
        out["logged_optimizer_fraction_of_recent_step"] = (opt / step) if opt is not None and step not in (None, 0.0) else None
        out["logged_forward_backward_fraction_of_recent_step"] = (fb / step) if fb is not None and step not in (None, 0.0) else None
        output.append(out)
    return output


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", action="append", help="Restrict to a phase. May be repeated.")
    args = parser.parse_args()

    phase_filter = set(args.phase or [])
    per_event = []
    for row in read_manifest(args.manifest):
        if phase_filter and row["phase"] not in phase_filter:
            continue
        path = jsonl_path(args.run_root, row)
        for event in read_train_events(path):
            per_event.append(
                {
                    "phase": row["phase"],
                    "dataset": row["dataset"],
                    "row_index": int(row["row_index"]),
                    "method": row["method"],
                    "method_label": method_label(row["method"]),
                    "activation": row["activation"],
                    "optimizer": row["optimizer"],
                    "forward_backward_seconds": finite_float(event.get("forward_backward_seconds")),
                    "optimizer_step_seconds": finite_float(event.get("optimizer_step_seconds")),
                    "seconds_per_step": finite_float(event.get("seconds_per_step")),
                    "tokens_per_second": finite_float(event.get("tokens_per_second")),
                }
            )

    rows = aggregate(per_event, ("phase", "dataset", "method"))
    fieldnames = [
        "phase",
        "dataset",
        "method",
        "method_label",
        "activation",
        "optimizer",
        "rows",
        "train_events",
        "logged_forward_backward_seconds_mean",
        "logged_forward_backward_seconds_std",
        "logged_optimizer_step_seconds_mean",
        "logged_optimizer_step_seconds_std",
        "seconds_per_step_mean",
        "tokens_per_second_mean",
        "logged_forward_backward_fraction_of_recent_step",
        "logged_optimizer_fraction_of_recent_step",
    ]
    write_csv(args.output, rows, fieldnames)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
