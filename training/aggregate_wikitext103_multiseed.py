#!/usr/bin/env python3
"""Aggregate WikiText-103 JSONL files by optimizer, activation, and seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline", default="silu", help="external baseline activation")
    parser.add_argument("--baseline-optimizer", default="", help="optimizer for the external baseline; defaults to each row optimizer")
    parser.add_argument("--classic-optimizer", default="adamw", help="classic optimizer used for same-activation optimizer gaps")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--log-path", default="")
    return parser.parse_args()


def parse_run_dir(run_dir: Path) -> dict[tuple[str, str, int], dict[str, object]]:
    records: dict[tuple[str, str, int], dict[str, object]] = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        current_key: tuple[str, str, int] | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                event_type = event.get("event")
                if event_type == "config":
                    optimizer = str(event.get("optimizer", "adamw"))
                    activation = str(event["activation"])
                    seed = int(event["seed"])
                    current_key = (optimizer, activation, seed)
                    records[current_key] = {
                        "optimizer": optimizer,
                        "activation": activation,
                        "seed": seed,
                        "params": int(event["params"]),
                        "steps": int(event["steps"]),
                        "config": event,
                        "final_eval": None,
                        "summary": None,
                    }
                    continue
                if current_key is None:
                    continue
                record = records[current_key]
                if event_type == "eval":
                    previous = record.get("final_eval")
                    if previous is None or int(event["step"]) >= int(previous["step"]):
                        record["final_eval"] = event
                elif event_type == "summary":
                    record["summary"] = event
    return records


def sample_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def build_rows(
    records: dict[tuple[str, str, int], dict[str, object]],
    baseline: str,
    baseline_optimizer: str,
    classic_optimizer: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    complete: dict[tuple[str, str, int], dict[str, object]] = {}
    for key, record in records.items():
        if record.get("final_eval") is None or record.get("summary") is None:
            continue
        complete[key] = record

    optimizers = sorted({optimizer for optimizer, _, _ in complete})
    seeds = sorted({seed for _, _, seed in complete})
    activations = sorted({activation for _, activation, _ in complete})
    per_seed: list[dict[str, object]] = []
    external_optimizer_default = baseline_optimizer or ""
    for optimizer in optimizers:
        for seed in seeds:
            external_optimizer = external_optimizer_default or optimizer
            base = complete.get((external_optimizer, baseline, seed))
            base_eval = base.get("final_eval") if base else None
            base_summary = base.get("summary") if base else None
            base_loss = float(base_eval["val_loss"]) if base_eval else math.nan
            base_ppl = float(base_eval["val_ppl"]) if base_eval else math.nan
            base_seconds = (
                float(base_summary["mean_seconds_per_step"]) if base_summary else math.nan
            )
            for activation in activations:
                record = complete.get((optimizer, activation, seed))
                if record is None:
                    continue
                classic = complete.get((classic_optimizer, activation, seed))
                classic_eval = classic.get("final_eval") if classic else None
                classic_summary = classic.get("summary") if classic else None
                classic_loss = float(classic_eval["val_loss"]) if classic_eval else math.nan
                classic_ppl = float(classic_eval["val_ppl"]) if classic_eval else math.nan
                classic_seconds = (
                    float(classic_summary["mean_seconds_per_step"]) if classic_summary else math.nan
                )
                final_eval = record["final_eval"]
                summary = record["summary"]
                val_loss = float(final_eval["val_loss"])
                val_ppl = float(final_eval["val_ppl"])
                seconds = float(summary["mean_seconds_per_step"])
                per_seed.append(
                    {
                        "optimizer": optimizer,
                        "seed": seed,
                        "activation": activation,
                        "params": int(record["params"]),
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "mean_seconds_per_step": seconds,
                        "tokens_per_second": float(summary["tokens_per_second"]),
                        "total_seconds": float(summary["total_seconds"]),
                        "external_baseline_optimizer": external_optimizer,
                        "external_baseline_activation": baseline,
                        "classic_optimizer": classic_optimizer,
                        "loss_gap_vs_external_baseline": val_loss - base_loss,
                        "ppl_gap_vs_external_baseline": val_ppl - base_ppl,
                        "time_ratio_vs_external_baseline": seconds / base_seconds,
                        "loss_gap_vs_classic_same_activation": val_loss - classic_loss,
                        "ppl_gap_vs_classic_same_activation": val_ppl - classic_ppl,
                        "time_ratio_vs_classic_same_activation": seconds / classic_seconds,
                        "loss_gap_vs_silu": val_loss - base_loss,
                        "ppl_gap_vs_silu": val_ppl - base_ppl,
                        "time_ratio_vs_silu": seconds / base_seconds,
                    }
                )

    aggregate: list[dict[str, object]] = []
    for optimizer in optimizers:
        for activation in activations:
            rows = [
                row
                for row in per_seed
                if row["optimizer"] == optimizer and row["activation"] == activation
            ]
            if not rows:
                continue
            aggregate.append(
                {
                    "optimizer": optimizer,
                    "activation": activation,
                    "seeds": len(rows),
                    "params": int(rows[0]["params"]),
                    "mean_val_loss": statistics.mean(float(r["val_loss"]) for r in rows),
                    "std_val_loss": sample_std([float(r["val_loss"]) for r in rows]),
                    "mean_val_ppl": statistics.mean(float(r["val_ppl"]) for r in rows),
                    "std_val_ppl": sample_std([float(r["val_ppl"]) for r in rows]),
                    "mean_seconds_per_step": statistics.mean(
                        float(r["mean_seconds_per_step"]) for r in rows
                    ),
                    "mean_loss_gap_vs_external_baseline": statistics.mean(
                        float(r["loss_gap_vs_external_baseline"]) for r in rows
                    ),
                    "std_loss_gap_vs_external_baseline": sample_std(
                        [float(r["loss_gap_vs_external_baseline"]) for r in rows]
                    ),
                    "mean_ppl_gap_vs_external_baseline": statistics.mean(
                        float(r["ppl_gap_vs_external_baseline"]) for r in rows
                    ),
                    "mean_time_ratio_vs_external_baseline": statistics.mean(
                        float(r["time_ratio_vs_external_baseline"]) for r in rows
                    ),
                    "mean_loss_gap_vs_classic_same_activation": statistics.mean(
                        float(r["loss_gap_vs_classic_same_activation"]) for r in rows
                    ),
                    "std_loss_gap_vs_classic_same_activation": sample_std(
                        [float(r["loss_gap_vs_classic_same_activation"]) for r in rows]
                    ),
                    "mean_ppl_gap_vs_classic_same_activation": statistics.mean(
                        float(r["ppl_gap_vs_classic_same_activation"]) for r in rows
                    ),
                    "mean_time_ratio_vs_classic_same_activation": statistics.mean(
                        float(r["time_ratio_vs_classic_same_activation"]) for r in rows
                    ),
                    "mean_loss_gap_vs_silu": statistics.mean(
                        float(r["loss_gap_vs_external_baseline"]) for r in rows
                    ),
                    "std_loss_gap_vs_silu": sample_std(
                        [float(r["loss_gap_vs_external_baseline"]) for r in rows]
                    ),
                    "mean_ppl_gap_vs_silu": statistics.mean(
                        float(r["ppl_gap_vs_external_baseline"]) for r in rows
                    ),
                    "mean_time_ratio_vs_silu": statistics.mean(
                        float(r["time_ratio_vs_external_baseline"]) for r in rows
                    ),
                }
            )
    return per_seed, aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def write_readme(
    out_dir: Path,
    run_dir: Path,
    job_id: str,
    log_path: str,
    per_seed: list[dict[str, object]],
    aggregate: list[dict[str, object]],
) -> None:
    per_seed_rows = [
        [
            str(row["optimizer"]),
            str(row["seed"]),
            str(row["activation"]),
            f"{int(row['params']):,}",
            fmt_float(row["val_loss"]),
            fmt_float(row["loss_gap_vs_external_baseline"]),
            fmt_float(row["loss_gap_vs_classic_same_activation"]),
            fmt_float(row["val_ppl"], 3),
            fmt_float(row["mean_seconds_per_step"], 6),
            f"{float(row['time_ratio_vs_external_baseline']):.3f}x",
            f"{float(row['time_ratio_vs_classic_same_activation']):.3f}x",
        ]
        for row in sorted(
            per_seed,
            key=lambda r: (str(r["optimizer"]), int(r["seed"]), str(r["activation"])),
        )
    ]
    aggregate_rows = [
        [
            str(row["optimizer"]),
            str(row["activation"]),
            str(row["seeds"]),
            f"{int(row['params']):,}",
            fmt_float(row["mean_val_loss"]),
            fmt_float(row["std_val_loss"]),
            fmt_float(row["mean_loss_gap_vs_external_baseline"]),
            fmt_float(row["std_loss_gap_vs_external_baseline"]),
            fmt_float(row["mean_loss_gap_vs_classic_same_activation"]),
            fmt_float(row["std_loss_gap_vs_classic_same_activation"]),
            fmt_float(row["mean_val_ppl"], 3),
            fmt_float(row["mean_seconds_per_step"], 6),
            f"{float(row['mean_time_ratio_vs_external_baseline']):.3f}x",
            f"{float(row['mean_time_ratio_vs_classic_same_activation']):.3f}x",
        ]
        for row in sorted(aggregate, key=lambda r: (str(r["optimizer"]), str(r["activation"])))
    ]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = f"""# WikiText-103 Benchmark Summary

Generated: {generated}

Raw run directory:

```text
{run_dir}
```

Slurm job:

```text
{job_id or "unknown"}
```

Slurm log:

```text
{log_path or "unknown"}
```

This result folder is an organized summary only. The raw JSONL logs remain in
the run directory above.

## Per-Seed Results

{markdown_table(per_seed_rows, ["Optimizer", "Seed", "Activation", "Params", "Val loss", "Gap vs external", "Gap vs classic", "PPL", "Sec/step", "Time vs external", "Time vs classic"])}

## Aggregate Results

{markdown_table(aggregate_rows, ["Optimizer", "Activation", "Seeds", "Params", "Mean loss", "Std loss", "Mean gap vs external", "Std gap vs external", "Mean gap vs classic", "Std gap vs classic", "Mean PPL", "Mean sec/step", "Mean time vs external", "Mean time vs classic"])}
"""
    out_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = parse_run_dir(args.run_dir)
    per_seed, aggregate = build_rows(records, args.baseline, args.baseline_optimizer, args.classic_optimizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_seed.csv", per_seed)
    write_csv(args.out_dir / "aggregate.csv", aggregate)
    payload = {
        "run_dir": str(args.run_dir),
        "job_id": args.job_id,
        "log_path": args.log_path,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_readme(args.out_dir, args.run_dir, args.job_id, args.log_path, per_seed, aggregate)


if __name__ == "__main__":
    main()
