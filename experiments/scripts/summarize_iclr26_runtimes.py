#!/usr/bin/env python3
"""Summarize completed ICLR run runtimes from JSONL summary records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


DEFAULT_OUTPUT = Path("experiments/results/iclr26_runtime_summary_2026_06_11")
MANIFEST = Path("experiments/manifests/iclr26_main_manifest.csv")
RUN_ROOT = Path("experiments/runs/iclr26_main")

# E1 FineWeb-Edu seed 2027 ran as Slurm job 158117 with Restarts=6.
# The final JSONLs do not append duplicate attempts, but rows in this matched
# cell are contaminated by restart/node effects and should not be used as the
# default clean throughput estimate.
RESTART_AFFECTED_ROWS = set(range(75, 90))

COMPLETED_E2_DATASET_SCOPES = {
    "dclm": "E2_m0_300m_dclm",
    "fineweb_edu": "E2_m0_300m_fineweb_edu",
}

SCOPE_LABEL = {
    "E1_m0_100m_all_datasets": "E1 M0/100M All Datasets",
    "E2_m0_300m_dclm": "E2 M0/300M DCLM",
    "E2_m0_300m_fineweb_edu": "E2 M0/300M FineWeb-Edu",
}

METHOD_ORDER = [
    "rlb_matrixpolicy_original",
    "silu_adamw",
    "rlb_adamw",
    "silu_muon",
    "rlb_muon",
    "silu_lion",
    "rlb_lion",
    "silu_soap",
    "rlb_soap",
    "silu_ademamix",
    "rlb_ademamix",
    "silu_came",
    "rlb_came",
    "silu_schedulefree",
    "rlb_schedulefree",
]

METHOD_LABEL = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def scope_for(row: dict[str, str]) -> str | None:
    phase = row["phase"]
    dataset = row["dataset"]
    if phase == "E1_m0_100m":
        return "E1_m0_100m_all_datasets"
    if phase == "E2_m0_300m" and dataset in COMPLETED_E2_DATASET_SCOPES:
        return COMPLETED_E2_DATASET_SCOPES[dataset]
    return None


def read_summary(path: Path) -> dict[str, float | int | str | bool] | None:
    if not path.exists():
        return None
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "summary":
                return record
    return None


def fmt_seconds(value: float) -> str:
    return f"{value:.1f}s"


def fmt_minutes(value: float) -> str:
    return f"{value / 60.0:.1f} min"


def fmt_hours(value: float) -> str:
    return f"{value / 3600.0:.2f} h"


def fmt_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    out = []
    for key_values, group in groups.items():
        totals = [float(r["total_seconds"]) for r in group]
        sps = [float(r["mean_seconds_per_step"]) for r in group]
        tps = [float(r["tokens_per_second"]) for r in group]
        first = group[0]
        item = {name: value for name, value in zip(keys, key_values)}
        item.update(
            {
                "method_label": METHOD_LABEL.get(str(first["method"]), str(first["method"])),
                "activation": first["activation"],
                "optimizer": first["optimizer"],
                "runs": len(group),
                "steps": first["steps"],
                "train_tokens_per_run": first["train_tokens"],
                "total_seconds_mean": mean(totals),
                "total_seconds_std": sample_std(totals),
                "total_seconds_min": min(totals),
                "total_seconds_max": max(totals),
                "total_minutes_mean": mean(totals) / 60.0,
                "total_hours_mean": mean(totals) / 3600.0,
                "mean_seconds_per_step": mean(sps),
                "tokens_per_second_mean": mean(tps),
            }
        )
        out.append(item)
    return sorted(
        out,
        key=lambda r: (
            str(r.get("scope", "")),
            str(r.get("dataset", "")),
            METHOD_ORDER.index(str(r["method"])) if str(r["method"]) in METHOD_ORDER else 999,
        ),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(rows: list[dict[str, object]], scope: str) -> str:
    scoped = [row for row in rows if row["scope"] == scope]
    lines = [
        "| Combo | Runs | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scoped:
        lines.append(
            "| {label} | {runs} | {mean_time} | {std_time} | {min_time}-{max_time} | {sps} | {tps} |".format(
                label=row["method_label"],
                runs=row["runs"],
                mean_time=fmt_minutes(float(row["total_seconds_mean"])),
                std_time=fmt_minutes(float(row["total_seconds_std"])),
                min_time=fmt_minutes(float(row["total_seconds_min"])),
                max_time=fmt_minutes(float(row["total_seconds_max"])),
                sps=fmt_number(float(row["mean_seconds_per_step"]), 4),
                tps=fmt_number(float(row["tokens_per_second_mean"]), 1),
            )
        )
    return "\n".join(lines)


def write_readme(
    output_dir: Path,
    raw_scope_rows: list[dict[str, object]],
    clean_scope_rows: list[dict[str, object]],
    per_row_count: int,
    clean_row_count: int,
) -> None:
    e1_rows = [row for row in raw_scope_rows if row["scope"] == "E1_m0_100m_all_datasets"]
    e1_total = sum(int(row["runs"]) for row in e1_rows)
    e2_totals = {
        scope: sum(int(row["runs"]) for row in raw_scope_rows if row["scope"] == scope)
        for scope in COMPLETED_E2_DATASET_SCOPES.values()
    }
    generated = datetime.now().strftime("%Y-%m-%d")
    e2_bullets = "\n".join(
        f"- {SCOPE_LABEL[scope]} completed cell: `{total}` rows, one dataset x three seeds x 15 methods."
        for scope, total in e2_totals.items()
        if total
    )
    clean_sections = "\n\n".join(
        f"## Clean {SCOPE_LABEL[scope]}\n\n{markdown_table(clean_scope_rows, scope)}"
        for scope in ["E1_m0_100m_all_datasets", *COMPLETED_E2_DATASET_SCOPES.values()]
    )
    raw_sections = "\n\n".join(
        f"## Raw All-Completed {SCOPE_LABEL[scope]}\n\n{markdown_table(raw_scope_rows, scope)}"
        for scope in ["E1_m0_100m_all_datasets", *COMPLETED_E2_DATASET_SCOPES.values()]
    )
    text = f"""# ICLR26 Runtime Summary

Generated: {generated}.

This package summarizes per optimizer/activation-combo runtime from completed JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and other launcher overhead. That is the comparable per-combo runtime because E1 jobs ran whole 15-row cells inside one Slurm allocation.

Included:

- E1 M0/100M all completed datasets: `{e1_total}` rows, five datasets x three seeds x 15 methods.
{e2_bullets}

Excluded:

- E2 FineWeb rows `330-374`, because that dataset cell is queued/incomplete until all 45 rows finish.
- E2 rows `375+`, because they have not been queued/completed yet.

## Runtime Quality Note

E1 FineWeb-Edu seed `2027` rows `75-89` ran as Slurm job `158117`, which completed with `Restarts=6`. The final JSONL files for those rows have one config record, one summary record, and no duplicate train steps, so `summary.total_seconds` is not directly summing archived requeue attempts. However, that matched cell is restart/node contaminated and produces pathological throughput outliers, especially rows `81-89`.

The clean tables below therefore exclude rows `75-89` from E1 runtime aggregates. Raw all-completed tables are still written to CSV for provenance.

Clean rows summarized: `{clean_row_count}`. Raw completed rows summarized: `{per_row_count}`.

{clean_sections}

{raw_sections}

## Files

- `runtime_by_scope_method_clean.csv`: default clean per-combo aggregate.
- `runtime_by_dataset_method_clean.csv`: default clean per-combo aggregate split by dataset.
- `runtime_by_scope_method.csv`: raw all-completed per-combo aggregate.
- `runtime_by_dataset_method.csv`: raw all-completed per-combo aggregate split by dataset.
- `runtime_per_row.csv`: one record per included completed manifest row.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    per_row: list[dict[str, object]] = []
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scope = scope_for(row)
            if scope is None:
                continue
            jsonl_path = args.run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
            summary = read_summary(jsonl_path)
            if summary is None:
                continue
            total_seconds = float(summary["total_seconds"])
            per_row.append(
                {
                    "scope": scope,
                    "phase": row["phase"],
                    "dataset": row["dataset"],
                    "row_index": int(row["row_index"]),
                    "row_id": row["row_id"],
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "method_label": METHOD_LABEL.get(row["method"], row["method"]),
                    "activation": row["activation"],
                    "optimizer": row["optimizer"],
                    "steps": int(summary["steps"]),
                    "completed_steps": int(summary["completed_steps"]),
                    "train_tokens": int(row["train_tokens"]),
                    "total_seconds": total_seconds,
                    "total_minutes": total_seconds / 60.0,
                    "total_hours": total_seconds / 3600.0,
                    "mean_seconds_per_step": float(summary["mean_seconds_per_step"]),
                    "tokens_per_second": float(summary["tokens_per_second"]),
                    "stopped_early": bool(summary.get("stopped_early", False)),
                    "restart_affected": int(row["row_index"]) in RESTART_AFFECTED_ROWS,
                    "include_in_clean_runtime": int(row["row_index"]) not in RESTART_AFFECTED_ROWS,
                    "jsonl": str(jsonl_path),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_row = sorted(per_row, key=lambda r: (str(r["scope"]), str(r["dataset"]), int(r["row_index"])))
    by_scope = aggregate(per_row, ("scope", "method"))
    by_dataset = aggregate(per_row, ("scope", "phase", "dataset", "method"))
    clean_per_row = [row for row in per_row if row["include_in_clean_runtime"]]
    clean_by_scope = aggregate(clean_per_row, ("scope", "method"))
    clean_by_dataset = aggregate(clean_per_row, ("scope", "phase", "dataset", "method"))

    write_csv(
        args.output_dir / "runtime_per_row.csv",
        per_row,
        [
            "scope",
            "phase",
            "dataset",
            "row_index",
            "row_id",
            "seed",
            "method",
            "method_label",
            "activation",
            "optimizer",
            "steps",
            "completed_steps",
            "train_tokens",
            "total_seconds",
            "total_minutes",
            "total_hours",
            "mean_seconds_per_step",
            "tokens_per_second",
            "stopped_early",
            "restart_affected",
            "include_in_clean_runtime",
            "jsonl",
        ],
    )
    aggregate_fields = [
        "scope",
        "phase",
        "dataset",
        "method",
        "method_label",
        "activation",
        "optimizer",
        "runs",
        "steps",
        "train_tokens_per_run",
        "total_seconds_mean",
        "total_seconds_std",
        "total_seconds_min",
        "total_seconds_max",
        "total_minutes_mean",
        "total_hours_mean",
        "mean_seconds_per_step",
        "tokens_per_second_mean",
    ]
    write_csv(args.output_dir / "runtime_by_scope_method.csv", by_scope, [f for f in aggregate_fields if f not in {"phase", "dataset"}])
    write_csv(args.output_dir / "runtime_by_dataset_method.csv", by_dataset, aggregate_fields)
    write_csv(args.output_dir / "runtime_by_scope_method_clean.csv", clean_by_scope, [f for f in aggregate_fields if f not in {"phase", "dataset"}])
    write_csv(args.output_dir / "runtime_by_dataset_method_clean.csv", clean_by_dataset, aggregate_fields)
    write_readme(args.output_dir, by_scope, clean_by_scope, len(per_row), len(clean_per_row))


if __name__ == "__main__":
    main()
