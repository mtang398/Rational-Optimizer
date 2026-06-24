#!/usr/bin/env python3
"""Summarize clean completed ICLR run runtimes from JSONL summary records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


DEFAULT_OUTPUT = Path("experiments/results/iclr26_runtime_summary_2026_06_11")
MANIFEST = Path("experiments/manifests/iclr26_main_manifest.csv")
RUN_ROOT = Path("experiments/runs/iclr26_main")
SAFE_E1_MATRIXPOLICY_MANIFEST = Path("experiments/manifests/iclr26_matrixpolicy_safe_speed_e1_manifest.csv")
SAFE_E2_MATRIXPOLICY_MANIFEST = Path("experiments/manifests/iclr26_matrixpolicy_safe_speed_e2_manifest.csv")
E1_RESTART_REPAIR_MANIFEST = Path("experiments/manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv")
MATRIXPOLICY_METHOD = "rlb_matrixpolicy_original"

# E1 FineWeb-Edu seed 2027 ran as Slurm job 158117 with Restarts=6.
# Rows 75-80 completed before the final restart and match adjacent seed timing.
# Rows 81-88 cannot provide trusted per-row runtime from the original artifacts:
# sacct --duplicates shows six preempted allocations and partial JSONLs for row
# 81, so the original row summaries are not clean timing records. They are
# skipped from the main manifest and overlaid from the completed clean repair
# manifest. Row 89 is replaced by the safe-speed MatrixPolicy rerun through the
# method-key overlay below.
RESTART_AFFECTED_ROWS = set(range(81, 89))


COMPLETED_E2_DATASET_SCOPES = {
    "dclm": "E2_m0_300m_dclm",
    "fineweb_edu": "E2_m0_300m_fineweb_edu",
    "fineweb": "E2_m0_300m_fineweb",
    "dolma_sample": "E2_m0_300m_dolma_sample",
    "c4_en": "E2_m0_300m_c4_en",
}

SCOPE_LABEL = {
    "E1_m0_100m_all_datasets": "E1 M0/100M All Datasets",
    "E2_m0_300m_dclm": "E2 M0/300M DCLM",
    "E2_m0_300m_fineweb_edu": "E2 M0/300M FineWeb-Edu",
    "E2_m0_300m_fineweb": "E2 M0/300M FineWeb",
    "E2_m0_300m_dolma_sample": "E2 M0/300M Dolma-sample",
    "E2_m0_300m_c4_en": "E2 M0/300M C4",
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
    parser.add_argument("--safe-e1-matrixpolicy-manifest", type=Path, default=SAFE_E1_MATRIXPOLICY_MANIFEST)
    parser.add_argument("--safe-e2-matrixpolicy-manifest", type=Path, default=SAFE_E2_MATRIXPOLICY_MANIFEST)
    parser.add_argument("--e1-restart-repair-manifest", type=Path, default=E1_RESTART_REPAIR_MANIFEST)
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


def fmt_minutes(value: float) -> str:
    return f"{value / 60.0:.1f} min"


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
    scope_rows: list[dict[str, object]],
    clean_row_count: int,
    skipped_original_row_count: int,
    repair_overlay_count: int,
) -> None:
    e1_total = sum(int(row["runs"]) for row in scope_rows if row["scope"] == "E1_m0_100m_all_datasets")
    e2_totals = {
        scope: sum(int(row["runs"]) for row in scope_rows if row["scope"] == scope)
        for scope in COMPLETED_E2_DATASET_SCOPES.values()
    }
    generated = datetime.now().strftime("%Y-%m-%d")
    e2_bullets = "\n".join(
        f"- {SCOPE_LABEL[scope]} completed cell: `{total}` rows, one dataset x three seeds x 15 methods."
        for scope, total in e2_totals.items()
        if total
    )
    sections = "\n\n".join(
        f"## {SCOPE_LABEL[scope]}\n\n{markdown_table(scope_rows, scope)}"
        for scope in ["E1_m0_100m_all_datasets", *COMPLETED_E2_DATASET_SCOPES.values()]
    )
    repair_line = (
        f"Completed clean repair overlay rows for E1 FineWeb-Edu seed `2027` rows `81-88`: `{repair_overlay_count}/8`."
    )
    text = f"""# ICLR26 Runtime Summary

Generated: {generated}.

This package summarizes clean per optimizer/activation-combo runtime from completed JSONL `summary` records. The runtime field is `summary.total_seconds`, i.e. training-harness wall time for a manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, launcher overhead, and pre-restart partial attempts.

Included in tracked runtime aggregates:

- E1 M0/100M clean rows: `{e1_total}` rows. E1 FineWeb-Edu seed `2027` job `158117` had `Restarts=6`; rows `75-80` are retained because their completed JSONL timings match adjacent seeds. Original rows `81-88` are skipped because the existing artifacts cannot reconstruct trusted per-row runtime after multiple preempted allocations and partial JSONLs. {repair_line} Row `89` is replaced by the completed safe-speed MatrixPolicy rerun when available.
{e2_bullets}

Excluded from tracked runtime aggregates:

- Original E1 FineWeb-Edu seed `2027` rows `81-88`: `{skipped_original_row_count}` rows skipped from the main manifest runtime source. They are overlaid from the completed clean repair manifest `experiments/manifests/iclr26_e1_fineweb_edu_seed2027_runtime_repair_manifest.csv`.
- Rows `465+` are outside E2.

No raw Slurm-elapsed E1 aggregate is tracked in this package. Runtime aggregates use completed JSONL `summary.total_seconds` only for clean row attempts; original restart-contaminated rows `81-88` are not assigned inferred row times.

Clean rows summarized: `{clean_row_count}`.

{sections}

## Files

- `runtime_by_scope_method_clean.csv`: clean per-combo aggregate.
- `runtime_by_dataset_method_clean.csv`: clean per-combo aggregate split by dataset.
- `runtime_per_row.csv`: one record per clean included manifest row.
"""
    (output_dir / "README.md").write_text(text)


def override_scope_for(row: dict[str, str]) -> str | None:
    phase = row["phase"]
    dataset = row["dataset"]
    if phase in {"E1_matrixpolicy_safe_speed_100m", "E1_fineweb_edu_seed2027_runtime_repair_100m"}:
        return "E1_m0_100m_all_datasets"
    if phase == "E2_matrixpolicy_safe_speed_300m" and dataset in COMPLETED_E2_DATASET_SCOPES:
        return COMPLETED_E2_DATASET_SCOPES[dataset]
    return None


def runtime_row_from_manifest(row: dict[str, str], scope: str, run_root: Path) -> dict[str, object] | None:
    jsonl_path = run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
    summary = read_summary(jsonl_path)
    if summary is None:
        return None
    total_seconds = float(summary["total_seconds"])
    return {
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
        "jsonl": str(jsonl_path),
    }


def overlay_completed_matrixpolicy_rows(
    per_row_by_key: dict[tuple[str, str, int, str], dict[str, object]],
    manifest: Path,
    run_root: Path,
) -> int:
    if not manifest.exists():
        return 0
    count = 0
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != MATRIXPOLICY_METHOD:
                continue
            scope = override_scope_for(row)
            if scope is None:
                continue
            item = runtime_row_from_manifest(row, scope, run_root)
            if item is None:
                continue
            key = (scope, str(item["dataset"]), int(item["seed"]), str(item["method"]))
            per_row_by_key[key] = item
            count += 1
    return count


def overlay_completed_repair_rows(
    per_row_by_key: dict[tuple[str, str, int, str], dict[str, object]],
    manifest: Path,
    run_root: Path,
) -> int:
    if not manifest.exists():
        return 0
    count = 0
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scope = override_scope_for(row)
            if scope is None:
                continue
            item = runtime_row_from_manifest(row, scope, run_root)
            if item is None:
                continue
            key = (scope, str(item["dataset"]), int(item["seed"]), str(item["method"]))
            per_row_by_key[key] = item
            count += 1
    return count


def main() -> None:
    args = parse_args()
    per_row_by_key: dict[tuple[str, str, int, str], dict[str, object]] = {}
    skipped_original_rows = 0
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scope = scope_for(row)
            if scope is None:
                continue
            row_index = int(row["row_index"])
            if row_index in RESTART_AFFECTED_ROWS:
                skipped_original_rows += 1
                continue
            item = runtime_row_from_manifest(row, scope, args.run_root)
            if item is None:
                continue
            key = (scope, str(item["dataset"]), int(item["seed"]), str(item["method"]))
            per_row_by_key[key] = item

    overlay_completed_matrixpolicy_rows(per_row_by_key, args.safe_e1_matrixpolicy_manifest, args.run_root)
    overlay_completed_matrixpolicy_rows(per_row_by_key, args.safe_e2_matrixpolicy_manifest, args.run_root)
    repair_overlay_count = overlay_completed_repair_rows(per_row_by_key, args.e1_restart_repair_manifest, args.run_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_row = sorted(per_row_by_key.values(), key=lambda r: (str(r["scope"]), str(r["dataset"]), int(r["row_index"]), str(r["method"])))
    by_scope = aggregate(per_row, ("scope", "method"))
    by_dataset = aggregate(per_row, ("scope", "phase", "dataset", "method"))

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
    write_csv(args.output_dir / "runtime_by_scope_method_clean.csv", by_scope, [f for f in aggregate_fields if f not in {"phase", "dataset"}])
    write_csv(args.output_dir / "runtime_by_dataset_method_clean.csv", by_dataset, aggregate_fields)
    write_readme(args.output_dir, by_scope, len(per_row), skipped_original_rows, repair_overlay_count)

if __name__ == "__main__":
    main()
