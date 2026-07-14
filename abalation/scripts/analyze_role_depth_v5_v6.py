#!/usr/bin/env python3
"""Compare completed MatrixPolicy role/depth V5/V6 ablations.

The script is intentionally strict: a row is included only when the JSONL has
one config, one summary, no duplicate train/eval steps, a completed summary, and
a final eval at the manifest step.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


DATASETS = ["dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"]
DATASET_LABELS = {
    "dclm": "DCLM",
    "fineweb_edu": "FineWeb-Edu",
    "fineweb": "FineWeb",
    "dolma_sample": "Dolma-sample",
    "c4_en": "C4",
}
TARGETS = {
    100_000_000: {
        "dclm": [4.90, 4.70, 4.55, 4.45, 4.35, 4.30],
        "fineweb_edu": [4.80, 4.60, 4.40, 4.30, 4.20, 4.10],
        "fineweb": [5.00, 4.80, 4.60, 4.50, 4.40, 4.35],
        "dolma_sample": [5.00, 4.80, 4.60, 4.50, 4.40, 4.35],
        "c4_en": [5.00, 4.80, 4.60, 4.50, 4.40, 4.30],
    },
    300_000_000: {
        "dclm": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
        "fineweb_edu": [4.20, 4.10, 4.00, 3.90, 3.85, 3.80, 3.75],
        "fineweb": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
        "dolma_sample": [4.20, 4.10, 4.00, 3.95, 3.90, 3.85, 3.82],
        "c4_en": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
    },
}
SOURCES = [
    (
        "full",
        Path("experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv"),
        Path("experiments/runs/iclr26_main"),
        {"rlb_matrixpolicy_original"},
    ),
    (
        "no_role_depth",
        Path("abalation/manifests/matrixpolicy_ablation_e1_e2_manifest.csv"),
        Path("abalation/runs/matrixpolicy_ablation_e1_e2"),
        {"rlb_matrixpolicy_no_role_depth"},
    ),
    (
        "role_depth_v2",
        Path("abalation/manifests/role_depth_v2_e1_e2_manifest.csv"),
        Path("abalation/runs/role_depth_v2_e1_e2"),
        {"rlb_matrixpolicy_role_depth_v2"},
    ),
    (
        "role_depth_v3",
        Path("abalation/manifests/role_depth_v3_v4_e1_e2_manifest.csv"),
        Path("abalation/runs/role_depth_v3_v4_e1_e2"),
        {"rlb_matrixpolicy_role_depth_v3"},
    ),
    (
        "role_depth_v4",
        Path("abalation/manifests/role_depth_v3_v4_e1_e2_manifest.csv"),
        Path("abalation/runs/role_depth_v3_v4_e1_e2"),
        {"rlb_matrixpolicy_role_depth_v4"},
    ),
    (
        "role_depth_v5",
        Path("abalation/manifests/role_depth_v5_v6_e1_e2_manifest.csv"),
        Path("abalation/runs/role_depth_v5_v6_e1_e2"),
        {"rlb_matrixpolicy_role_depth_v5"},
    ),
    (
        "role_depth_v6",
        Path("abalation/manifests/role_depth_v5_v6_e1_e2_manifest.csv"),
        Path("abalation/runs/role_depth_v5_v6_e1_e2"),
        {"rlb_matrixpolicy_role_depth_v6"},
    ),
]
METHOD_ORDER = ["full", "no_role_depth", "role_depth_v2", "role_depth_v3", "role_depth_v4", "role_depth_v5", "role_depth_v6"]
OUT_DIR = Path("abalation/results/role_depth_v5_v6_comparison")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_jsonl(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None, list[str]]:
    evals: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    config_count = 0
    train_steps: list[int] = []
    eval_steps: list[int] = []
    timing_guard_failed = False
    issues: list[str] = []
    if not path.exists():
        return evals, None, ["missing_jsonl"]
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                issues.append("json_decode_error")
                continue
            event = record.get("event")
            if event == "config":
                config_count += 1
            elif event == "train":
                if record.get("step") is not None:
                    train_steps.append(int(record["step"]))
                loss = safe_float(record.get("loss"))
                if not math.isfinite(loss):
                    issues.append(f"nonfinite_train_loss_step_{record.get('step')}")
            elif event == "eval":
                evals.append(record)
                if record.get("step") is not None:
                    eval_steps.append(int(record["step"]))
                loss = safe_float(record.get("val_loss"))
                if not math.isfinite(loss):
                    issues.append(f"nonfinite_val_loss_step_{record.get('step')}")
            elif event == "summary":
                summaries.append(record)
            elif event == "timing_guard_failed":
                timing_guard_failed = True
    if config_count != 1:
        issues.append(f"config_count={config_count}")
    if len(summaries) != 1:
        issues.append(f"summary_count={len(summaries)}")
    if len(train_steps) != len(set(train_steps)):
        issues.append("duplicate_train_steps")
    if len(eval_steps) != len(set(eval_steps)):
        issues.append("duplicate_eval_steps")
    if timing_guard_failed:
        issues.append("timing_guard_failed")
    return sorted(evals, key=lambda item: int(item.get("step", -1))), (summaries[-1] if summaries else None), issues


def load_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for label, manifest, run_root, wanted_methods in SOURCES:
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] not in DATASETS or row["method"] not in wanted_methods:
                    continue
                steps = int(row["steps"])
                train_tokens = int(row["train_tokens"])
                jsonl = run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
                evals, summary, issues = read_jsonl(jsonl)
                final_eval = next((rec for rec in evals if int(rec.get("step", -1)) == steps), None)
                if summary is None:
                    issues.append("missing_summary")
                else:
                    if int(summary.get("steps", -1)) != steps:
                        issues.append("summary_steps_mismatch")
                    if int(summary.get("completed_steps", -1)) != steps:
                        issues.append("summary_completed_steps_mismatch")
                if final_eval is None:
                    issues.append("missing_final_eval")
                if issues:
                    skipped.append(
                        {
                            "source": label,
                            "dataset": row["dataset"],
                            "seed": row["seed"],
                            "method": row["method"],
                            "train_tokens": train_tokens,
                            "jsonl": str(jsonl),
                            "issues": ";".join(dict.fromkeys(issues)),
                        }
                    )
                    continue
                rows.append(
                    {
                        "source": label,
                        "dataset": row["dataset"],
                        "seed": int(row["seed"]),
                        "method": row["method"],
                        "train_tokens": train_tokens,
                        "steps": steps,
                        "global_tokens_per_step": int(row["global_tokens_per_step"]),
                        "evals": evals,
                        "summary": summary,
                        "final_loss": safe_float(final_eval.get("val_loss")),
                        "final_ppl": safe_float(final_eval.get("val_ppl")),
                        "total_seconds": safe_float(summary.get("total_seconds")),
                        "mean_seconds_per_step": safe_float(summary.get("mean_seconds_per_step")),
                        "tokens_per_second": safe_float(summary.get("tokens_per_second")),
                        "slurm_job_id": summary.get("slurm_job_id"),
                        "slurm_restart_count": summary.get("slurm_restart_count"),
                        "slurm_node": summary.get("slurm_node"),
                        "jsonl": str(jsonl),
                    }
                )
    return rows, skipped


def first_hit(row: dict[str, object], target: float) -> tuple[int | None, int | None, float | None]:
    tokens_per_step = int(row["global_tokens_per_step"])
    for rec in row["evals"]:  # type: ignore[index]
        step = int(rec.get("step", -1))
        loss = safe_float(rec.get("val_loss"))
        if step > 0 and math.isfinite(loss) and loss <= target:
            return step, step * tokens_per_step, safe_float(rec.get("val_ppl"))
    return None, None, None


def target_hit_minutes(row: dict[str, object], step: int | None) -> float | None:
    if step is None:
        return None
    return step * float(row["mean_seconds_per_step"]) / 60.0


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fmt_mtokens(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value) / 1_000_000:.2f}M"


def fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.2f}%"


def fmt_opt_float(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ""
    value_f = float(value)
    if not math.isfinite(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def main() -> None:
    rows, skipped = load_rows()
    by_key = {(row["train_tokens"], row["dataset"], row["seed"], row["source"]): row for row in rows}
    coverage_rows: list[dict[str, object]] = []
    for row in rows:
        coverage_rows.append(
            {
                "status": "valid",
                "budget_tokens": row["train_tokens"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method": row["source"],
                "jsonl": row["jsonl"],
                "issues": "",
                "slurm_job_id": row["slurm_job_id"],
                "slurm_restart_count": row["slurm_restart_count"],
                "slurm_node": row["slurm_node"],
            }
        )
    for row in skipped:
        coverage_rows.append(
            {
                "status": "skipped",
                "budget_tokens": row["train_tokens"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "method": row["source"],
                "jsonl": row["jsonl"],
                "issues": row["issues"],
                "slurm_job_id": "",
                "slurm_restart_count": "",
                "slurm_node": "",
            }
        )
    final_rows = [
        {
            "budget_tokens": row["train_tokens"],
            "dataset": row["dataset"],
            "seed": row["seed"],
            "method": row["source"],
            "final_val_loss": f"{float(row['final_loss']):.6f}",
            "final_val_ppl": f"{float(row['final_ppl']):.3f}",
            "total_minutes": f"{float(row['total_seconds']) / 60.0:.3f}",
            "mean_seconds_per_step": f"{float(row['mean_seconds_per_step']):.6f}",
            "tokens_per_second": f"{float(row['tokens_per_second']):.3f}",
            "slurm_job_id": row["slurm_job_id"],
            "slurm_restart_count": row["slurm_restart_count"],
            "slurm_node": row["slurm_node"],
            "jsonl": row["jsonl"],
        }
        for row in sorted(rows, key=lambda item: (int(item["train_tokens"]), str(item["dataset"]), int(item["seed"]), METHOD_ORDER.index(str(item["source"]))))
    ]
    target_rows: list[dict[str, object]] = []
    savings_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for budget, target_map in TARGETS.items():
        for dataset in DATASETS:
            for seed in sorted({int(row["seed"]) for row in rows if row["train_tokens"] == budget and row["dataset"] == dataset}):
                for method in METHOD_ORDER:
                    row = by_key.get((budget, dataset, seed, method))
                    if row is None:
                        continue
                    for target in target_map[dataset]:
                        step, tokens, ppl = first_hit(row, target)
                        hit_minutes = target_hit_minutes(row, step)
                        target_rows.append(
                            {
                                "budget_tokens": budget,
                                "dataset": dataset,
                                "seed": seed,
                                "method": method,
                                "target_loss": f"{target:.2f}",
                                "first_hit_step": step,
                                "first_hit_tokens": tokens,
                                "first_hit_ppl": "" if ppl is None else f"{ppl:.3f}",
                                "hit_status": "hit" if tokens is not None else "not_hit",
                                "target_hit_minutes": fmt_opt_float(hit_minutes, 3),
                                "total_minutes": f"{float(row['total_seconds']) / 60.0:.3f}",
                                "mean_seconds_per_step": f"{float(row['mean_seconds_per_step']):.6f}",
                                "tokens_per_second": f"{float(row['tokens_per_second']):.3f}",
                                "final_val_loss": f"{float(row['final_loss']):.6f}",
                                "slurm_job_id": row["slurm_job_id"],
                                "slurm_restart_count": row["slurm_restart_count"],
                                "slurm_node": row["slurm_node"],
                            }
                        )
                for target in target_map[dataset]:
                    full_row = by_key.get((budget, dataset, seed, "full"))
                    if full_row is None:
                        continue
                    full_step, full_tokens, _ = first_hit(full_row, target)
                    if full_tokens is None:
                        continue
                    for method in METHOD_ORDER:
                        if method == "full":
                            continue
                        row = by_key.get((budget, dataset, seed, method))
                        if row is None:
                            continue
                        step, tokens, _ = first_hit(row, target)
                        if tokens is None:
                            continue
                        saved = tokens - full_tokens
                        savings_rows.append(
                            {
                                "budget_tokens": budget,
                                "dataset": dataset,
                                "seed": seed,
                                "target_loss": f"{target:.2f}",
                                "comparator": method,
                                "full_tokens": full_tokens,
                                "comparator_tokens": tokens,
                                "full_saves_tokens": saved,
                                "full_saves_percent_vs_comparator": saved / tokens if tokens else "",
                                "full_total_minutes": f"{float(full_row['total_seconds']) / 60.0:.3f}",
                                "comparator_total_minutes": f"{float(row['total_seconds']) / 60.0:.3f}",
                            }
                        )
                    for baseline in ("full", "no_role_depth"):
                        baseline_row = by_key.get((budget, dataset, seed, baseline))
                        if baseline_row is None:
                            continue
                        baseline_step, baseline_tokens, baseline_ppl = first_hit(baseline_row, target)
                        baseline_hit_minutes = target_hit_minutes(baseline_row, baseline_step)
                        for method in METHOD_ORDER:
                            if method == baseline:
                                continue
                            row = by_key.get((budget, dataset, seed, method))
                            if row is None:
                                continue
                            method_step, method_tokens, method_ppl = first_hit(row, target)
                            method_hit_minutes = target_hit_minutes(row, method_step)
                            common_hit = baseline_tokens is not None and method_tokens is not None
                            delta_tokens = (method_tokens - baseline_tokens) if common_hit else None
                            delta_steps = (method_step - baseline_step) if common_hit else None
                            delta_hit_minutes = (
                                method_hit_minutes - baseline_hit_minutes
                                if method_hit_minutes is not None and baseline_hit_minutes is not None
                                else None
                            )
                            baseline_total_minutes = float(baseline_row["total_seconds"]) / 60.0
                            method_total_minutes = float(row["total_seconds"]) / 60.0
                            paired_rows.append(
                                {
                                    "budget_tokens": budget,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "target_loss": f"{target:.2f}",
                                    "baseline": baseline,
                                    "method": method,
                                    "baseline_hit": int(baseline_tokens is not None),
                                    "method_hit": int(method_tokens is not None),
                                    "baseline_first_hit_step": baseline_step,
                                    "method_first_hit_step": method_step,
                                    "baseline_first_hit_tokens": baseline_tokens,
                                    "method_first_hit_tokens": method_tokens,
                                    "baseline_first_hit_ppl": "" if baseline_ppl is None else f"{baseline_ppl:.3f}",
                                    "method_first_hit_ppl": "" if method_ppl is None else f"{method_ppl:.3f}",
                                    "baseline_target_hit_minutes": fmt_opt_float(baseline_hit_minutes, 3),
                                    "method_target_hit_minutes": fmt_opt_float(method_hit_minutes, 3),
                                    "delta_method_minus_baseline_tokens": delta_tokens,
                                    "delta_method_minus_baseline_steps": delta_steps,
                                    "delta_method_minus_baseline_tokens_percent": (
                                        "" if not common_hit or baseline_tokens == 0 else f"{delta_tokens / baseline_tokens:.6f}"
                                    ),
                                    "delta_method_minus_baseline_hit_minutes": fmt_opt_float(delta_hit_minutes, 3),
                                    "baseline_total_minutes": f"{baseline_total_minutes:.3f}",
                                    "method_total_minutes": f"{method_total_minutes:.3f}",
                                    "delta_method_minus_baseline_total_minutes": f"{method_total_minutes - baseline_total_minutes:.3f}",
                                    "baseline_final_val_loss": f"{float(baseline_row['final_loss']):.6f}",
                                    "method_final_val_loss": f"{float(row['final_loss']):.6f}",
                                    "delta_method_minus_baseline_final_loss": f"{float(row['final_loss']) - float(baseline_row['final_loss']):.6f}",
                                    "baseline_slurm_node": baseline_row["slurm_node"],
                                    "method_slurm_node": row["slurm_node"],
                                    "baseline_restart_count": baseline_row["slurm_restart_count"],
                                    "method_restart_count": row["slurm_restart_count"],
                                }
                            )

    aggregate_rows: list[dict[str, object]] = []
    paired_grouped: dict[tuple[int, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in paired_rows:
        paired_grouped[
            (
                int(row["budget_tokens"]),
                str(row["dataset"]),
                str(row["target_loss"]),
                str(row["baseline"]),
                str(row["method"]),
            )
        ].append(row)
    for key in sorted(paired_grouped):
        budget, dataset, target_loss, baseline, method = key
        vals = paired_grouped[key]
        common = [
            row
            for row in vals
            if row["baseline_hit"] == 1 and row["method_hit"] == 1
        ]
        delta_tokens = [float(row["delta_method_minus_baseline_tokens"]) for row in common]
        delta_hit_minutes = [
            float(row["delta_method_minus_baseline_hit_minutes"])
            for row in common
            if row["delta_method_minus_baseline_hit_minutes"] != ""
        ]
        delta_final = [float(row["delta_method_minus_baseline_final_loss"]) for row in vals]
        wins = sum(1 for value in delta_tokens if value < 0)
        ties = sum(1 for value in delta_tokens if value == 0)
        losses = sum(1 for value in delta_tokens if value > 0)
        sorted_delta = sorted(delta_tokens)
        median_delta = ""
        if sorted_delta:
            mid = len(sorted_delta) // 2
            if len(sorted_delta) % 2:
                median_delta = f"{sorted_delta[mid]:.3f}"
            else:
                median_delta = f"{(sorted_delta[mid - 1] + sorted_delta[mid]) / 2.0:.3f}"
        aggregate_rows.append(
            {
                "budget_tokens": budget,
                "dataset": dataset,
                "target_loss": target_loss,
                "baseline": baseline,
                "method": method,
                "n_pairs": len(vals),
                "common_hits": len(common),
                "baseline_only_hits": sum(1 for row in vals if row["baseline_hit"] == 1 and row["method_hit"] == 0),
                "method_only_hits": sum(1 for row in vals if row["baseline_hit"] == 0 and row["method_hit"] == 1),
                "both_missing": sum(1 for row in vals if row["baseline_hit"] == 0 and row["method_hit"] == 0),
                "wins_method_earlier": wins,
                "ties_same_tokens": ties,
                "losses_method_later": losses,
                "mean_delta_method_minus_baseline_tokens": "" if not delta_tokens else f"{mean(delta_tokens):.3f}",
                "median_delta_method_minus_baseline_tokens": median_delta,
                "mean_delta_method_minus_baseline_hit_minutes": "" if not delta_hit_minutes else f"{mean(delta_hit_minutes):.3f}",
                "mean_delta_method_minus_baseline_final_loss": "" if not delta_final else f"{mean(delta_final):.6f}",
            }
        )

    write_csv(
        OUT_DIR / "final_loss_runtime.csv",
        final_rows,
        [
            "budget_tokens",
            "dataset",
            "seed",
            "method",
            "final_val_loss",
            "final_val_ppl",
            "total_minutes",
            "mean_seconds_per_step",
            "tokens_per_second",
            "slurm_job_id",
            "slurm_restart_count",
            "slurm_node",
            "jsonl",
        ],
    )
    write_csv(
        OUT_DIR / "token_to_target_by_seed.csv",
        target_rows,
        [
            "budget_tokens",
            "dataset",
            "seed",
            "method",
            "target_loss",
            "first_hit_step",
            "first_hit_tokens",
            "first_hit_ppl",
            "hit_status",
            "target_hit_minutes",
            "total_minutes",
            "mean_seconds_per_step",
            "tokens_per_second",
            "final_val_loss",
            "slurm_job_id",
            "slurm_restart_count",
            "slurm_node",
        ],
    )
    write_csv(
        OUT_DIR / "full_vs_ablation_token_savings_by_seed.csv",
        savings_rows,
        [
            "budget_tokens",
            "dataset",
            "seed",
            "target_loss",
            "comparator",
            "full_tokens",
            "comparator_tokens",
            "full_saves_tokens",
            "full_saves_percent_vs_comparator",
            "full_total_minutes",
            "comparator_total_minutes",
        ],
    )
    write_csv(
        OUT_DIR / "paired_method_vs_baseline_by_seed.csv",
        paired_rows,
        [
            "budget_tokens",
            "dataset",
            "seed",
            "target_loss",
            "baseline",
            "method",
            "baseline_hit",
            "method_hit",
            "baseline_first_hit_step",
            "method_first_hit_step",
            "baseline_first_hit_tokens",
            "method_first_hit_tokens",
            "baseline_first_hit_ppl",
            "method_first_hit_ppl",
            "baseline_target_hit_minutes",
            "method_target_hit_minutes",
            "delta_method_minus_baseline_tokens",
            "delta_method_minus_baseline_steps",
            "delta_method_minus_baseline_tokens_percent",
            "delta_method_minus_baseline_hit_minutes",
            "baseline_total_minutes",
            "method_total_minutes",
            "delta_method_minus_baseline_total_minutes",
            "baseline_final_val_loss",
            "method_final_val_loss",
            "delta_method_minus_baseline_final_loss",
            "baseline_slurm_node",
            "method_slurm_node",
            "baseline_restart_count",
            "method_restart_count",
        ],
    )
    write_csv(
        OUT_DIR / "aggregate_method_vs_baseline_by_target.csv",
        aggregate_rows,
        [
            "budget_tokens",
            "dataset",
            "target_loss",
            "baseline",
            "method",
            "n_pairs",
            "common_hits",
            "baseline_only_hits",
            "method_only_hits",
            "both_missing",
            "wins_method_earlier",
            "ties_same_tokens",
            "losses_method_later",
            "mean_delta_method_minus_baseline_tokens",
            "median_delta_method_minus_baseline_tokens",
            "mean_delta_method_minus_baseline_hit_minutes",
            "mean_delta_method_minus_baseline_final_loss",
        ],
    )
    write_csv(
        OUT_DIR / "coverage_validity_by_run.csv",
        coverage_rows,
        [
            "status",
            "budget_tokens",
            "dataset",
            "seed",
            "method",
            "jsonl",
            "issues",
            "slurm_job_id",
            "slurm_restart_count",
            "slurm_node",
        ],
    )
    write_csv(
        OUT_DIR / "skipped_incomplete_or_invalid.csv",
        skipped,
        ["source", "dataset", "seed", "method", "train_tokens", "jsonl", "issues"],
    )

    print(f"completed_rows={len(rows)} skipped_rows={len(skipped)}")
    by_budget_source = defaultdict(int)
    for row in rows:
        by_budget_source[(row["train_tokens"], row["source"])] += 1
    for budget in sorted({int(row["train_tokens"]) for row in rows} | {100_000_000, 300_000_000}):
        parts = [f"{method}={by_budget_source[(budget, method)]}" for method in METHOD_ORDER]
        print(f"{budget // 1_000_000}M " + " ".join(parts))

    print("\nMean final loss and runtime:")
    for budget in (100_000_000, 300_000_000):
        for method in METHOD_ORDER:
            vals = [row for row in rows if row["train_tokens"] == budget and row["source"] == method]
            if not vals:
                continue
            print(
                f"{budget // 1_000_000}M {method}: "
                f"loss={mean(float(row['final_loss']) for row in vals):.6f} "
                f"time={mean(float(row['total_seconds']) for row in vals) / 60.0:.2f}m "
                f"n={len(vals)}"
            )

    print("\nFull MatrixPolicy token savings vs ablations, common seed-target hits:")
    grouped: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in savings_rows:
        grouped[(int(row["budget_tokens"]), str(row["dataset"]), str(row["comparator"]))].append(row)
    for key in sorted(grouped):
        budget, dataset, comparator = key
        vals = grouped[key]
        saved = [float(row["full_saves_tokens"]) for row in vals]
        pct = [float(row["full_saves_percent_vs_comparator"]) for row in vals if row["full_saves_percent_vs_comparator"] != ""]
        print(
            f"{budget // 1_000_000}M {DATASET_LABELS[dataset]} vs {comparator}: "
            f"saved={fmt_mtokens(mean(saved))} pct={fmt_pct(mean(pct) if pct else math.nan)} n={len(vals)}"
        )
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
