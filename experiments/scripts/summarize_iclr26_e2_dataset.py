#!/usr/bin/env python3
"""Summarize one completed ICLR26 E2 M0/300M dataset cell."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


MANIFEST = Path("experiments/manifests/iclr26_main_manifest.csv")
RUN_ROOT = Path("experiments/runs/iclr26_main")
TIMING_NODE_OVERRIDES = Path("experiments/manifests/iclr26_timing_node_overrides.csv")
PHASE = "E2_m0_300m"
MATRIXPOLICY_METHOD = "rlb_matrixpolicy_original"
REPLACEMENT_RLB_METHODS = {
    "rlb_adamw",
    "rlb_lion",
    "rlb_soap",
    "rlb_muon",
    "rlb_schedulefree",
    "rlb_came",
    "rlb_ademamix",
}
ADAMW_METHOD = "silu_adamw"

DEFAULT_TARGETS = {
    "dclm": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
    "fineweb_edu": [4.20, 4.10, 4.00, 3.90, 3.85, 3.80, 3.75],
    "fineweb": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
    "dolma_sample": [4.20, 4.10, 4.00, 3.95, 3.90, 3.85, 3.82],
    "c4_en": [4.40, 4.30, 4.20, 4.10, 4.05, 4.00],
}

DATASET_LABEL = {
    "dclm": "DCLM",
    "fineweb_edu": "FineWeb-Edu",
    "fineweb": "FineWeb",
    "dolma_sample": "Dolma-sample",
    "c4_en": "C4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Manifest dataset id, e.g. dclm or fineweb_edu.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--completed-date", default="2026-06-12")
    parser.add_argument("--targets", type=float, nargs="*", default=None)
    parser.add_argument("--matrixpolicy-manifest", type=Path, default=None)
    parser.add_argument("--matrixpolicy-phase", type=str, default=None)
    parser.add_argument("--replacement-manifest", type=Path, default=None)
    parser.add_argument("--replacement-phase", type=str, default=None)
    parser.add_argument("--timing-node-overrides", type=Path, default=TIMING_NODE_OVERRIDES)
    parser.add_argument("--matrixpolicy-max-seconds-per-step", type=float, default=0.0)
    parser.add_argument("--matrixpolicy-denylist-nodes", default="sablab-gpu-12")
    parser.add_argument("--allow-timing-anomalies", action="store_true")
    return parser.parse_args()


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return stdev(values)


def fmt_float(value: float, digits: int = 6) -> str:
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return "nan"


def fmt_tokens(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "not reached"
    return f"{value / 1_000_000:.1f}M"


def fmt_percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_timing_node_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle) if row.get("row_id")}


def apply_timing_node_override(
    row: dict[str, str],
    summary: dict[str, object] | None,
    overrides: dict[str, dict[str, str]],
) -> dict[str, object] | None:
    if summary is None or summary.get("slurm_node"):
        return summary
    override = overrides.get(row.get("row_id", ""))
    if not override:
        return summary
    updated = dict(summary)
    updated["slurm_job_id"] = override.get("slurm_job_id", updated.get("slurm_job_id", ""))
    updated["slurm_restart_count"] = override.get("slurm_restarts", updated.get("slurm_restart_count", ""))
    updated["slurm_node"] = override.get("slurm_node", updated.get("slurm_node", ""))
    updated["timing_node_override_reason"] = override.get("reason", "")
    return updated


def read_jsonl(path: Path) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    evals: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    config_count = 0
    timing_guard_failed = False
    train_steps: list[int] = []
    eval_steps: list[int] = []
    if not path.exists():
        return evals, None
    with path.open("r", errors="replace") as handle:
        for raw in handle:
            if not raw.startswith("{"):
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "config":
                config_count += 1
            elif event == "eval":
                evals.append(record)
                if record.get("step") is not None:
                    eval_steps.append(int(record["step"]))
            elif event == "summary":
                summaries.append(record)
            elif event == "timing_guard_failed":
                timing_guard_failed = True
            elif event == "train" and record.get("step") is not None:
                train_steps.append(int(record["step"]))
    if not summaries:
        return evals, None
    summary = dict(summaries[-1])
    summary["_jsonl_config_count"] = config_count
    summary["_jsonl_summary_count"] = len(summaries)
    summary["_jsonl_train_duplicate_steps"] = len(train_steps) - len(set(train_steps))
    summary["_jsonl_eval_duplicate_steps"] = len(eval_steps) - len(set(eval_steps))
    summary["_jsonl_timing_guard_failed"] = timing_guard_failed
    return evals, summary


TIMING_VALIDATED_MATRIXPOLICY_PHASES = {
    "E2_matrixpolicy_safe_speed_300m",
    "E2_rational_only_300m",
}


def timing_integrity_issues(
    row: dict[str, str],
    summary: dict[str, object] | None,
    max_matrixpolicy_sps: float,
    denylist_nodes: set[str],
) -> list[str]:
    if summary is None:
        return []
    issues = []
    if int(summary.get("_jsonl_config_count", 0)) != 1:
        issues.append(f"config_count={summary.get('_jsonl_config_count')}")
    if int(summary.get("_jsonl_summary_count", 0)) != 1:
        issues.append(f"summary_count={summary.get('_jsonl_summary_count')}")
    if int(summary.get("_jsonl_train_duplicate_steps", 0)) != 0:
        issues.append(f"duplicate_train_steps={summary.get('_jsonl_train_duplicate_steps')}")
    if int(summary.get("_jsonl_eval_duplicate_steps", 0)) != 0:
        issues.append(f"duplicate_eval_steps={summary.get('_jsonl_eval_duplicate_steps')}")
    if bool(summary.get("_jsonl_timing_guard_failed", False)):
        issues.append("timing_guard_failed_event_present")
    if row.get("optimizer") == "rational_matrix_policy_onpolicy" and row.get("phase") in TIMING_VALIDATED_MATRIXPOLICY_PHASES:
        node = str(summary.get("slurm_node", "") or "")
        if node in denylist_nodes:
            issues.append(f"denylisted_slurm_node={node}")
        if max_matrixpolicy_sps > 0.0:
            sps = float(summary.get("mean_seconds_per_step", 0.0))
            if sps > max_matrixpolicy_sps:
                issues.append(f"matrixpolicy_mean_seconds_per_step={sps:.4f}>{max_matrixpolicy_sps:.4f}")
    return issues


def _load_phase_rows(
    manifest: Path,
    run_root: Path,
    dataset: str,
    phase: str,
    wanted_methods: set[str] | None = None,
    max_matrixpolicy_sps: float = 0.0,
    denylist_nodes: set[str] | None = None,
    node_overrides: dict[str, dict[str, str]] | None = None,
    allow_timing_anomalies: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != phase or row["dataset"] != dataset:
                continue
            if wanted_methods is not None and row["method"] not in wanted_methods:
                continue
            jsonl_path = run_root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"
            evals, summary = read_jsonl(jsonl_path)
            summary = apply_timing_node_override(row, summary, {} if node_overrides is None else node_overrides)
            issues = timing_integrity_issues(row, summary, max_matrixpolicy_sps, set() if denylist_nodes is None else denylist_nodes)
            if issues and not allow_timing_anomalies:
                joined = "; ".join(issues)
                raise RuntimeError(f"Timing integrity check failed for {jsonl_path}: {joined}. Rerun/repair this row; do not exclude it from aggregates.")
            target_step = int(row["steps"])
            final_eval = next((item for item in reversed(evals) if int(item.get("step", -1)) == target_step), None)
            final_loss = safe_float(final_eval.get("val_loss")) if final_eval else math.nan
            final_ppl = safe_float(final_eval.get("val_ppl")) if final_eval else math.nan
            complete = (
                summary is not None
                and int(summary.get("completed_steps", -1)) == target_step
                and final_eval is not None
                and int(summary.get("steps", -1)) == target_step
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "row": int(row["row_index"]),
                    "row_id": row["row_id"],
                    "source_phase": row["phase"],
                    "source_row_index": int(row["row_index"]),
                    "source_row_id": row["row_id"],
                    "seed": int(row["seed"]),
                    "method": row["method"],
                    "activation": row["activation"],
                    "optimizer": row["optimizer"],
                    "complete": complete,
                    "final_val_loss": final_loss,
                    "final_val_ppl": final_ppl,
                    "steps": target_step,
                    "global_tokens_per_step": int(row["global_tokens_per_step"]),
                    "total_tokens": target_step * int(row["global_tokens_per_step"]),
                    "val_skip_tokens": int(row["val_skip_tokens"]),
                    "val_tokens": int(row["val_tokens"]),
                    "eval_interval": int(row["eval_interval"]),
                    "jsonl": str(jsonl_path),
                    "timing_integrity_issues": ";".join(issues),
                    "stopped_early": False if summary is None else bool(summary.get("stopped_early", False)),
                    "early_stop_reason": "" if summary is None else summary.get("early_stop_reason", ""),
                    "slurm_job_id": "" if summary is None else summary.get("slurm_job_id", ""),
                    "slurm_restart_count": "" if summary is None else summary.get("slurm_restart_count", ""),
                    "slurm_node": "" if summary is None else summary.get("slurm_node", ""),
                    "timing_attempt_id": "" if summary is None else summary.get("timing_attempt_id", ""),
                    "timing_node_override_reason": "" if summary is None else summary.get("timing_node_override_reason", ""),
                    "evals": evals,
                    "summary": summary,
                }
            )
    return rows


def load_rows(
    manifest: Path,
    run_root: Path,
    dataset: str,
    matrixpolicy_manifest: Path | None = None,
    matrixpolicy_phase: str | None = None,
    replacement_manifest: Path | None = None,
    replacement_phase: str | None = None,
    max_matrixpolicy_sps: float = 0.0,
    denylist_nodes: set[str] | None = None,
    node_overrides: dict[str, dict[str, str]] | None = None,
    allow_timing_anomalies: bool = False,
) -> list[dict[str, object]]:
    rows = _load_phase_rows(manifest, run_root, dataset, PHASE, max_matrixpolicy_sps=max_matrixpolicy_sps, denylist_nodes=denylist_nodes, node_overrides=node_overrides, allow_timing_anomalies=allow_timing_anomalies)
    by_key = {(int(row["seed"]), str(row["method"])): row for row in rows}

    def overlay(overrides: list[dict[str, object]]) -> None:
        for row in overrides:
            key = (int(row["seed"]), str(row["method"]))
            original = by_key.get(key)
            if original is not None:
                row["row"] = original["row"]
                row["row_id"] = original["row_id"]
            by_key[key] = row

    if matrixpolicy_manifest is not None and matrixpolicy_phase is not None:
        overlay(_load_phase_rows(matrixpolicy_manifest, run_root, dataset, matrixpolicy_phase, {MATRIXPOLICY_METHOD}, max_matrixpolicy_sps=max_matrixpolicy_sps, denylist_nodes=denylist_nodes, node_overrides=node_overrides, allow_timing_anomalies=allow_timing_anomalies))
    if replacement_manifest is not None and replacement_phase is not None:
        overlay(_load_phase_rows(replacement_manifest, run_root, dataset, replacement_phase, REPLACEMENT_RLB_METHODS, max_matrixpolicy_sps=max_matrixpolicy_sps, denylist_nodes=denylist_nodes, node_overrides=node_overrides, allow_timing_anomalies=allow_timing_anomalies))
    return sorted(by_key.values(), key=lambda row: int(row["row"]))

def final_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    out: list[dict[str, object]] = []
    for method, group in grouped.items():
        losses = [float(row["final_val_loss"]) for row in group if bool(row["complete"]) and math.isfinite(float(row["final_val_loss"]))]
        diverged = len(group) - len(losses)
        if losses:
            mean_loss = mean(losses)
            std_loss = sample_std(losses)
            min_loss = min(losses)
            max_loss = max(losses)
        else:
            mean_loss = std_loss = min_loss = max_loss = math.nan
        out.append(
            {
                "method": method,
                "mean_final_val_loss": mean_loss,
                "std_final_val_loss": std_loss,
                "min_final_val_loss": min_loss,
                "max_final_val_loss": max_loss,
                "finite_seeds": len(losses),
                "diverged_seeds": diverged,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            not math.isfinite(float(row["mean_final_val_loss"])),
            float(row["mean_final_val_loss"]) if math.isfinite(float(row["mean_final_val_loss"])) else math.inf,
            str(row["method"]),
        ),
    )


def runtime_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    out: list[dict[str, object]] = []
    for method, group in grouped.items():
        summaries = [row["summary"] for row in group if row.get("summary") is not None]
        totals = [float(summary["total_seconds"]) for summary in summaries]  # type: ignore[index]
        sps = [float(summary["mean_seconds_per_step"]) for summary in summaries]  # type: ignore[index]
        tps = [float(summary["tokens_per_second"]) for summary in summaries]  # type: ignore[index]
        if not totals:
            continue
        first = group[0]
        out.append(
            {
                "method": method,
                "activation": first["activation"],
                "optimizer": first["optimizer"],
                "runs": len(totals),
                "early_stop_runs": sum(1 for row in group if bool(row.get("stopped_early", False))),
                "total_seconds_mean": mean(totals),
                "total_seconds_std": sample_std(totals),
                "total_seconds_min": min(totals),
                "total_seconds_max": max(totals),
                "mean_seconds_per_step": mean(sps),
                "tokens_per_second_mean": mean(tps),
            }
        )
    return sorted(out, key=lambda row: (float(row["total_seconds_mean"]), str(row["method"])))


def first_hit_tokens(row: dict[str, object], target: float) -> tuple[int | None, int | None]:
    tokens_per_step = int(row["global_tokens_per_step"])
    for record in row["evals"]:  # type: ignore[union-attr]
        step = int(record.get("step", -1))
        val_loss = safe_float(record.get("val_loss"))
        if step > 0 and math.isfinite(val_loss) and val_loss <= target:
            return step, step * tokens_per_step
    return None, None


def token_savings(rows: list[dict[str, object]], targets: list[float]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_seed_method: dict[tuple[int, str], dict[str, object]] = {
        (int(row["seed"]), str(row["method"])): row for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows if str(row["method"]) != MATRIXPOLICY_METHOD})
    per_seed: list[dict[str, object]] = []
    aggregate: list[dict[str, object]] = []

    for target in targets:
        matrix_hits_by_seed: dict[int, tuple[int | None, int | None]] = {}
        second_hits: dict[int, tuple[str | None, int | None, int | None]] = {}
        adamw_hits: dict[int, tuple[int | None, int | None]] = {}

        for seed in seeds:
            mp_row = by_seed_method.get((seed, MATRIXPOLICY_METHOD))
            matrix_step, matrix_tokens = first_hit_tokens(mp_row, target) if mp_row else (None, None)
            matrix_hits_by_seed[seed] = (matrix_step, matrix_tokens)

            best_method = None
            best_step = None
            best_tokens = None
            for method in methods:
                row = by_seed_method.get((seed, method))
                if row is None:
                    continue
                step, tokens = first_hit_tokens(row, target)
                if tokens is None:
                    continue
                if best_tokens is None or tokens < best_tokens or (tokens == best_tokens and method < str(best_method)):
                    best_method = method
                    best_step = step
                    best_tokens = tokens
            second_hits[seed] = (best_method, best_step, best_tokens)

            adamw_row = by_seed_method.get((seed, ADAMW_METHOD))
            adamw_hits[seed] = first_hit_tokens(adamw_row, target) if adamw_row else (None, None)

            per_seed.append(
                {
                    "target_loss": target,
                    "seed": seed,
                    "matrixpolicy_step": matrix_step,
                    "matrixpolicy_tokens": matrix_tokens,
                    "second_best_method": best_method,
                    "second_best_step": best_step,
                    "second_best_tokens": best_tokens,
                    "silu_adamw_step": adamw_hits[seed][0],
                    "silu_adamw_tokens": adamw_hits[seed][1],
                }
            )

        mp_all = [tokens for _, tokens in matrix_hits_by_seed.values() if tokens is not None]
        second_common = [
            (matrix_hits_by_seed[seed][1], second_hits[seed][2])
            for seed in seeds
            if matrix_hits_by_seed[seed][1] is not None and second_hits[seed][2] is not None
        ]
        adamw_common = [
            (matrix_hits_by_seed[seed][1], adamw_hits[seed][1])
            for seed in seeds
            if matrix_hits_by_seed[seed][1] is not None and adamw_hits[seed][1] is not None
        ]

        mp_second_mean = mean([float(pair[0]) for pair in second_common]) if second_common else math.nan
        second_mean = mean([float(pair[1]) for pair in second_common]) if second_common else math.nan
        mp_adamw_mean = mean([float(pair[0]) for pair in adamw_common]) if adamw_common else math.nan
        adamw_mean = mean([float(pair[1]) for pair in adamw_common]) if adamw_common else math.nan
        saved_second = second_mean - mp_second_mean if second_common else math.nan
        saved_adamw = adamw_mean - mp_adamw_mean if adamw_common else math.nan

        aggregate.append(
            {
                "target_loss": target,
                "matrixpolicy_mean_tokens_all_hits": mean([float(tokens) for tokens in mp_all]) if len(mp_all) == len(seeds) else math.nan,
                "matrixpolicy_hit_seeds": len(mp_all),
                "matrixpolicy_mean_tokens_second_best_common": mp_second_mean,
                "second_best_mean_tokens_common": second_mean,
                "second_best_common_seeds": len(second_common),
                "saved_tokens_vs_second_best": saved_second,
                "saved_fraction_vs_second_best": saved_second / second_mean if second_common and second_mean else math.nan,
                "matrixpolicy_mean_tokens_silu_adamw_common": mp_adamw_mean,
                "silu_adamw_mean_tokens_common": adamw_mean,
                "silu_adamw_common_seeds": len(adamw_common),
                "saved_tokens_vs_silu_adamw": saved_adamw,
                "saved_fraction_vs_silu_adamw": saved_adamw / adamw_mean if adamw_common and adamw_mean else math.nan,
            }
        )

    return aggregate, per_seed


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float) and math.isnan(value):
                    value = "nan"
                cleaned[field] = value
            writer.writerow(cleaned)


def final_summary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Method | Final val loss mean +/- sample std | Min | Max | Notes |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        mean_loss = float(row["mean_final_val_loss"])
        std_loss = float(row["std_final_val_loss"])
        min_loss = float(row["min_final_val_loss"])
        max_loss = float(row["max_final_val_loss"])
        if math.isfinite(mean_loss):
            value = f"{fmt_float(mean_loss)} +/- {fmt_float(std_loss)}"
        else:
            value = "nan/diverged"
        notes = ""
        if int(row["diverged_seeds"]) > 0:
            notes = f"{row['diverged_seeds']} diverged/non-finite seeds"
        lines.append(
            f"| {row['method']} | {value} | {fmt_float(min_loss)} | {fmt_float(max_loss)} | {notes} |"
        )
    return "\n".join(lines)


def per_seed_gap_markdown(rows: list[dict[str, object]]) -> tuple[str, bool]:
    by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    lines = [
        "| Seed | MatrixPolicy final loss | Best non-MP method | Best non-MP final loss | Gap |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    all_best = True
    for seed in sorted(by_seed):
        seed_rows = by_seed[seed]
        mp_row = next(row for row in seed_rows if row["method"] == MATRIXPOLICY_METHOD)
        mp_loss = float(mp_row["final_val_loss"])
        non_mp = [
            row for row in seed_rows
            if row["method"] != MATRIXPOLICY_METHOD and math.isfinite(float(row["final_val_loss"]))
        ]
        best = min(non_mp, key=lambda row: float(row["final_val_loss"]))
        gap = float(best["final_val_loss"]) - mp_loss
        if not (math.isfinite(mp_loss) and gap > 0):
            all_best = False
        lines.append(
            f"| {seed} | {fmt_float(mp_loss)} | {best['method']} | {fmt_float(float(best['final_val_loss']))} | {fmt_float(gap)} |"
        )
    return "\n".join(lines), all_best


def runtime_summary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Method | Runs | Early stops | Mean runtime | Std | Range | Mean s/step | Mean tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {runs} | {early} | {mean:.1f} min | {std:.1f} min | {minv:.1f}-{maxv:.1f} min | {sps:.4f} | {tps:.1f} |".format(
                method=row["method"],
                runs=row["runs"],
                early=row.get("early_stop_runs", 0),
                mean=float(row["total_seconds_mean"]) / 60.0,
                std=float(row["total_seconds_std"]) / 60.0,
                minv=float(row["total_seconds_min"]) / 60.0,
                maxv=float(row["total_seconds_max"]) / 60.0,
                sps=float(row["mean_seconds_per_step"]),
                tps=float(row["tokens_per_second_mean"]),
            )
        )
    return "\n".join(lines)


def token_savings_markdown(rows: list[dict[str, object]], seed_count: int) -> str:
    lines = [
        "| Target loss | MP all-hit mean | Vs fastest non-MP: MP -> comparator (seeds) | Saved | Saved % | Vs SiLU+AdamW: MP -> AdamW (seeds) | Saved | Saved % |",
        "| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        target = float(row["target_loss"])
        mp_all = float(row["matrixpolicy_mean_tokens_all_hits"])
        second_seeds = int(row["second_best_common_seeds"])
        adamw_seeds = int(row["silu_adamw_common_seeds"])
        second_cmp = (
            f"{fmt_tokens(float(row['matrixpolicy_mean_tokens_second_best_common']))} -> "
            f"{fmt_tokens(float(row['second_best_mean_tokens_common']))} ({second_seeds}/{seed_count})"
            if second_seeds
            else f"not reached (0/{seed_count})"
        )
        adamw_cmp = (
            f"{fmt_tokens(float(row['matrixpolicy_mean_tokens_silu_adamw_common']))} -> "
            f"{fmt_tokens(float(row['silu_adamw_mean_tokens_common']))} ({adamw_seeds}/{seed_count})"
            if adamw_seeds
            else f"not reached (0/{seed_count})"
        )
        lines.append(
            "| {target:.2f} | {mp_all} | {second_cmp} | {second_saved} | {second_pct} | {adamw_cmp} | {adamw_saved} | {adamw_pct} |".format(
                target=target,
                mp_all=fmt_tokens(mp_all),
                second_cmp=second_cmp,
                second_saved=fmt_tokens(float(row["saved_tokens_vs_second_best"])),
                second_pct=fmt_percent(float(row["saved_fraction_vs_second_best"])),
                adamw_cmp=adamw_cmp,
                adamw_saved=fmt_tokens(float(row["saved_tokens_vs_silu_adamw"])),
                adamw_pct=fmt_percent(float(row["saved_fraction_vs_silu_adamw"])),
            )
        )
    return "\n".join(lines)


def curve_figures_markdown(dataset: str, label: str) -> str:
    fig_dir = "../iclr26_e2_figures"
    return f"""## Dense Curve Figures

All-method view:

![{label} E2 validation loss mean +/- std, all methods]({fig_dir}/{dataset}_core_validation_loss_mean_std.svg)

![{label} E2 validation PPL mean +/- std, all methods]({fig_dir}/{dataset}_core_validation_ppl_mean_std.svg)

![{label} E2 training loss mean +/- std, all methods]({fig_dir}/{dataset}_core_training_loss_mean_std.svg)

Clean comparison view:

![{label} E2 validation loss mean +/- std, clean comparison]({fig_dir}/{dataset}_clean_validation_loss_mean_std.svg)

![{label} E2 validation PPL mean +/- std, clean comparison]({fig_dir}/{dataset}_clean_validation_ppl_mean_std.svg)

![{label} E2 training loss mean +/- std, clean comparison]({fig_dir}/{dataset}_clean_training_loss_mean_std.svg)
"""


def write_readme(
    output_dir: Path,
    dataset: str,
    completed_date: str,
    rows: list[dict[str, object]],
    final_rows: list[dict[str, object]],
    runtime_rows: list[dict[str, object]],
    token_rows: list[dict[str, object]],
) -> None:
    label = DATASET_LABEL.get(dataset, dataset)
    row_start = min(int(row["row"]) for row in rows)
    row_end = max(int(row["row"]) for row in rows)
    steps = int(rows[0]["steps"])
    tokens_per_step = int(rows[0]["global_tokens_per_step"])
    total_tokens = int(rows[0]["total_tokens"])
    val_skip_tokens = int(rows[0]["val_skip_tokens"])
    val_tokens = int(rows[0]["val_tokens"])
    eval_interval = int(rows[0]["eval_interval"])
    seeds = sorted({int(row["seed"]) for row in rows})

    mp_summary = next(row for row in final_rows if row["method"] == MATRIXPOLICY_METHOD)
    next_best = [row for row in final_rows if row["method"] != MATRIXPOLICY_METHOD and math.isfinite(float(row["mean_final_val_loss"]))]
    next_best = sorted(next_best, key=lambda row: float(row["mean_final_val_loss"]))[:3]
    gap_table, all_best = per_seed_gap_markdown(rows)
    best_sentence = "MatrixPolicy is best on all three" if all_best else "MatrixPolicy is not best on every"
    next_best_text = ", ".join(
        f"`{row['method']}` at `{fmt_float(float(row['mean_final_val_loss']))} +/- {fmt_float(float(row['std_final_val_loss']))}`"
        for row in next_best
    )

    matrixpolicy_replaced = any(
        row["method"] == MATRIXPOLICY_METHOD and row.get("source_phase") != PHASE
        for row in rows
    )
    rlb_controls_replaced = any(
        str(row["method"]) in REPLACEMENT_RLB_METHODS and row.get("source_phase") != PHASE
        for row in rows
    )
    stopped_early_count = sum(1 for row in rows if bool(row.get("stopped_early", False)))
    replacement_bits = []
    if matrixpolicy_replaced:
        replacement_bits.append("MatrixPolicy entries use replacement JSONL rows for the same method and seed")
    if rlb_controls_replaced:
        replacement_bits.append("non-MatrixPolicy RLB optimizer controls use global-rational/no-local-atom (`rlb_fused_global_rational`) replacement rows")
    replacement_note = (
        "\n" + "; ".join(replacement_bits) + "; the `row` column remains the matched main-manifest E2 row, while `source_phase`/`source_row_id` record the actual timed run.\n"
        if replacement_bits
        else ""
    )
    completion_sentence = (
        f"All 45 paper-facing rows have final eval at step `{steps}`."
        if stopped_early_count == 0
        else f"The cell contains 45 paper-facing rows; `{stopped_early_count}` stopped early and are reported as diverged/non-finite rather than excluded."
    )

    text = f"""# ICLR26 E2 {label} 300M Summary

Completed: {completed_date}. Manifest rows `{row_start}-{row_end}` define the full {label} E2 M0/300M cell: 3 seeds x 15 fixed methods. {completion_sentence}

Each row uses `{tokens_per_step}` global tokens/step for about `{total_tokens / 1_000_000:.1f}M` train tokens. Validation uses the E2 {label} slice from the manifest: `val_skip_tokens={val_skip_tokens}`, `val_tokens={val_tokens}`, `eval_interval={eval_interval}`.{replacement_note}
## Final Validation Loss

{final_summary_markdown(final_rows)}

{best_sentence} {label} E2 seeds. Mean final val loss is `{fmt_float(float(mp_summary["mean_final_val_loss"]))} +/- {fmt_float(float(mp_summary["std_final_val_loss"]))}`; the next-best aggregate methods are {next_best_text}.

## Per-Seed MatrixPolicy Gap

{gap_table}

## Runtime Summary

`summary.total_seconds` is training-harness wall time for the manifest row. It excludes Slurm queue wait, dependency wait, token-cache construction, extension compilation, and launcher overhead. MatrixPolicy replacement rows must pass JSONL integrity checks and the denylisted-node guard; an optional per-step timing ceiling can be enabled manually, but is off by default. Failures abort generation and require rerun/repair rather than exclusion.

{runtime_summary_markdown(runtime_rows)}

{curve_figures_markdown(dataset, label)}
## Token-To-Target Savings

This table asks how many training tokens were needed to first reach a validation-loss threshold. It does not change the fixed-budget protocol; all completed runs still trained to about `{total_tokens / 1_000_000:.1f}M` tokens. The readout uses the native eval cadence of {eval_interval} steps, or `{tokens_per_step * eval_interval / 1_000_000:.2f}M` tokens.

`Second-best` means the fastest non-MatrixPolicy method to reach the target within the same seed. `AdamW` means the standard `silu_adamw` row. Savings and proportions are computed only on seeds where both MatrixPolicy and the comparator reached the target, so the comparison column explicitly shows `MP -> comparator` tokens and the shared seed count.

{token_savings_markdown(token_rows, len(seeds))}

## Files

- `final_summary.csv`: aggregate final validation losses by method.
- `per_seed_summary.csv`: per-row final results and JSONL provenance paths.
- `runtime_summary.csv`: aggregate runtime by method.
- `token_savings.csv`: aggregate token-to-target savings.
- `token_savings_per_seed.csv`: per-seed threshold hits and comparator identities.
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    args = parse_args()
    denylist_nodes = {node.strip() for node in args.matrixpolicy_denylist_nodes.split(",") if node.strip()}
    node_overrides = load_timing_node_overrides(args.timing_node_overrides)
    targets = args.targets if args.targets is not None else DEFAULT_TARGETS.get(args.dataset)
    if not targets:
        raise SystemExit(f"No default targets for dataset {args.dataset}; pass --targets.")

    rows = load_rows(
        args.manifest,
        args.run_root,
        args.dataset,
        args.matrixpolicy_manifest,
        args.matrixpolicy_phase,
        args.replacement_manifest,
        args.replacement_phase,
        args.matrixpolicy_max_seconds_per_step,
        denylist_nodes,
        node_overrides,
        args.allow_timing_anomalies,
    )
    if not rows:
        raise SystemExit(f"No {PHASE} rows found for dataset {args.dataset}.")
    incomplete = [row for row in rows if not bool(row["complete"]) and not bool(row.get("stopped_early", False))]
    if incomplete:
        missing = ", ".join(str(row["row"]) for row in incomplete)
        raise SystemExit(f"Dataset {args.dataset} is incomplete; missing complete rows without early-stop summaries: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = final_summary(rows)
    runtime_rows = runtime_summary(rows)
    token_rows, token_seed_rows = token_savings(rows, list(targets))

    write_csv(
        args.output_dir / "per_seed_summary.csv",
        rows,
        [
            "row",
            "row_id",
            "source_phase",
            "source_row_index",
            "source_row_id",
            "seed",
            "method",
            "activation",
            "optimizer",
            "complete",
            "final_val_loss",
            "final_val_ppl",
            "steps",
            "global_tokens_per_step",
            "total_tokens",
            "slurm_job_id",
            "slurm_restart_count",
            "slurm_node",
            "timing_attempt_id",
            "timing_node_override_reason",
            "timing_integrity_issues",
            "jsonl",
        ],
    )
    write_csv(
        args.output_dir / "final_summary.csv",
        final_rows,
        [
            "method",
            "mean_final_val_loss",
            "std_final_val_loss",
            "min_final_val_loss",
            "max_final_val_loss",
            "finite_seeds",
            "diverged_seeds",
        ],
    )
    write_csv(
        args.output_dir / "runtime_summary.csv",
        runtime_rows,
        [
            "method",
            "activation",
            "optimizer",
            "runs",
            "early_stop_runs",
            "total_seconds_mean",
            "total_seconds_std",
            "total_seconds_min",
            "total_seconds_max",
            "mean_seconds_per_step",
            "tokens_per_second_mean",
        ],
    )
    write_csv(
        args.output_dir / "token_savings.csv",
        token_rows,
        [
            "target_loss",
            "matrixpolicy_mean_tokens_all_hits",
            "matrixpolicy_hit_seeds",
            "matrixpolicy_mean_tokens_second_best_common",
            "second_best_mean_tokens_common",
            "second_best_common_seeds",
            "saved_tokens_vs_second_best",
            "saved_fraction_vs_second_best",
            "matrixpolicy_mean_tokens_silu_adamw_common",
            "silu_adamw_mean_tokens_common",
            "silu_adamw_common_seeds",
            "saved_tokens_vs_silu_adamw",
            "saved_fraction_vs_silu_adamw",
        ],
    )
    write_csv(
        args.output_dir / "token_savings_per_seed.csv",
        token_seed_rows,
        [
            "target_loss",
            "seed",
            "matrixpolicy_step",
            "matrixpolicy_tokens",
            "second_best_method",
            "second_best_step",
            "second_best_tokens",
            "silu_adamw_step",
            "silu_adamw_tokens",
        ],
    )
    write_readme(args.output_dir, args.dataset, args.completed_date, rows, final_rows, runtime_rows, token_rows)


if __name__ == "__main__":
    main()
