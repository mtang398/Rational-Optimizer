#!/usr/bin/env python3
"""Build an incremental or final result ledger for the 24-row matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MATRIX = PACKAGE / "matrix.json"
EXACT_OPTIMIZER_KEY = "factorized_every_step_rfd_gradient_ledger_muon_v1"


def matrix_payload():
    payload = json.loads(MATRIX.read_text())
    expected_schema = (
        "factorized_every_step_robust_finite_difference_gradient_ledger_muon_"
        "transfer_matrix_v1"
    )
    if payload.get("schema") != expected_schema:
        raise RuntimeError("full-method transfer matrix schema changed")
    if payload.get("matrix_rows") != 24 or len(payload.get("rows", ())) != 24:
        raise RuntimeError("full-method transfer matrix inventory changed")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PACKAGE / "results")
    parser.add_argument("--output", type=Path, default=PACKAGE / "RESULTS.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    matrix = matrix_payload()
    complete = []
    failed = []
    execution_failures = []
    pending = []
    for row in matrix["rows"]:
        root = args.results_root / f"row-{int(row['matrix_index']):02d}"
        report = root / "RESULT.json"
        screen = root / "STEP1000_SCREEN.json"
        if report.is_file():
            item = json.loads(report.read_text())
            if item.get("status") != "complete":
                raise RuntimeError(f"completed row report has bad status: {report}")
            complete.append(item)
        elif screen.is_file():
            item = json.loads(screen.read_text())
            if item.get("status") == "failed_negative_interrupted":
                failed.append(item)
            else:
                pending.append(row["matrix_index"])
        else:
            failure_files = sorted(root.glob("FAILURE-*.json"))
            failure = json.loads(failure_files[-1].read_text()) if failure_files else None
            if failure:
                execution_failures.append(failure)
            pending.append(row["matrix_index"])
    terminal_count = len(complete) + len(failed)
    if args.require_complete and terminal_count != 24:
        raise RuntimeError(
            f"matrix is not terminal: complete={len(complete)}, failed={len(failed)}, "
            f"pending={pending}"
        )
    grouped = defaultdict(list)
    for item in complete:
        grouped[(item["model"], item["dataset"])].append(item)
    summaries = []
    for (model, dataset), rows in sorted(grouped.items()):
        leads = [float(row["absolute_endpoint_lead"]) for row in rows]
        candidate = [float(row["candidate_endpoint_loss"]) for row in rows]
        control = [float(row["control_endpoint_loss"]) for row in rows]
        summaries.append({
            "model": model,
            "dataset": dataset,
            "completed_seeds": sorted(int(row["seed"]) for row in rows),
            "rfd_endpoint_loss_mean": statistics.fmean(candidate),
            "rfd_endpoint_loss_std": statistics.stdev(candidate) if len(candidate) > 1 else None,
            "control_endpoint_loss_mean": statistics.fmean(control),
            "control_endpoint_loss_std": statistics.stdev(control) if len(control) > 1 else None,
            "absolute_endpoint_lead_mean": statistics.fmean(leads),
            "absolute_endpoint_lead_std": statistics.stdev(leads) if len(leads) > 1 else None,
            "step1000_lead_mean": statistics.fmean(
                float(row["step1000_absolute_lead"]) for row in rows
            ),
            "exact_same_allocation_endpoint_total_time_ratio": (
                sum(float(row["candidate_endpoint_total_seconds"]) for row in rows)
                / sum(float(row["control_endpoint_total_seconds"]) for row in rows)
            ),
            "all_rows_pass_final_1_05_time_gate": all(
                row["passes_final_1_05_time_gate"] for row in rows
            ),
        })
    repo_bytes = sum(
        path.stat().st_size for path in PACKAGE.rglob("*") if path.is_file()
    )
    result = {
        "schema": "factorized_ledger_full_transfer_results_v1",
        "status": (
            "complete" if terminal_count == 24 else "running"
        ),
        "matrix_rows": 24,
        "complete_rows": len(complete),
        "failed_negative_step1000_rows": len(failed),
        "pending_rows": pending,
        "method": EXACT_OPTIMIZER_KEY,
        "matrix_sha256": sha256(MATRIX),
        "dataset_summaries": summaries,
        "row_results": sorted(complete, key=lambda row: row["matrix_index"]),
        "failed_rows": sorted(failed, key=lambda row: row["matrix_index"]),
        "retryable_execution_failures": sorted(
            execution_failures, key=lambda row: row["matrix_index"]
        ),
        "all_24_dataset_seed_rows_terminal": terminal_count == 24,
        "all_completed_rows_positive_at_step1000": all(
            float(row["step1000_absolute_lead"]) >= 0.0 for row in complete
        ),
        "all_completed_rows_pass_final_1_05_time_gate": all(
            row["passes_final_1_05_time_gate"] for row in complete
        ),
        "campaign_artifact_bytes": repo_bytes,
        "storage_policy": "no token-cache duplication; rank-0 JSONL plus compact reports only",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.resolve()}\n"
    )
    print(rendered, end="")


if __name__ == "__main__":
    main()
