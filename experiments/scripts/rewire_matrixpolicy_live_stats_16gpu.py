#!/usr/bin/env python3
"""Rewire the queued correction campaign from two to four GPU-job chains."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = ROOT / "experiments/corrections/matrixpolicy_live_stats_20260712"
SUBMISSION_ROOT = CAMPAIGN_ROOT / "submissions"
STAGES = ("e9_preflight", "main", "e8", "e9")
CHAIN_COUNT = 4


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def normalize_dependency(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.strip())
    if value in {"", "(null)", "N/A"}:
        return ""
    grouped: dict[str, list[str]] = defaultdict(list)
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        dependency_type, separator, job_text = part.partition(":")
        if not separator:
            raise ValueError(f"cannot parse dependency {part!r}")
        grouped[dependency_type].extend(job_id for job_id in job_text.split(":") if job_id)
    return ",".join(
        f"{dependency_type}:{':'.join(job_ids)}"
        for dependency_type, job_ids in grouped.items()
    )


def load_submission(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 281 or sum(row["kind"] == "row" for row in rows) != 270:
        raise SystemExit("submission ledger does not describe the complete frozen campaign")
    return rows


def queued_jobs(job_ids: list[str]) -> dict[str, dict[str, str]]:
    result = run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(job_ids),
            "-o",
            "%i|%T|%E",
        ]
    )
    observed: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        job_id, state, dependency = line.split("|", 2)
        observed[job_id] = {
            "state": state,
            "dependency": normalize_dependency(dependency),
        }
    return observed


def planned_updates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_stage = {
        stage: sorted(
            (row for row in rows if row["kind"] == "row" and row["stage"] == stage),
            key=lambda row: int(row["row_index"]),
        )
        for stage in STAGES
    }
    validators = {
        stage: next(
            row["job_id"]
            for row in rows
            if row["kind"] == "validator" and row["stage"] == f"validate_{stage}"
        )
        for stage in STAGES
    }
    prerequisites = {
        "e9_preflight": next(row["job_id"] for row in rows if row["kind"] == "gate"),
        "main": validators["e9_preflight"],
        "e8": validators["main"],
        "e9": validators["e8"],
    }

    updates: list[dict[str, str]] = []
    for stage in STAGES:
        stage_rows = by_stage[stage]
        indices = [int(row["row_index"]) for row in stage_rows]
        if indices != list(range(len(stage_rows))):
            raise SystemExit(f"{stage} row indices are not contiguous")
        by_index = {int(row["row_index"]): row for row in stage_rows}
        for index, row in by_index.items():
            if index < CHAIN_COUNT:
                dependency = f"afterok:{prerequisites[stage]}"
            else:
                dependency = f"afterany:{by_index[index - CHAIN_COUNT]['job_id']}"
            updates.append(
                {
                    "job_id": row["job_id"],
                    "stage": stage,
                    "kind": "row",
                    "row_index": str(index),
                    "new_dependency": dependency,
                }
            )

        terminals = []
        for chain in range(CHAIN_COUNT):
            chain_rows = [
                row
                for index, row in by_index.items()
                if index % CHAIN_COUNT == chain
            ]
            if not chain_rows:
                raise SystemExit(f"{stage} does not populate chain {chain}")
            terminals.append(
                max(chain_rows, key=lambda row: int(row["row_index"]))["job_id"]
            )
        updates.append(
            {
                "job_id": validators[stage],
                "stage": stage,
                "kind": "validator",
                "row_index": "",
                "new_dependency": "afterany:" + ":".join(terminals),
            }
        )
    return updates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required to change queued Slurm dependencies")
    ledger = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    rows = load_submission(ledger)
    updates = planned_updates(rows)
    job_ids = [update["job_id"] for update in updates]
    before = queued_jobs(job_ids)
    missing = sorted(set(job_ids) - before.keys())
    if missing:
        raise SystemExit(f"jobs are no longer queued: {missing[:12]}")
    nonpending = {
        job_id: record["state"]
        for job_id, record in before.items()
        if record["state"] != "PENDING"
    }
    if nonpending:
        raise SystemExit(f"refusing to rewire non-pending jobs: {nonpending}")
    for update in updates:
        update["old_dependency"] = before[update["job_id"]]["dependency"]

    changed: list[dict[str, str]] = []
    try:
        for update in updates:
            if update["old_dependency"] == update["new_dependency"]:
                continue
            run(
                [
                    "scontrol",
                    "update",
                    f"JobId={update['job_id']}",
                    f"Dependency={update['new_dependency']}",
                ]
            )
            changed.append(update)
    except BaseException:
        for update in reversed(changed):
            subprocess.run(
                [
                    "scontrol",
                    "update",
                    f"JobId={update['job_id']}",
                    f"Dependency={update['old_dependency']}",
                ],
                cwd=ROOT,
                check=False,
            )
        raise

    after = queued_jobs(job_ids)
    mismatches = {
        update["job_id"]: {
            "expected": update["new_dependency"],
            "observed": after.get(update["job_id"], {}).get("dependency"),
        }
        for update in updates
        if after.get(update["job_id"], {}).get("dependency")
        != update["new_dependency"]
    }
    if mismatches:
        raise SystemExit(f"post-update dependency verification failed: {mismatches}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = SUBMISSION_ROOT / f"concurrency_rewire_16gpu_{stamp}.csv"
    fields = (
        "job_id",
        "stage",
        "kind",
        "row_index",
        "old_dependency",
        "new_dependency",
    )
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updates)
        handle.flush()
        os.fsync(handle.fileno())
    summary = {
        "campaign": "matrixpolicy_live_stats_20260712",
        "chain_count": CHAIN_COUNT,
        "max_concurrent_gpu_jobs": CHAIN_COUNT,
        "gpus_per_job": 4,
        "max_concurrent_gpus": 16,
        "updated_jobs": len(changed),
        "verified_jobs": len(updates),
        "source_ledger": str(ledger.relative_to(ROOT)),
        "rewire_ledger": str(output_csv.relative_to(ROOT)),
        "rewired_unix_time": time.time(),
    }
    output_json = output_csv.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
