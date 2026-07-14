#!/usr/bin/env python3
"""Replace queued one-row correction jobs with persistent stage workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = ROOT / "experiments/corrections/matrixpolicy_live_stats_20260712"
FREEZE_PATH = CAMPAIGN_ROOT / "freeze.json"
LOG_ROOT = CAMPAIGN_ROOT / "logs"
RUN_ROOT = CAMPAIGN_ROOT / "runs"
SUBMISSION_ROOT = CAMPAIGN_ROOT / "submissions"
WORKER = ROOT / "experiments/scripts/run_matrixpolicy_live_stats_worker.sh"
LEDGER = SUBMISSION_ROOT / "submission_20260712_163526.csv"
NODE_EXCLUDE = "sablab-gpu-12,seo-compute-01"

STAGES = {
    "e9_preflight": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e9_preflight.csv",
        "rows": 10,
        "workers": 1,
        "walltime": "00:30:00",
        "prerequisite": "",
        "validator": "848507",
        "sync_interval": "1",
        "prefix": "mps-pfw",
    },
    "main": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_main.csv",
        "rows": 30,
        "workers": 4,
        "walltime": "24:00:00",
        "prerequisite": "848507",
        "validator": "848538",
        "sync_interval": "0",
        "prefix": "mps-mainw",
    },
    "e8": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e8.csv",
        "rows": 80,
        "workers": 4,
        "walltime": "24:00:00",
        "prerequisite": "848538",
        "validator": "848619",
        "sync_interval": "0",
        "prefix": "mps-e8w",
    },
    "e9": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e9.csv",
        "rows": 150,
        "workers": 4,
        "walltime": "24:00:00",
        "prerequisite": "848619",
        "validator": "848770",
        "sync_interval": "0",
        "prefix": "mps-e9w",
    },
}


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def parse_job_id(stdout: str) -> str:
    job_id = stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {stdout!r}")
    return job_id


def normalize_dependency(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.strip())
    if value in {"", "(null)", "N/A"}:
        return ""
    grouped: dict[str, list[str]] = {}
    for part in value.split(","):
        dependency_type, separator, job_text = part.strip().partition(":")
        if not separator:
            raise ValueError(f"cannot parse dependency {part!r}")
        grouped.setdefault(dependency_type, []).extend(
            job_id for job_id in job_text.split(":") if job_id
        )
    return ",".join(
        f"{dependency_type}:{':'.join(job_ids)}"
        for dependency_type, job_ids in grouped.items()
    )


def queued(job_ids: list[str]) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    result = run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T|%E|%l"]
    )
    records: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        job_id, state, dependency, walltime = line.split("|", 3)
        records[job_id] = {
            "state": state,
            "dependency": normalize_dependency(dependency),
            "walltime": walltime,
        }
    return records


def current_dependency(job_id: str) -> str:
    output = run(["scontrol", "show", "job", "-o", job_id]).stdout
    match = re.search(r"(?:^|\s)Dependency=(\S+)", output)
    if match is None:
        raise RuntimeError(f"cannot read dependency for job {job_id}")
    return normalize_dependency(match.group(1))


def chunked(values: list[str], size: int = 100) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def load_old_rows() -> list[dict[str, str]]:
    with LEDGER.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["kind"] == "row" and row["stage"] in STAGES]
    if len(selected) != 270:
        raise RuntimeError(f"expected 270 old row jobs, found {len(selected)}")
    return selected


def stage_rows(stage: dict[str, object], worker_index: int) -> list[int]:
    workers = int(stage["workers"])
    return list(range(worker_index, int(stage["rows"]), workers))


def worker_command(
    freeze: dict[str, object],
    stage_name: str,
    worker_index: int,
) -> tuple[list[str], list[int]]:
    stage = STAGES[stage_name]
    indices = stage_rows(stage, worker_index)
    manifest = Path(stage["manifest"])
    frozen_manifest = freeze["manifests"][stage_name]
    prerequisite = str(stage["prerequisite"])
    dependency = f"afterok:{prerequisite}" if prerequisite else ""
    command = [
        "sbatch",
        "--parsable",
        "--partition=gpu",
        "--nodes=1",
        "--gres=gpu:nvidia_rtx_a6000:4",
        "--constraint=nvlink",
        f"--exclude={NODE_EXCLUDE}",
        "--cpus-per-task=32",
        "--mem=128G",
        f"--time={stage['walltime']}",
        "--requeue",
        "--signal=B:USR1@300",
        "--open-mode=append",
        f"--job-name={stage['prefix']}-{worker_index}",
        f"--output={LOG_ROOT / f'{stage['prefix']}-{worker_index}-%j.out'}",
    ]
    if dependency:
        command.append(f"--dependency={dependency}")
    exports = ",".join(
        (
            "ALL",
            f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
            f"MANIFEST={manifest}",
            f"OUTPUT_ROOT={RUN_ROOT / stage_name}",
            f"ROW_INDICES={':'.join(map(str, indices))}",
            f"MATRIXPOLICY_DDP_SYNC_CHECK_INTERVAL={stage['sync_interval']}",
            f"CAMPAIGN_FREEZE_SHA256={freeze['freeze_sha256']}",
            f"CAMPAIGN_RUNTIME_SHA256={freeze['runtime_sha256']}",
            f"CAMPAIGN_MANIFEST_SHA256={frozen_manifest['sha256']}",
        )
    )
    command.extend((f"--export={exports}", str(WORKER)))
    return command, indices


def submit_watchdog(
    stage_name: str,
    validator: str,
    downstream: list[str],
) -> str:
    report = CAMPAIGN_ROOT / "validation" / f"{stage_name}.json"
    helper = (
        CAMPAIGN_ROOT
        / "runtime/experiments/scripts/cancel_failed_matrixpolicy_live_stats_dependencies.py"
    )
    wrapped = [
        str(ROOT / ".venv-cu128/bin/python"),
        str(helper),
        "--report",
        str(report),
    ]
    for job_id in downstream:
        wrapped.extend(("--job-id", job_id))
    command = [
        "sbatch",
        "--parsable",
        "--partition=default_partition",
        "--nodes=1",
        "--cpus-per-task=1",
        "--mem=2G",
        "--time=00:10:00",
        "--requeue",
        "--open-mode=append",
        f"--job-name=mps-wwatch-{stage_name}",
        f"--output={LOG_ROOT / f'worker-watch-{stage_name}-%j.out'}",
        f"--dependency=afterany:{validator}",
        f"--wrap={shlex.join(wrapped)}",
    ]
    return parse_job_id(run(command).stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required")
    if not WORKER.is_file() or not os.access(WORKER, os.X_OK):
        raise SystemExit(f"worker launcher is not executable: {WORKER}")

    freeze = json.loads(FREEZE_PATH.read_text())
    old_rows = load_old_rows()
    old_ids = [row["job_id"] for row in old_rows]
    old_queue = queued(old_ids)
    running_old = {
        job_id: record["state"]
        for job_id, record in old_queue.items()
        if record["state"] != "PENDING"
    }
    if running_old:
        raise SystemExit(f"refusing conversion with running old rows: {running_old}")
    pending_old_ids = sorted(old_queue, key=int)

    validator_ids = [str(STAGES[name]["validator"]) for name in STAGES]
    old_validator_dependencies = {
        job_id: current_dependency(job_id) for job_id in validator_ids
    }
    new_jobs: list[str] = []
    worker_records: list[dict[str, object]] = []
    rewired_validators: list[str] = []
    old_jobs_cancelled = False
    try:
        for ids in chunked(pending_old_ids):
            run(["scontrol", "hold", *ids])

        workers_by_stage: dict[str, list[str]] = {}
        for stage_name, stage in STAGES.items():
            stage_workers: list[str] = []
            for worker_index in range(int(stage["workers"])):
                command, indices = worker_command(freeze, stage_name, worker_index)
                job_id = parse_job_id(run(command).stdout)
                new_jobs.append(job_id)
                stage_workers.append(job_id)
                worker_records.append(
                    {
                        "stage": stage_name,
                        "worker_index": worker_index,
                        "job_id": job_id,
                        "dependency": (
                            f"afterok:{stage['prerequisite']}"
                            if stage["prerequisite"]
                            else ""
                        ),
                        "walltime": stage["walltime"],
                        "row_indices": ":".join(map(str, indices)),
                    }
                )
            workers_by_stage[stage_name] = stage_workers

        for stage_name, stage in STAGES.items():
            validator = str(stage["validator"])
            dependency = "afterany:" + ":".join(workers_by_stage[stage_name])
            run(["scontrol", "update", f"JobId={validator}", f"Dependency={dependency}"])
            rewired_validators.append(validator)

        expected_dependencies = {
            record["job_id"]: str(record["dependency"])
            for record in worker_records
        }
        observed_workers = queued(new_jobs)
        if set(observed_workers) != set(new_jobs):
            raise RuntimeError("not every worker job is present in Slurm")
        for job_id, expected in expected_dependencies.items():
            observed = observed_workers[job_id]
            if observed["state"] != "PENDING":
                raise RuntimeError(f"worker {job_id} is unexpectedly {observed['state']}")
            if observed["dependency"] != expected:
                raise RuntimeError(
                    f"worker {job_id} dependency mismatch: "
                    f"{observed['dependency']} != {expected}"
                )
        for stage_name, stage in STAGES.items():
            expected = "afterany:" + ":".join(workers_by_stage[stage_name])
            observed = current_dependency(str(stage["validator"]))
            if observed != expected:
                raise RuntimeError(
                    f"validator {stage['validator']} dependency mismatch: {observed} != {expected}"
                )

        downstream_by_stage = {
            "e9_preflight": workers_by_stage["main"],
            "main": workers_by_stage["e8"],
            "e8": workers_by_stage["e9"],
            "e9": ["848771"],
        }
        watchdogs = {
            stage_name: submit_watchdog(
                stage_name,
                str(STAGES[stage_name]["validator"]),
                downstream_by_stage[stage_name],
            )
            for stage_name in STAGES
        }
        new_jobs.extend(watchdogs.values())

        for ids in chunked(pending_old_ids):
            run(["scancel", *ids])
        old_jobs_cancelled = True
        if queued(pending_old_ids):
            raise RuntimeError("superseded one-row jobs remain queued after cancellation")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = SUBMISSION_ROOT / f"persistent_workers_{stamp}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "stage",
                    "worker_index",
                    "job_id",
                    "dependency",
                    "walltime",
                    "row_indices",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(worker_records)
            handle.flush()
            os.fsync(handle.fileno())
        summary = {
            "campaign": "matrixpolicy_live_stats_20260712",
            "freeze_sha256": freeze["freeze_sha256"],
            "worker_ledger": str(csv_path.relative_to(ROOT)),
            "workers_by_stage": workers_by_stage,
            "watchdogs": watchdogs,
            "validator_dependencies": {
                name: "afterany:" + ":".join(workers_by_stage[name])
                for name in STAGES
            },
            "retired_pending_row_jobs": len(pending_old_ids),
            "max_concurrent_gpu_workers": 4,
            "gpus_per_worker": 4,
            "submitted_unix_time": time.time(),
        }
        json_path = csv_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True))
    except BaseException:
        if not old_jobs_cancelled:
            for validator in reversed(rewired_validators):
                run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={validator}",
                        f"Dependency={old_validator_dependencies[validator]}",
                    ],
                    check=False,
                )
            for ids in chunked(new_jobs):
                run(["scancel", *ids], check=False)
            for ids in chunked(pending_old_ids):
                run(["scontrol", "release", *ids], check=False)
        raise


if __name__ == "__main__":
    main()
