#!/usr/bin/env python3
"""Submit a fresh persistent-worker DAG for the correction campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
PYTHON = ROOT / ".venv-cu128/bin/python"
VALIDATOR = ROOT / "experiments/scripts/validate_matrixpolicy_live_stats_correction.py"
VALIDATOR_BUILDER = ROOT / "experiments/scripts/build_matrixpolicy_live_stats_correction.py"
WATCHDOG = (
    CAMPAIGN_ROOT
    / "runtime/experiments/scripts/cancel_failed_matrixpolicy_live_stats_dependencies.py"
)
NODE_EXCLUDE = "sablab-gpu-12,seo-compute-01"

STAGES = {
    "e9_preflight": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e9_preflight.csv",
        "rows": 10,
        "workers": 1,
        "walltime": "00:30:00",
        "sync_interval": "1",
        "prefix": "mps-pfw2",
    },
    "main": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_main.csv",
        "rows": 30,
        "workers": 4,
        "walltime": "08:00:00",
        "sync_interval": "0",
        "prefix": "mps-mainw2",
    },
    "e8": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e8.csv",
        "rows": 80,
        "workers": 4,
        "walltime": "12:00:00",
        "sync_interval": "0",
        "prefix": "mps-e8w2",
    },
    "e9": {
        "manifest": CAMPAIGN_ROOT / "manifests/matrixpolicy_live_stats_20260712_e9.csv",
        "rows": 150,
        "workers": 4,
        "walltime": "20:00:00",
        "sync_interval": "0",
        "prefix": "mps-e9w2",
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


def stage_rows(stage: dict[str, object], worker_index: int) -> list[int]:
    return list(range(worker_index, int(stage["rows"]), int(stage["workers"])))


def submit_worker(
    freeze: dict[str, object],
    stage_name: str,
    worker_index: int,
    prerequisite: str,
) -> tuple[str, list[int]]:
    stage = STAGES[stage_name]
    manifest = Path(stage["manifest"])
    indices = stage_rows(stage, worker_index)
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
    if prerequisite:
        command.append(f"--dependency=afterok:{prerequisite}")
    frozen_manifest = freeze["manifests"][stage_name]
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
    return parse_job_id(run(command).stdout), indices


def submit_validator(
    stage_name: str,
    workers: list[str],
    stamp: str,
) -> str:
    dependency = "afterany:" + ":".join(workers)
    command = [
        "sbatch",
        "--parsable",
        "--partition=default_partition",
        "--nodes=1",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:30:00",
        "--requeue",
        "--open-mode=append",
        f"--job-name=mps-v2-{stage_name}",
        f"--output={LOG_ROOT / f'validate-v2-{stage_name}-{stamp}-%j.out'}",
        f"--dependency={dependency}",
    ]
    stage = STAGES[stage_name]
    validator_command = [
        "env",
        f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
        str(PYTHON),
        str(VALIDATOR),
        stage_name,
        "--manifest",
        str(stage["manifest"]),
        "--output-root",
        str(RUN_ROOT / stage_name),
    ]
    command.append(f"--wrap={shlex.join(validator_command)}")
    return parse_job_id(run(command).stdout)


def submit_final_validator(e9_validator: str, stamp: str) -> str:
    validator_command = [
        "env",
        f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
        str(PYTHON),
        str(VALIDATOR),
        "final",
    ]
    command = [
        "sbatch",
        "--parsable",
        "--partition=default_partition",
        "--nodes=1",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:30:00",
        "--requeue",
        "--open-mode=append",
        "--job-name=mps-v2-final",
        f"--output={LOG_ROOT / f'validate-v2-final-{stamp}-%j.out'}",
        f"--dependency=afterok:{e9_validator}",
        f"--wrap={shlex.join(validator_command)}",
    ]
    return parse_job_id(run(command).stdout)


def submit_watchdog(
    stage_name: str,
    validator: str,
    downstream: list[str],
    stamp: str,
) -> str:
    wrapped = [
        str(PYTHON),
        str(WATCHDOG),
        "--report",
        str(CAMPAIGN_ROOT / "validation" / f"{stage_name}.json"),
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
        f"--job-name=mps-v2-watch-{stage_name}",
        f"--output={LOG_ROOT / f'watch-v2-{stage_name}-{stamp}-%j.out'}",
        f"--dependency=afterany:{validator}",
        f"--wrap={shlex.join(wrapped)}",
    ]
    return parse_job_id(run(command).stdout)


def verify_coverage(
    records: list[dict[str, object]],
    stage_names: list[str],
) -> None:
    for stage_name in stage_names:
        stage = STAGES[stage_name]
        observed: list[int] = []
        for record in records:
            if record["stage"] == stage_name:
                observed.extend(int(value) for value in str(record["row_indices"]).split(":"))
        expected = list(range(int(stage["rows"])))
        if sorted(observed) != expected or len(observed) != len(set(observed)):
            raise RuntimeError(f"worker coverage mismatch for {stage_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument(
        "--resume-after-preflight",
        action="store_true",
        help="resume at main after verifying the existing 10/10 preflight report",
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required")
    if not WORKER.is_file() or not os.access(WORKER, os.X_OK):
        raise SystemExit(f"worker launcher is not executable: {WORKER}")
    existing = run(["squeue", "-h", "-u", os.environ.get("USER", "mt872"), "-o", "%i|%j"]).stdout
    conflicting = [
        line
        for line in existing.splitlines()
        if "|mps-" in line
    ]
    if conflicting:
        raise SystemExit("existing MatrixPolicy campaign jobs remain:\n" + "\n".join(conflicting))

    freeze = json.loads(FREEZE_PATH.read_text())
    if args.resume_after_preflight:
        preflight_report_path = CAMPAIGN_ROOT / "validation/e9_preflight.json"
        if not preflight_report_path.is_file():
            raise SystemExit("preflight validation report is missing")
        preflight_report = json.loads(preflight_report_path.read_text())
        expected_preflight = {
            "status": "pass",
            "expected_rows": 10,
            "passed_rows": 10,
            "freeze_sha256": freeze["freeze_sha256"],
        }
        mismatches = {
            key: (preflight_report.get(key), expected)
            for key, expected in expected_preflight.items()
            if preflight_report.get(key) != expected
        }
        if mismatches:
            raise SystemExit(f"preflight validation barrier did not pass: {mismatches}")
        stage_names = ["main", "e8", "e9"]
    else:
        stage_names = list(STAGES)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    submitted: list[str] = []
    records: list[dict[str, object]] = []
    workers_by_stage: dict[str, list[str]] = {}
    validators: dict[str, str] = {}
    try:
        prerequisite = ""
        for stage_name in stage_names:
            stage = STAGES[stage_name]
            workers: list[str] = []
            for worker_index in range(int(stage["workers"])):
                job_id, indices = submit_worker(
                    freeze,
                    stage_name,
                    worker_index,
                    prerequisite,
                )
                submitted.append(job_id)
                workers.append(job_id)
                records.append(
                    {
                        "stage": stage_name,
                        "worker_index": worker_index,
                        "job_id": job_id,
                        "prerequisite": prerequisite,
                        "walltime": stage["walltime"],
                        "row_indices": ":".join(map(str, indices)),
                    }
                )
            workers_by_stage[stage_name] = workers
            validator = submit_validator(stage_name, workers, stamp)
            submitted.append(validator)
            validators[stage_name] = validator
            prerequisite = validator

        final_validator = submit_final_validator(validators["e9"], stamp)
        submitted.append(final_validator)
        downstream = {
            "main": workers_by_stage["e8"],
            "e8": workers_by_stage["e9"],
            "e9": [final_validator],
        }
        if "e9_preflight" in stage_names:
            downstream["e9_preflight"] = workers_by_stage["main"]
        watchdogs: dict[str, str] = {}
        for stage_name in stage_names:
            job_id = submit_watchdog(
                stage_name,
                validators[stage_name],
                downstream[stage_name],
                stamp,
            )
            submitted.append(job_id)
            watchdogs[stage_name] = job_id

        verify_coverage(records, stage_names)
        queued = run(
            ["squeue", "-h", "-j", ",".join(submitted), "-o", "%i|%T"]
        ).stdout.splitlines()
        observed = {line.split("|", 1)[0] for line in queued}
        if observed != set(submitted):
            raise RuntimeError(
                f"fresh DAG is incomplete in Slurm: missing {sorted(set(submitted) - observed)}"
            )

        csv_path = SUBMISSION_ROOT / f"persistent_worker_dag_{stamp}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "stage",
                    "worker_index",
                    "job_id",
                    "prerequisite",
                    "walltime",
                    "row_indices",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        summary = {
            "campaign": "matrixpolicy_live_stats_20260712",
            "freeze_sha256": freeze["freeze_sha256"],
            "worker_ledger": str(csv_path.relative_to(ROOT)),
            "worker_launcher_sha256": hashlib.sha256(WORKER.read_bytes()).hexdigest(),
            "validator_sha256": hashlib.sha256(VALIDATOR.read_bytes()).hexdigest(),
            "validator_builder_sha256": hashlib.sha256(
                VALIDATOR_BUILDER.read_bytes()
            ).hexdigest(),
            "workers_by_stage": workers_by_stage,
            "validators": validators,
            "final_validator": final_validator,
            "watchdogs": watchdogs,
            "max_concurrent_gpu_workers": 4,
            "gpus_per_worker": 4,
            "submitted_jobs": len(submitted),
            "submitted_unix_time": time.time(),
            "resume_after_preflight": args.resume_after_preflight,
            "supersedes_failed_validators": [
                "848507",
                "848538",
                "848619",
                "848770",
                "853057",
                "853062",
                "853067",
                "853072",
            ],
        }
        json_path = csv_path.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, sort_keys=True))
    except BaseException:
        if submitted:
            run(["scancel", *submitted], check=False)
        raise


if __name__ == "__main__":
    main()
