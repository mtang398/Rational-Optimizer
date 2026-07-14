#!/usr/bin/env python3
"""Submit the frozen MatrixPolicy live-stat correction campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "matrixpolicy_live_stats_20260712"
CAMPAIGN_ROOT = ROOT / "experiments/corrections" / CAMPAIGN_ID
FREEZE_PATH = CAMPAIGN_ROOT / "freeze.json"
BUILDER = ROOT / "experiments/scripts/build_matrixpolicy_live_stats_correction.py"
RUNTIME_ROOT = CAMPAIGN_ROOT / "runtime"
ROW_LAUNCHER = RUNTIME_ROOT / "experiments/scripts/run_matrixpolicy_live_stats_correction_job.sh"
GATE_LAUNCHER = RUNTIME_ROOT / "experiments/scripts/run_matrixpolicy_live_stats_nccl_gate.sh"
VALIDATOR = RUNTIME_ROOT / "experiments/scripts/validate_matrixpolicy_live_stats_correction.py"
PYTHON = ROOT / ".venv-cu128/bin/python"
LOG_ROOT = CAMPAIGN_ROOT / "logs"
RUN_ROOT = CAMPAIGN_ROOT / "runs"
SUBMISSION_ROOT = CAMPAIGN_ROOT / "submissions"
NODE_EXCLUDE = "sablab-gpu-12,seo-compute-01"

STAGES = {
    "e9_preflight": {
        "manifest": CAMPAIGN_ROOT / "manifests" / f"{CAMPAIGN_ID}_e9_preflight.csv",
        "rows": 10,
        "time": "00:15:00",
        "sync_interval": "1",
        "prefix": "mps-pf",
    },
    "main": {
        "manifest": CAMPAIGN_ROOT / "manifests" / f"{CAMPAIGN_ID}_main.csv",
        "rows": 30,
        "time": "24:00:00",
        "sync_interval": "0",
        "prefix": "mps-main",
    },
    "e8": {
        "manifest": CAMPAIGN_ROOT / "manifests" / f"{CAMPAIGN_ID}_e8.csv",
        "rows": 80,
        "time": "12:00:00",
        "sync_interval": "0",
        "prefix": "mps-e8",
    },
    "e9": {
        "manifest": CAMPAIGN_ROOT / "manifests" / f"{CAMPAIGN_ID}_e9.csv",
        "rows": 150,
        "time": "12:00:00",
        "sync_interval": "0",
        "prefix": "mps-e9",
    },
}


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=capture,
        text=True,
    )


def prepare_campaign() -> dict[str, object]:
    env = {**os.environ, "RATIONALOPT_WORKSPACE_ROOT": str(ROOT)}
    if not FREEZE_PATH.is_file():
        subprocess.run(
            [str(PYTHON), str(BUILDER), "build"],
            cwd=ROOT,
            env=env,
            check=True,
        )
    subprocess.run(
        [str(PYTHON), str(BUILDER), "verify"],
        cwd=ROOT,
        env=env,
        check=True,
    )
    freeze = json.loads(FREEZE_PATH.read_text())
    for kind, stage in STAGES.items():
        path = stage["manifest"]
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != stage["rows"]:
            raise SystemExit(
                f"{kind} manifest expected {stage['rows']} rows, found {len(rows)}"
            )
    return freeze


def existing_jobs() -> list[str]:
    result = run(
        ["squeue", "-h", "-u", os.environ.get("USER", "mt872"), "-o", "%i %j %T"]
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_job_id(stdout: str) -> str:
    job_id = stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise SystemExit(f"unexpected sbatch output: {stdout!r}")
    return job_id


class Submitter:
    def __init__(self, freeze: dict[str, object], dry_run: bool, ledger: Path):
        self.freeze = freeze
        self.dry_run = dry_run
        self.ledger = ledger
        self.sequence = 0
        self.records: list[dict[str, object]] = []
        self.handle = ledger.open("w", newline="")
        self.writer = csv.DictWriter(
            self.handle,
            fieldnames=(
                "sequence",
                "stage",
                "kind",
                "row_index",
                "row_id",
                "chain",
                "job_id",
                "dependency",
            ),
        )
        self.writer.writeheader()
        self._flush()

    def _flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def submit(
        self,
        command: list[str],
        *,
        stage: str,
        kind: str,
        dependency: str = "",
        row_index: str = "",
        row_id: str = "",
        chain: str = "",
    ) -> str:
        self.sequence += 1
        if self.dry_run:
            job_id = f"DRY{self.sequence:04d}"
        else:
            job_id = parse_job_id(run(command).stdout)
        record = {
            "sequence": self.sequence,
            "stage": stage,
            "kind": kind,
            "row_index": row_index,
            "row_id": row_id,
            "chain": chain,
            "job_id": job_id,
            "dependency": dependency,
        }
        self.records.append(record)
        self.writer.writerow(record)
        self._flush()
        return job_id

    def close(self) -> None:
        self.handle.close()

    def rollback(self) -> None:
        if self.dry_run:
            return
        job_ids = [
            str(record["job_id"])
            for record in self.records
            if str(record["job_id"]).isdigit()
        ]
        for start in range(0, len(job_ids), 100):
            subprocess.run(
                ["scancel", *job_ids[start : start + 100]],
                cwd=ROOT,
                check=False,
            )


def common_gpu_command(name: str, output: Path, walltime: str) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        "--partition=gpu",
        "--nodes=1",
        "--gres=gpu:nvidia_rtx_a6000:4",
        "--constraint=nvlink",
        f"--exclude={NODE_EXCLUDE}",
        "--cpus-per-task=32",
        "--mem=128G",
        f"--time={walltime}",
        "--requeue",
        "--signal=B:USR1@300",
        "--open-mode=append",
        f"--job-name={name}",
        f"--output={output}",
    ]


def add_dependency(command: list[str], dependency: str) -> None:
    if dependency:
        command.append(f"--dependency={dependency}")


def submit_gate(submitter: Submitter) -> str:
    output = LOG_ROOT / "nccl-gate-%j.out"
    command = common_gpu_command("mps-nccl-gate", output, "01:00:00")
    exports = ",".join(
        (
            "ALL",
            f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
            f"CAMPAIGN_FREEZE_SHA256={submitter.freeze['freeze_sha256']}",
            f"CAMPAIGN_RUNTIME_SHA256={submitter.freeze['runtime_sha256']}",
        )
    )
    command.extend([f"--export={exports}", str(GATE_LAUNCHER)])
    return submitter.submit(command, stage="nccl_gate", kind="gate")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def submit_row_stage(
    submitter: Submitter,
    kind: str,
    prerequisite_job: str,
) -> tuple[str, str]:
    stage = STAGES[kind]
    manifest: Path = stage["manifest"]
    frozen_manifest = submitter.freeze["manifests"][kind]
    output_root = RUN_ROOT / kind
    rows = load_rows(manifest)
    previous: list[str | None] = [None, None]
    for row in rows:
        index = int(row["row_index"])
        chain = index % 2
        dependency = (
            f"afterok:{prerequisite_job}"
            if previous[chain] is None
            else f"afterany:{previous[chain]}"
        )
        command = common_gpu_command(
            f"{stage['prefix']}-r{index:03d}",
            LOG_ROOT / f"{stage['prefix']}-r{index:03d}-%j.out",
            str(stage["time"]),
        )
        add_dependency(command, dependency)
        exports = ",".join(
            (
                "ALL",
                f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
                f"MANIFEST={manifest}",
                f"OUTPUT_ROOT={output_root}",
                f"ROW_START={index}",
                f"MATRIXPOLICY_DDP_SYNC_CHECK_INTERVAL={stage['sync_interval']}",
                f"CAMPAIGN_FREEZE_SHA256={submitter.freeze['freeze_sha256']}",
                f"CAMPAIGN_RUNTIME_SHA256={submitter.freeze['runtime_sha256']}",
                f"CAMPAIGN_MANIFEST_SHA256={frozen_manifest['sha256']}",
            )
        )
        command.extend([f"--export={exports}", str(ROW_LAUNCHER)])
        job_id = submitter.submit(
            command,
            stage=kind,
            kind="row",
            dependency=dependency,
            row_index=str(index),
            row_id=row["row_id"],
            chain=str(chain),
        )
        previous[chain] = job_id
    if previous[0] is None or previous[1] is None:
        raise AssertionError(f"stage {kind} did not populate both chains")
    return previous[0], previous[1]


def submit_validator(
    submitter: Submitter,
    kind: str,
    terminal_jobs: tuple[str, ...],
) -> str:
    dependency = "afterany:" + ":".join(terminal_jobs)
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
        f"--job-name=mps-validate-{kind}",
        f"--output={LOG_ROOT / f'validate-{kind}-%j.out'}",
    ]
    add_dependency(command, dependency)
    if kind == "final":
        validator_args = ["final"]
    else:
        stage = STAGES[kind]
        validator_args = [
            kind,
            "--manifest",
            str(stage["manifest"]),
            "--output-root",
            str(RUN_ROOT / kind),
        ]
    wrap = " ".join(
        [
            f"RATIONALOPT_WORKSPACE_ROOT={ROOT}",
            str(PYTHON),
            str(VALIDATOR),
            *validator_args,
        ]
    )
    command.append(f"--wrap={wrap}")
    return submitter.submit(
        command,
        stage=f"validate_{kind}",
        kind="validator",
        dependency=dependency,
    )


def submit_watchdog(
    submitter: Submitter,
    validated_kind: str,
    validator_job: str,
    downstream_job_ids: list[str],
) -> str:
    dependency = f"afterany:{validator_job}"
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
        f"--job-name=mps-watch-{validated_kind}",
        f"--output={LOG_ROOT / f'watch-{validated_kind}-%j.out'}",
        f"--dependency={dependency}",
    ]
    report = CAMPAIGN_ROOT / "validation" / f"{validated_kind}.json"
    wrap_parts = [
        str(PYTHON),
        str(
            RUNTIME_ROOT
            / "experiments/scripts/cancel_failed_matrixpolicy_live_stats_dependencies.py"
        ),
        "--report",
        str(report),
    ]
    for job_id in downstream_job_ids:
        wrap_parts.extend(("--job-id", job_id))
    command.append(f"--wrap={shlex.join(wrap_parts)}")
    return submitter.submit(
        command,
        stage=f"watch_{validated_kind}",
        kind="watchdog",
        dependency=dependency,
    )


def submit_campaign(freeze: dict[str, object], dry_run: bool) -> dict[str, object]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ledger = SUBMISSION_ROOT / f"submission_{stamp}.csv"
    submitter = Submitter(freeze, dry_run, ledger)
    try:
        gate = submit_gate(submitter)
        preflight = submit_row_stage(submitter, "e9_preflight", gate)
        preflight_validator = submit_validator(submitter, "e9_preflight", preflight)
        main = submit_row_stage(submitter, "main", preflight_validator)
        main_validator = submit_validator(submitter, "main", main)
        e8 = submit_row_stage(submitter, "e8", main_validator)
        e8_validator = submit_validator(submitter, "e8", e8)
        e9 = submit_row_stage(submitter, "e9", e8_validator)
        e9_validator = submit_validator(submitter, "e9", e9)
        final_validator = submit_validator(submitter, "final", (e9_validator,))
        base_records = list(submitter.records)
        validator_jobs = (
            ("e9", e9_validator),
            ("e8", e8_validator),
            ("main", main_validator),
            ("e9_preflight", preflight_validator),
            ("nccl_gate", gate),
        )
        for validated_kind, validator_job in validator_jobs:
            validator_sequence = next(
                int(record["sequence"])
                for record in base_records
                if record["job_id"] == validator_job
            )
            downstream = [
                str(record["job_id"])
                for record in submitter.records
                if int(record["sequence"]) > validator_sequence
                and str(record["job_id"]).isdigit()
            ]
            submit_watchdog(
                submitter,
                validated_kind,
                validator_job,
                downstream,
            )
    except BaseException as error:
        submitter.rollback()
        failure_path = SUBMISSION_ROOT / f"submission_{stamp}.failed.json"
        failure_path.write_text(
            json.dumps(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "error": repr(error),
                    "rolled_back_jobs": len(submitter.records),
                    "ledger": str(ledger.relative_to(ROOT)),
                    "failed_unix_time": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    finally:
        submitter.close()
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "dry_run": dry_run,
        "freeze_sha256": freeze["freeze_sha256"],
        "submitted_jobs": len(submitter.records),
        "row_jobs": sum(record["kind"] == "row" for record in submitter.records),
        "gate_job": gate,
        "final_validator_job": final_validator,
        "ledger": str(ledger.relative_to(ROOT)),
        "submitted_unix_time": time.time(),
    }
    summary_path = SUBMISSION_ROOT / f"submission_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-existing-jobs", action="store_true")
    args = parser.parse_args()
    if not args.confirm_submit and not args.dry_run:
        parser.error("use --confirm-submit to queue the campaign")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = CAMPAIGN_ROOT / ".submit.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another correction-campaign submission is active") from error
        freeze = prepare_campaign()
        jobs = existing_jobs()
        if jobs and not args.allow_existing_jobs:
            raise SystemExit(
                "existing Slurm jobs detected before campaign submission:\n" + "\n".join(jobs)
            )
        submit_campaign(freeze, args.dry_run)


if __name__ == "__main__":
    main()
