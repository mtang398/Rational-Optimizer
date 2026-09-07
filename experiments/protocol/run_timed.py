#!/usr/bin/env python3
"""Run one command and record its complete process wall time."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    if not args.arm.strip():
        raise SystemExit("--arm must be non-empty")

    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(command)

    def forward(signum, _frame):
        if process.poll() is None:
            process.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return_code = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    ended_ns = time.monotonic_ns()
    payload = {
        "schema": "rationalopt_process_wall_clock_v1",
        "arm": args.arm,
        "command": command,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "elapsed_seconds": (ended_ns - started_ns) / 1_000_000_000.0,
        "return_code": int(return_code),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    }
    write_atomic(args.output, payload)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
