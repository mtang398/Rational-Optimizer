#!/usr/bin/env python3
"""Cancel downstream correction jobs when a stage validator does not pass."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--job-id", action="append", default=[])
    args = parser.parse_args()

    status = "missing"
    if args.report.is_file():
        try:
            status = str(json.loads(args.report.read_text()).get("status", "missing"))
        except (json.JSONDecodeError, OSError):
            status = "invalid"
    if status == "pass":
        print(f"validator passed; no downstream cancellation: {args.report}")
        return

    job_ids = sorted({job_id for job_id in args.job_id if job_id.isdigit()})
    if not job_ids:
        raise SystemExit(f"validator status={status}; no downstream job IDs were supplied")
    for start in range(0, len(job_ids), 100):
        subprocess.run(["scancel", *job_ids[start : start + 100]], check=True)
    print(
        f"validator status={status}; cancelled {len(job_ids)} downstream jobs using {args.report}"
    )


if __name__ == "__main__":
    main()
