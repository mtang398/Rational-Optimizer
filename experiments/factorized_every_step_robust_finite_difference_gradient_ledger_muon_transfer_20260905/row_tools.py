#!/usr/bin/env python3
"""Small shell-facing helpers for collision-free matrix execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import suite


def records(path: Path):
    if not path.is_file():
        return []
    result = []
    with path.open() as handle:
        for line in handle:
            result.append(json.loads(line))
    return result


def terminal(path: Path, steps: int) -> bool:
    rows = records(path)
    configs = [row for row in rows if row.get("event") == "config"]
    summaries = [row for row in rows if row.get("event") == "summary"]
    endpoints = [
        row for row in rows
        if row.get("event") == "eval" and int(row.get("step", -1)) == steps
    ]
    return bool(
        len(configs) == 1
        and len(summaries) == 1
        and len(endpoints) == 1
        and int(summaries[0].get("completed_steps", -1)) == steps
        and not summaries[0].get("stopped_early")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    describe = sub.add_parser("describe")
    describe.add_argument("--matrix-index", required=True, type=int)
    describe.add_argument("--output-root", required=True, type=Path)
    check = sub.add_parser("terminal")
    check.add_argument("--path", required=True, type=Path)
    check.add_argument("--steps", required=True, type=int)
    args = parser.parse_args()
    if args.command == "terminal":
        raise SystemExit(0 if terminal(args.path, args.steps) else 3)
    row = suite.row_at(args.matrix_index)
    fields = (
        row["model"], row["dataset"], str(row["seed"]), str(row["steps"]),
        str(suite.jsonl_path(row, "control", args.output_root)),
        str(suite.jsonl_path(row, "candidate", args.output_root)),
        suite.run_name(row, "control"), suite.run_name(row, "candidate"),
    )
    print("\n".join(fields))


if __name__ == "__main__":
    main()

