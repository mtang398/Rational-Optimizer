#!/usr/bin/env python3
"""Persist a machine-readable row failure without mislabeling preemption as science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", required=True, type=int)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    row = suite.row_at(args.matrix_index)
    screen = None
    if args.screen is not None and args.screen.is_file():
        try:
            screen = json.loads(args.screen.read_text())
        except Exception as error:
            screen = {"parse_error": repr(error)}
    negative = bool(
        isinstance(screen, dict)
        and screen.get("status") == "failed_negative_interrupted"
    )
    status = (
        "failed_negative_step1000"
        if negative
        else "retryable_execution_failure"
    )
    payload = {
        "schema": "factorized_ledger_full_transfer_failure_v1",
        "matrix_index": row["matrix_index"],
        "model": row["model"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "stage": args.stage,
        "exit_code": args.exit_code,
        "status": status,
        "scientific_terminal": negative,
        "step1000_screen": screen,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
