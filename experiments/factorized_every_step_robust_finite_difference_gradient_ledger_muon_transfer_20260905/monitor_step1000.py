#!/usr/bin/env python3
"""Interrupt a transfer row if the candidate trails its paired step-1,000 control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import suite
from .row_tools import records


def evals(path: Path) -> dict[int, float]:
    return {
        int(row["step"]): float(row["val_loss"])
        for row in records(path) if row.get("event") == "eval"
    }


def write(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.resolve()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", required=True, type=int)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    row = suite.row_at(args.matrix_index)
    control = evals(args.control)
    if 1000 not in control:
        raise RuntimeError("paired control lacks step 1,000")
    candidate = evals(args.candidate)
    if 1000 not in candidate:
        raise SystemExit(3)
    lead = control[1000] - candidate[1000]
    payload = {
        "schema": "factorized_ledger_full_transfer_step1000_screen_v1",
        "matrix_index": row["matrix_index"],
        "model": row["model"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "control": row["control_name"],
        "control_step1000_loss": control[1000],
        "candidate_step1000_loss": candidate[1000],
        "candidate_step1000_lead": lead,
        "status": "pass_nonnegative" if lead >= 0.0 else "failed_negative_interrupted",
    }
    write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if lead >= 0.0 else 4)


if __name__ == "__main__":
    main()
