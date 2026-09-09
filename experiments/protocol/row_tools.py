#!/usr/bin/env python3
"""Small shell-facing helpers for collision-free matrix execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE / "activation_optimizer_manifest.csv"
PRIMARY_SUITE = "18l_1024d_100m_tokens_3050_steps"
CANDIDATE_OPTIMIZERS = ("tiller_v1", "tiller_then_muon_v1")
BASELINE_STAGES = {
    "muon": {"muon"},
    "adamw": {"adamw"},
    "remaining": {
        "lion", "soap_adamw", "ademamix", "adafactor_came",
        "schedule_free_adamw",
    },
}


def manifest_stage_indices(
    manifest: Path, phase: str, stage: str
) -> list[int]:
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["phase"] == phase]
    if not selected:
        raise RuntimeError(f"unknown suite: {phase}")
    if stage != "all":
        optimizers = BASELINE_STAGES.get(stage)
        if optimizers is None:
            raise RuntimeError(f"unknown baseline stage: {stage}")
        selected = [row for row in selected if row["optimizer"] in optimizers]
    return [int(row["row_index"]) for row in selected]


def matrix_suite_indices(
    phase: str, optimizer: str = "tiller_v1"
) -> list[int]:
    if optimizer not in CANDIDATE_OPTIMIZERS:
        raise RuntimeError(f"unknown candidate optimizer: {optimizer}")
    payload = json.loads((PACKAGE / "matrix.json").read_text())
    if (payload.get("schema") != "tiller_evaluation_matrix_v1"
            or payload.get("matrix_rows") != len(payload.get("rows", ()))):
        raise RuntimeError("candidate matrix inventory changed")
    selected = [
        int(row["matrix_index"])
        for row in payload["rows"]
        if row["phase"] == phase and row["candidate_optimizer"] == optimizer
    ]
    if not selected:
        raise RuntimeError(f"unknown TILLER suite/optimizer: {phase}/{optimizer}")
    return selected


def ordinal_value(values: list[int], ordinal: int, label: str) -> int:
    if not 0 <= ordinal < len(values):
        raise RuntimeError(
            f"{label} ordinal {ordinal} is outside 0..{len(values) - 1}"
        )
    return values[ordinal]


def records(path: Path):
    if not path.is_file():
        return []
    result = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A rank-zero writer may be interrupted between the write and its
            # newline. Only an incomplete final record is safe to ignore.
            if index == len(lines) - 1:
                break
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}")
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object JSONL record {index + 1} in {path}")
        result.append(record)
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
    describe.add_argument("--control-root", required=True, type=Path)
    check = sub.add_parser("terminal")
    check.add_argument("--path", required=True, type=Path)
    check.add_argument("--steps", required=True, type=int)
    manifest_index = sub.add_parser("manifest-index")
    manifest_index.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    manifest_index.add_argument("--suite", default=PRIMARY_SUITE)
    manifest_index.add_argument(
        "--stage", choices=(*BASELINE_STAGES, "all"), default="muon"
    )
    manifest_index.add_argument("--ordinal", required=True, type=int)
    matrix_index = sub.add_parser("matrix-index")
    matrix_index.add_argument("--suite", default=PRIMARY_SUITE)
    matrix_index.add_argument(
        "--optimizer", choices=CANDIDATE_OPTIMIZERS, default="tiller_v1"
    )
    matrix_index.add_argument("--ordinal", required=True, type=int)
    args = parser.parse_args()
    if args.command == "terminal":
        raise SystemExit(0 if terminal(args.path, args.steps) else 3)
    if args.command == "manifest-index":
        values = manifest_stage_indices(args.manifest, args.suite, args.stage)
        print(ordinal_value(values, args.ordinal, "manifest stage"))
        return
    if args.command == "matrix-index":
        values = matrix_suite_indices(args.suite, args.optimizer)
        print(ordinal_value(values, args.ordinal, "TILLER suite"))
        return
    from . import suite

    row = suite.row_at(args.matrix_index)
    fields = (
        row["model"], row["dataset"], str(row["seed"]), str(row["steps"]),
        str(suite.jsonl_path(row, "control", args.control_root)),
        str(suite.jsonl_path(row, "candidate", args.output_root)),
        suite.run_name(row, "control"), suite.run_name(row, "candidate"),
    )
    print("\n".join(fields))


if __name__ == "__main__":
    main()
