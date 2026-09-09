#!/usr/bin/env python3
"""Interrupt a transfer row if the candidate trails its paired step-1,000 control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import suite
from .analyze_row import (
    control_source_identity_mismatch,
    expected_experiment_identity,
)
from .row_tools import records


def evals(path: Path) -> dict[int, float]:
    return {
        int(row["step"]): float(row["val_loss"])
        for row in records(path) if row.get("event") == "eval"
    }


def audited_config(path: Path, row: dict, arm: str) -> dict:
    configs = [record for record in records(path) if record.get("event") == "config"]
    if len(configs) != 1:
        if arm == "candidate" and len(configs) == 0:
            raise SystemExit(3)
        raise RuntimeError(f"{arm} must contain exactly one config record")
    config = configs[0]
    prefix = "control" if arm == "control" else "candidate"
    expected = {
        "activation": row[f"{prefix}_activation"],
        "optimizer": row[f"{prefix}_optimizer"],
        "dataset": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_revision": row["dataset_revision"],
        "tokenizer": row["tokenizer"],
        "tokenizer_revision": row["tokenizer_revision"],
        "experiment_identity": expected_experiment_identity(row, arm),
        "steps": row["steps"],
        "seed": row["seed"],
        "train_tokens": row["max_train_tokens"],
        "val_tokens": row["max_val_tokens"],
    }
    mismatch = {
        key: {"observed": config.get(key), "required": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if arm == "control":
        mismatch.update(
            {
                f"control_source_identity.{key}": value
                for key, value in control_source_identity_mismatch(config, row).items()
            }
        )
    else:
        identity = config.get("tiller_experiment_identity", {})
        for key, value in {
            "matrix_index": row["matrix_index"],
            "source_manifest_row_index": row["source_manifest_row_index"],
            "source_manifest_row_id": row["source_manifest_row_id"],
        }.items():
            if identity.get(key) != value:
                mismatch[f"matrix_identity.{key}"] = {
                    "observed": identity.get(key), "required": value
                }
        if identity.get("passed") is not True:
            mismatch["matrix_identity.passed"] = identity.get("passed")
    if mismatch:
        raise RuntimeError(f"{arm} configuration mismatch: {mismatch}")
    return config


def write(path: Path, payload: dict) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", required=True, type=int)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    row = suite.row_at(args.matrix_index)
    control_config = audited_config(args.control, row, "control")
    control = evals(args.control)
    if 1000 not in control:
        raise RuntimeError("paired control lacks step 1,000")
    candidate_config = audited_config(args.candidate, row, "candidate")
    for key in (
        "train_token_sample_sha256", "val_token_sample_sha256",
        "first_batch_index_sha256", "validation_index_sha256",
    ):
        if control_config.get(key) != candidate_config.get(key):
            raise RuntimeError(f"paired data/order identity differs at {key}")
    candidate = evals(args.candidate)
    if 1000 not in candidate:
        raise SystemExit(3)
    lead = control[1000] - candidate[1000]
    payload = {
        "schema": "tiller_step1000_screen_v1",
        "matrix_index": row["matrix_index"],
        "phase": row["phase"],
        "model": row["model"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "control": row["control_name"],
        "candidate": row["candidate_name"],
        "candidate_optimizer": row["candidate_optimizer"],
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
