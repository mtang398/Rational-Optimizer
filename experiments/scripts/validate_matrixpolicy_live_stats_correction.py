#!/usr/bin/env python3
"""Validate frozen inputs and completed correction-campaign JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    os.environ.get("RATIONALOPT_WORKSPACE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
CAMPAIGN_ID = "matrixpolicy_live_stats_20260712"
CAMPAIGN_ROOT = ROOT / "experiments/corrections" / CAMPAIGN_ID
FREEZE_PATH = CAMPAIGN_ROOT / "freeze.json"
VALIDATION_ROOT = CAMPAIGN_ROOT / "validation"
DENIED_NODES = {"sablab-gpu-12", "seo-compute-01"}
GLOBAL_SCOPE_ARMS = {"A3", "A5", "A6", "A7", "A8", "A9"}
MATRIX_TELEMETRY_KEYS = {
    "matrix_policy_adam_lr_scale_by_layer_role",
    "matrix_policy_applied_muon_mix_by_layer_role",
    "matrix_policy_apply_muon_update",
    "matrix_policy_group_policy_enabled",
    "matrix_policy_group_scale_mean",
    "matrix_policy_muon_mix_by_layer_role",
    "matrix_policy_pair_rescale_enabled",
    "matrix_policy_update_rms_by_role",
    "matrix_policy_weight_rms_by_role",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"non-object JSON record at {path}:{line_number}")
            records.append(record)
    return records


def finite_metric_records(records: list[dict[str, object]]) -> bool:
    keys = {
        "loss",
        "val_loss",
        "val_ppl",
        "mean_seconds_per_step",
        "total_seconds",
        "tokens_per_second",
    }
    for record in records:
        for key in keys:
            value = record.get(key)
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(value):
                return False
    return True


def value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, bool):
        return False
    if isinstance(actual, int):
        try:
            return actual == int(str(expected))
        except ValueError:
            return False
    if isinstance(actual, float):
        try:
            return math.isclose(actual, float(str(expected)), rel_tol=1e-12, abs_tol=1e-12)
        except ValueError:
            return False
    return str(actual) == str(expected)


def expanded_config_expectations(row: dict[str, str]) -> dict[str, object]:
    expected: dict[str, object] = {
        "activation": row["activation"],
        "optimizer": row["optimizer"],
        "seed": row["seed"],
        "steps": row["steps"],
        "eval_interval": row["eval_interval"],
        "eval_batches": row["eval_batches"],
        "batch_size_per_gpu": row["batch_size"],
        "grad_accum": row["grad_accum"],
        "seq_len": row["seq_len"],
        "layers": row["layers"],
        "d_model": row["d_model"],
        "heads": row["heads"],
        "ffn_dim": row["ffn_dim"],
        "optimizer_lr": row["lr"],
        "optimizer_min_lr": row["min_lr"],
        "optimizer_weight_decay": row["weight_decay"],
        "dataset": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_text_column": row["text_column"],
        "train_split": row["train_split"],
        "validation_split": row["val_split"],
        "validation_skip_tokens": row["val_skip_tokens"],
        "train_tokens": row["train_tokens"],
        "val_tokens": row["val_tokens"],
        "global_tokens_per_step": row["global_tokens_per_step"],
        "world_size": 4,
        "timing_guard_max_seconds_per_step": 0.0,
    }
    tokens = shlex.split(row.get("extra_args", ""))
    index = 0
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--"):
            raise ValueError(f"unexpected extra-argument token {flag!r}")
        name = flag[2:]
        if name.startswith("no-"):
            expected[name[3:].replace("-", "_")] = False
            index += 1
            continue
        key = name.replace("-", "_")
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            expected[key] = tokens[index + 1]
            index += 2
        else:
            expected[key] = True
            index += 1
    return expected


def validate_event_schema(
    row: dict[str, str],
    records: list[dict[str, object]],
    kind: str,
) -> list[str]:
    errors: list[str] = []
    steps = int(row["steps"])
    eval_interval = int(row["eval_interval"])
    expected_eval_steps = {1, *range(eval_interval, steps + 1, eval_interval)}
    observed_eval_steps = [
        int(record.get("step", -1))
        for record in records
        if record.get("event") == "eval"
    ]
    if observed_eval_steps != sorted(expected_eval_steps):
        errors.append(
            f"evaluation cadence mismatch: expected {sorted(expected_eval_steps)}, found {observed_eval_steps}"
        )
    expected_train_steps = [1, *range(10, steps + 1, 10)]
    observed_train = [record for record in records if record.get("event") == "train"]
    observed_train_steps = [int(record.get("step", -1)) for record in observed_train]
    if observed_train_steps != expected_train_steps:
        errors.append("training-log cadence mismatch")
    active_times = []
    for record in records:
        if record.get("event") not in {"train", "eval"}:
            continue
        active = record.get("active_seconds_after_event")
        if not isinstance(active, (int, float)) or not math.isfinite(active):
            errors.append(f"{record.get('event')} step {record.get('step')} lacks cumulative timing")
            continue
        active_times.append(float(active))
        if record.get("event") == "eval":
            at_loss = record.get("active_seconds_at_val_loss")
            if not isinstance(at_loss, (int, float)) or not math.isfinite(at_loss):
                errors.append(f"eval step {record.get('step')} lacks val-loss timing")
    if active_times != sorted(active_times):
        errors.append("cumulative active time is not monotone")
    if row["optimizer"] == "rational_matrix_policy_onpolicy":
        for record in observed_train:
            missing = sorted(MATRIX_TELEMETRY_KEYS - record.keys())
            if missing:
                errors.append(
                    f"MatrixPolicy train step {record.get('step')} lacks telemetry {missing}"
                )
                break
        if kind == "e9_preflight" and observed_train:
            arm = row.get("arm_id", "")
            expected_group = arm not in {"A2", "A4"}
            expected_pair = arm not in {"A2", "A8"}
            expected_muon_apply = arm not in {"A6", "A7"}
            last = observed_train[-1]
            for key, expected in (
                ("matrix_policy_group_policy_enabled", expected_group),
                ("matrix_policy_pair_rescale_enabled", expected_pair),
                ("matrix_policy_apply_muon_update", expected_muon_apply),
            ):
                if last.get(key) is not expected:
                    errors.append(
                        f"{arm} intervention telemetry {key} expected {expected}, found {last.get(key)!r}"
                    )
    return errors


def expected_scope(row: dict[str, str]) -> str:
    if row["optimizer"] != "rational_matrix_policy_onpolicy":
        return "disabled" if row["activation"] == "silu" else "telemetry_only"
    arm = row.get("arm_id", "")
    if arm and arm not in GLOBAL_SCOPE_ARMS:
        return "telemetry_only"
    return "optimizer_gains_global_weighted_train_only"


def validate_gpu_metadata(config: dict[str, object]) -> list[str]:
    errors: list[str] = []
    node = str(config.get("slurm_node") or "")
    if not node:
        errors.append("missing Slurm node")
    if node in DENIED_NODES:
        errors.append(f"denylisted Slurm node {node}")
    metadata = config.get("gpu_metadata")
    if not isinstance(metadata, list) or len(metadata) != 4:
        errors.append("GPU metadata does not contain four devices")
    else:
        for entry in metadata:
            if not isinstance(entry, dict) or "A6000" not in str(entry.get("name", "")):
                errors.append(f"non-A6000 GPU metadata: {entry!r}")
    return errors


def validate_row(
    row: dict[str, str],
    output_root: Path,
    freeze: dict[str, object],
    manifest_sha: str,
    kind: str,
) -> dict[str, object]:
    path = (
        output_root
        / row["phase"]
        / row["dataset"]
        / row["row_id"]
        / f"{row['activation']}.jsonl"
    )
    errors: list[str] = []
    if not path.is_file():
        return {
            "row_index": int(row["row_index"]),
            "row_id": row["row_id"],
            "status": "missing",
            "errors": [f"missing JSONL: {path}"],
        }
    try:
        records = read_jsonl(path)
    except ValueError as error:
        return {
            "row_index": int(row["row_index"]),
            "row_id": row["row_id"],
            "status": "invalid",
            "errors": [str(error)],
        }
    configs = [record for record in records if record.get("event") == "config"]
    summaries = [record for record in records if record.get("event") == "summary"]
    evals = [record for record in records if record.get("event") == "eval"]
    if len(configs) != 1:
        errors.append(f"expected one config record, found {len(configs)}")
    if len(summaries) != 1:
        errors.append(f"expected one summary record, found {len(summaries)}")
    config = configs[0] if len(configs) == 1 else {}
    summary = summaries[0] if len(summaries) == 1 else {}
    if config:
        errors.extend(validate_gpu_metadata(config))
        expected_values = {
            "e9_row_id": row["row_id"],
            "e9_manifest_sha256": manifest_sha,
            "e9_freeze_sha256": freeze["freeze_sha256"],
            "e9_runtime_freeze_sha256": freeze["runtime_sha256"],
            "rlb_live_stats_scope": expected_scope(row),
        }
        for field, expected in expected_values.items():
            if config.get(field) != expected:
                errors.append(
                    f"config {field} mismatch: expected {expected!r}, found {config.get(field)!r}"
                )
        if row.get("arm_id"):
            for field, expected in (
                ("e9_arm_id", row["arm_id"]),
                ("e9_design_version", row["design_version"]),
            ):
                if config.get(field) != expected:
                    errors.append(
                        f"config {field} mismatch: expected {expected!r}, found {config.get(field)!r}"
                    )
        try:
            expanded = expanded_config_expectations(row)
        except ValueError as error:
            errors.append(str(error))
            expanded = {}
        for field, expected in expanded.items():
            if field not in config:
                errors.append(f"expanded config is missing {field}")
            elif not value_matches(config[field], expected):
                errors.append(
                    f"expanded config {field} mismatch: expected {expected!r}, found {config[field]!r}"
                )
        if not isinstance(config.get("params"), int) or int(config["params"]) <= 0:
            errors.append("expanded config has no positive parameter count")
        expected_interval = 1 if kind == "e9_preflight" else 0
        if int(config.get("matrixpolicy_ddp_sync_check_interval", -1)) != expected_interval:
            errors.append("unexpected MatrixPolicy DDP consistency-check interval")
    terminal_failure = bool(summary.get("stopped_early")) and bool(summary.get("stop_reason"))
    complete = (
        int(summary.get("steps", -1)) == int(row["steps"])
        and int(summary.get("completed_steps", -1)) == int(row["steps"])
        and bool(evals)
        and int(evals[-1].get("step", -1)) == int(row["steps"])
    )
    if not complete and not (kind == "e9" and terminal_failure):
        errors.append("row did not reach its required terminal condition")
    if terminal_failure and kind != "e9":
        errors.append(f"unexpected numerical failure: {summary.get('stop_reason')}")
    if not terminal_failure:
        errors.extend(validate_event_schema(row, records, kind))
    if not finite_metric_records(records) and not terminal_failure:
        errors.append("non-finite metric in a nominally completed row")
    if config and summary and config.get("timing_attempt_id") != summary.get("timing_attempt_id"):
        errors.append("config and summary belong to different timing attempts")
    if any(record.get("event") == "attempt_interrupted" for record in records):
        errors.append("final JSONL contains an interrupted attempt")
    if summary:
        total_seconds = summary.get("total_seconds")
        if not isinstance(total_seconds, (int, float)) or not math.isfinite(total_seconds):
            errors.append("summary lacks finite final-attempt timing")
        if complete:
            expected_tokens = int(row["steps"]) * int(row["global_tokens_per_step"])
            if int(summary.get("completed_tokens", -1)) != expected_tokens:
                errors.append("summary completed-token count is inconsistent")
    if kind == "e9_preflight" and row["optimizer"] == "rational_matrix_policy_onpolicy":
        checks = [
            record.get("ddp_parameter_sync_max_abs")
            for record in records
            if record.get("event") == "train"
            and record.get("ddp_parameter_sync_max_abs") is not None
        ]
        if not checks:
            errors.append("preflight emitted no cross-rank parameter checks")
        elif any(float(value) != 0.0 for value in checks):
            errors.append(f"cross-rank MatrixPolicy parameter mismatch: {max(map(float, checks))}")
    return {
        "row_index": int(row["row_index"]),
        "row_id": row["row_id"],
        "arm_id": row.get("arm_id", ""),
        "dataset": row["dataset"],
        "seed": int(row["seed"]),
        "status": "pass" if not errors else "invalid",
        "terminal_failure": terminal_failure,
        "errors": errors,
        "jsonl": str(path.relative_to(ROOT)),
        "jsonl_sha256": sha256(path),
        "initial_state_sha256": config.get("initial_state_sha256"),
        "train_token_sample_sha256": config.get("train_token_sample_sha256"),
        "val_token_sample_sha256": config.get("val_token_sample_sha256"),
        "first_batch_index_sha256": config.get("first_batch_index_sha256"),
        "validation_index_sha256": config.get("validation_index_sha256"),
        "slurm_node": config.get("slurm_node"),
        "slurm_restart_count": summary.get("slurm_restart_count"),
        "total_seconds": summary.get("total_seconds"),
    }


def validate_block_fingerprints(
    kind: str,
    source_rows: list[dict[str, str]],
    validated: list[dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    by_id = {row["row_id"]: row for row in validated}
    groups: defaultdict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    if kind == "e8":
        for row in source_rows:
            groups[(row["dataset"], row["seed"])].append(by_id[row["row_id"]])
    elif kind in {"e9", "e9_preflight"}:
        for row in source_rows:
            groups[(row["dataset"], row["seed"])].append(by_id[row["row_id"]])
    else:
        return errors
    shared_fields = (
        "train_token_sample_sha256",
        "val_token_sample_sha256",
        "first_batch_index_sha256",
        "validation_index_sha256",
    )
    for key, rows in groups.items():
        passing = [row for row in rows if row["status"] == "pass"]
        for field in shared_fields:
            values = {row.get(field) for row in passing}
            if len(values) > 1:
                errors.append(f"{key} has inconsistent {field}: {sorted(values)}")
        if kind == "e8":
            initial_states = {row.get("initial_state_sha256") for row in passing}
            if len(initial_states) > 1:
                errors.append(f"{key} has inconsistent MatrixPolicy initial states")
        if kind.startswith("e9"):
            rlb_values = {
                row.get("initial_state_sha256")
                for row in passing
                if row.get("arm_id") != "A0"
            }
            if len(rlb_values) > 1:
                errors.append(f"{key} has inconsistent RLB initial states")
    return errors


def validate_stage(kind: str, manifest: Path, output_root: Path) -> Path:
    subprocess.run(
        [
            str(ROOT / ".venv-cu128/bin/python"),
            str(ROOT / "experiments/scripts/build_matrixpolicy_live_stats_correction.py"),
            "verify-static-runtime",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = json.loads(FREEZE_PATH.read_text())
    manifest_sha = sha256(manifest)
    frozen_entries = [
        entry for entry in freeze["manifests"].values() if entry["sha256"] == manifest_sha
    ]
    if len(frozen_entries) != 1:
        raise SystemExit(f"manifest is not uniquely bound to the campaign freeze: {manifest}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validated = [
        validate_row(row, output_root, freeze, manifest_sha, kind) for row in rows
    ]
    errors = validate_block_fingerprints(kind, rows, validated)
    failed = [row for row in validated if row["status"] != "pass"]
    if failed:
        errors.append(f"{len(failed)} of {len(validated)} rows failed validation")
    report = {
        "campaign_id": CAMPAIGN_ID,
        "kind": kind,
        "status": "pass" if not errors else "fail",
        "freeze_sha256": freeze["freeze_sha256"],
        "runtime_sha256": freeze["runtime_sha256"],
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_sha256": manifest_sha,
        "output_root": str(output_root.relative_to(ROOT)),
        "expected_rows": len(rows),
        "passed_rows": len(rows) - len(failed),
        "terminal_failures": sum(bool(row.get("terminal_failure")) for row in validated),
        "errors": errors,
        "rows": validated,
        "validated_unix_time": time.time(),
    }
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_ROOT / f"{kind}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("kind", "status", "expected_rows", "passed_rows", "terminal_failures", "errors")}, sort_keys=True))
    if errors:
        raise SystemExit(1)
    return path


def validate_final() -> None:
    subprocess.run(
        [
            str(ROOT / ".venv-cu128/bin/python"),
            str(ROOT / "experiments/scripts/build_matrixpolicy_live_stats_correction.py"),
            "verify-static",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    freeze = json.loads(FREEZE_PATH.read_text())
    gate_path = VALIDATION_ROOT / "nccl_gate.json"
    errors: list[str] = []
    if not gate_path.is_file():
        errors.append("NCCL gate result is missing")
    else:
        gate = json.loads(gate_path.read_text())
        if gate.get("status") != "pass":
            errors.append("NCCL gate did not pass")
        if gate.get("campaign_freeze_sha256") != freeze["freeze_sha256"]:
            errors.append("NCCL gate freeze mismatch")
    for kind in ("e9_preflight", "main", "e8", "e9"):
        path = VALIDATION_ROOT / f"{kind}.json"
        if not path.is_file():
            errors.append(f"{kind} validation report is missing")
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "pass":
            errors.append(f"{kind} validation did not pass")
        if report.get("freeze_sha256") != freeze["freeze_sha256"]:
            errors.append(f"{kind} validation freeze mismatch")
        for row in report.get("rows", []):
            relative = row.get("jsonl")
            expected_sha = row.get("jsonl_sha256")
            if not relative or not expected_sha:
                errors.append(f"{kind} validation row lacks JSONL provenance")
                continue
            path = ROOT / str(relative)
            if not path.is_file() or sha256(path) != expected_sha:
                errors.append(f"{kind} JSONL changed after stage validation: {relative}")
    result = {
        "campaign_id": CAMPAIGN_ID,
        "status": "pass" if not errors else "fail",
        "freeze_sha256": freeze["freeze_sha256"],
        "errors": errors,
        "validated_unix_time": time.time(),
    }
    path = VALIDATION_ROOT / "final.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("e9_preflight", "main", "e8", "e9", "final"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.kind == "final":
        validate_final()
        return
    if args.manifest is None or args.output_root is None:
        parser.error("stage validation requires --manifest and --output-root")
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    validate_stage(args.kind, manifest, output_root)


if __name__ == "__main__":
    main()
