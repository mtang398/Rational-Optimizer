#!/usr/bin/env python3
"""Audit and summarize one paired full endpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import suite
from .row_tools import records, terminal


PACKAGE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wall_clock(path: Path, arm: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {arm} process wall clock: {path}")
    payload = json.loads(path.read_text())
    expected = {
        "schema": "rationalopt_process_wall_clock_v1",
        "arm": arm,
        "return_code": 0,
    }
    mismatch = {
        key: {"observed": payload.get(key), "required": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch or float(payload.get("elapsed_seconds", 0.0)) <= 0.0:
        raise RuntimeError(f"invalid {arm} process wall clock: {mismatch or payload}")
    return payload


def hardware_record(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing hardware audit: {path}")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "four_a6000_nvlink_p2p_allocation_audit_v1"
        or payload.get("passed") is not True
    ):
        raise RuntimeError(f"invalid hardware audit: {path}")
    return payload


def control_hardware_path(control: Path, row: dict, config: dict) -> Path:
    run_root = control.parents[3]
    candidates = sorted(
        run_root.glob(f"hardware-*-{row['source_manifest_row_index']}.json")
    )
    matches = []
    for path in candidates:
        try:
            payload = hardware_record(path)
        except (json.JSONDecodeError, RuntimeError):
            continue
        if (
            str(payload.get("slurm_job_id")) == str(config.get("slurm_job_id"))
            and payload.get("node") == config.get("slurm_node")
        ):
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            "expected one hardware audit for the selected control run, found "
            f"{len(matches)} among {len(candidates)} candidates"
        )
    return matches[0]


def bound_allocation(config: dict, clock: dict, hardware: dict, arm: str) -> dict:
    identity = {
        "summary_job_id": str(config.get("slurm_job_id")),
        "wall_clock_job_id": str(clock.get("slurm_job_id")),
        "hardware_job_id": str(hardware.get("slurm_job_id")),
        "summary_node": config.get("slurm_node"),
        "wall_clock_node": clock.get("hostname"),
        "hardware_node": hardware.get("node"),
    }
    if len({
        identity["summary_job_id"],
        identity["wall_clock_job_id"],
        identity["hardware_job_id"],
    }) != 1 or len({
        identity["summary_node"],
        identity["wall_clock_node"],
        identity["hardware_node"],
    }) != 1:
        raise RuntimeError(f"{arm} timing provenance is not allocation-bound: {identity}")
    return identity


def matched_hardware(control: dict, candidate: dict) -> dict:
    fields = (
        "schema",
        "requested_features",
        "requested_tres",
        "visible_cuda_count",
        "selected_local_rank_gpu_names",
        "every_selected_rank_has_a_direct_peer",
        "every_selected_physical_gpu_has_an_nvlink_peer",
        "nccl_p2p_disable",
        "nccl_p2p_level",
        "nccl_shm_disable",
        "torch_fallback",
    )
    comparison = {
        field: {
            "control": control.get(field),
            "candidate": candidate.get(field),
            "equal": control.get(field) == candidate.get(field),
        }
        for field in fields
    }
    if not all(item["equal"] for item in comparison.values()):
        raise RuntimeError(f"control and candidate hardware standards differ: {comparison}")
    return comparison


def manifest_row_sha256(row: dict[str, str]) -> str:
    payload = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def source_manifest_row(row: dict) -> dict[str, str]:
    manifest = PACKAGE / "activation_optimizer_manifest.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    index = int(row["source_manifest_row_index"])
    if not 0 <= index < len(rows):
        raise RuntimeError(f"source manifest row index is outside the public manifest: {index}")
    source = rows[index]
    mismatch = {
        "row_index": {"observed": source.get("row_index"), "required": str(index)},
        "row_id": {
            "observed": source.get("row_id"),
            "required": row["source_manifest_row_id"],
        },
    }
    mismatch = {
        key: value
        for key, value in mismatch.items()
        if str(value["observed"]) != str(value["required"])
    }
    if mismatch:
        raise RuntimeError(f"source manifest row mismatch: {mismatch}")
    return source


def expected_experiment_identity(row: dict, arm: str) -> str:
    if arm == "control":
        return "none"
    if arm == "candidate":
        return suite.experiment_identity(row)
    raise RuntimeError(f"unknown arm {arm!r}")


def control_source_identity_mismatch(config: dict, row: dict) -> dict:
    source = source_manifest_row(row)
    required = {
        "source_manifest_sha256": hashlib.sha256(
            (PACKAGE / "activation_optimizer_manifest.csv").read_bytes()
        ).hexdigest(),
        "source_manifest_row_index": str(row["source_manifest_row_index"]),
        "source_manifest_row_id": row["source_manifest_row_id"],
        "source_manifest_row_sha256": manifest_row_sha256(source),
    }
    mismatch = {
        key: {"observed": config.get(key), "required": value}
        for key, value in required.items()
        if str(config.get(key, "")) != str(value)
    }
    for key in ("source_freeze_sha256",):
        value = config.get(key)
        if not is_sha256_hex(value):
            mismatch[key] = {
                "observed": value,
                "required": "64 lowercase/uppercase hex characters from the reused control source",
            }
    return mismatch


def expected_config(row: dict, arm: str) -> dict:
    """Return the config fields that bind one endpoint to its matrix cell."""

    return {
        "activation": row[f"{'control' if arm == 'control' else 'candidate'}_activation"],
        "optimizer": row[f"{'control' if arm == 'control' else 'candidate'}_optimizer"],
        "fairness_contract": suite.contract(row),
        "experiment_identity": expected_experiment_identity(row, arm),
        "dataset": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_revision": row["dataset_revision"],
        "dataset_streaming": True,
        "dataset_text_column": row["dataset_text_column"],
        "train_split": row["train_split"],
        "validation_split": row["validation_split"],
        "train_skip_documents": row["train_skip_documents"],
        "validation_skip_documents": row["validation_skip_documents"],
        "train_skip_tokens": row["train_skip_tokens"],
        "validation_skip_tokens": row["validation_skip_tokens"],
        "train_tokens": row["max_train_tokens"],
        "val_tokens": row["max_val_tokens"],
        "steps": row["steps"],
        "layers": row["layers"],
        "d_model": row["d_model"],
        "heads": row["heads"],
        "ffn_dim": row["ffn_dim"],
        "seq_len": row["seq_len"],
        "batch_size_per_gpu": row["batch_size"],
        "grad_accum": row["grad_accum"],
        "global_tokens_per_step": row["global_tokens_per_step"],
        "eval_interval": row["eval_interval"],
        "eval_batches": row["eval_batches"],
        "log_interval": row["log_interval"],
        "seed": row["seed"],
        "optimizer_lr": row["lr"],
        "optimizer_min_lr": row["min_lr"],
        "warmup_steps": row["warmup_steps"],
        "optimizer_weight_decay": row["weight_decay"],
        "optimizer_beta1": row["beta1"],
        "optimizer_beta2": row["beta2"],
        "optimizer_eps": row["eps"],
        "grad_clip": row["grad_clip"],
        "init_std": row["init_std"],
        "rational_init": row["rational_init"],
        "post_rational_init": row["post_rational_init"],
        "rational_group_size": row["rational_group_size"],
        "rational_max_groups": row["rational_max_groups"],
        "probe_batch_size": row["probe_batch_size"],
        "matrix_spectrum_interval": row["matrix_spectrum_interval"],
        "telemetry_rlb_stat_every": row["telemetry_rlb_stat_every"],
        "sam_rho": row["sam_rho"],
        "sam_adaptive": row["sam_adaptive"],
        "tokenizer": row["tokenizer"],
        "tokenizer_revision": row["tokenizer_revision"],
        "world_size": 4,
        "params": suite.PARAMETER_COUNTS[(row["model"], arm)],
    }


def audited(path: Path, row: dict, arm: str):
    if not terminal(path, int(row["steps"])):
        raise RuntimeError(f"{arm} endpoint is incomplete or duplicated: {path}")
    rows = records(path)
    config = next(record for record in rows if record.get("event") == "config")
    summary = next(record for record in rows if record.get("event") == "summary")
    evals = {
        int(record["step"]): float(record["val_loss"])
        for record in rows if record.get("event") == "eval"
    }
    expected = expected_config(row, arm)
    mismatch = {
        key: {"observed": config.get(key), "required": value}
        for key, value in expected.items() if config.get(key) != value
    }
    fairness = config.get("optimizer_lr_wd_fairness", {})
    if fairness.get("contract") != suite.contract(row):
        mismatch["optimizer_lr_wd_fairness.contract"] = fairness.get("contract")
    if fairness.get("passed") is not True:
        mismatch["optimizer_lr_wd_fairness.passed"] = fairness.get("passed")
    for key, required in (
        ("base_lr", row["lr"]),
        ("minimum_lr", row["min_lr"]),
        ("base_weight_decay", row["weight_decay"]),
    ):
        if float(fairness.get(key, float("nan"))) != float(required):
            mismatch[f"optimizer_lr_wd_fairness.{key}"] = fairness.get(key)
    for group in fairness.get("groups", ()):
        if float(group.get("lr_scale", float("nan"))) != 1.0:
            mismatch[f"group_{group.get('group_index')}_lr_scale"] = group.get("lr_scale")
        if float(group.get("weight_decay", float("nan"))) not in {
            0.0, float(row["weight_decay"])
        }:
            mismatch[f"group_{group.get('group_index')}_weight_decay"] = group.get(
                "weight_decay"
            )
    for name, value in fairness.get("internal_lr_wd_scalars", {}).items():
        if float(value) != 1.0:
            mismatch[f"internal:{name}"] = value
    if arm == "control":
        mismatch.update(
            {
                f"control_source_identity.{key}": value
                for key, value in control_source_identity_mismatch(config, row).items()
            }
        )
    else:
        identity = config.get("tiller_experiment_identity", {})
        if (
            identity.get("passed") is not True
            or identity.get("matrix_index") != row["matrix_index"]
            or identity.get("source_manifest_row_index")
            != row["source_manifest_row_index"]
            or identity.get("source_manifest_row_id") != row["source_manifest_row_id"]
        ):
            mismatch["matrix_identity"] = identity
    if mismatch:
        raise RuntimeError(f"{arm} configuration mismatch: {mismatch}")
    if 1000 not in evals or int(row["steps"]) not in evals:
        raise RuntimeError(f"{arm} lacks required evaluation checkpoints")
    return config, evals, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-index", required=True, type=int)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-wall-clock", type=Path)
    parser.add_argument("--candidate-wall-clock", type=Path)
    parser.add_argument("--control-hardware", type=Path)
    parser.add_argument("--hardware", type=Path)
    args = parser.parse_args()
    if args.control_wall_clock is None:
        args.control_wall_clock = args.control.with_suffix(".wall_clock.json")
    if args.candidate_wall_clock is None:
        args.candidate_wall_clock = args.output.parent / "CANDIDATE_WALL_CLOCK.json"
    if args.hardware is None:
        args.hardware = args.output.parent / "HARDWARE.json"
    row = suite.row_at(args.matrix_index)
    control_config, control_eval, control_summary = audited(args.control, row, "control")
    candidate_config, candidate_eval, candidate_summary = audited(
        args.candidate, row, "candidate"
    )
    control_wall_clock = wall_clock(
        args.control_wall_clock, f"silu_{row['control_optimizer']}"
    )
    candidate_wall_clock = wall_clock(args.candidate_wall_clock, "candidate")
    if args.control_hardware is None:
        args.control_hardware = control_hardware_path(
            args.control, row, control_config
        )
    control_hardware = hardware_record(args.control_hardware)
    candidate_hardware = hardware_record(args.hardware)
    schedule_identity = {
        "realized_lr_audit_steps": {
            "control": control_summary.get("realized_lr_audit_steps"),
            "candidate": candidate_summary.get("realized_lr_audit_steps"),
        },
        "realized_lr_trace_sha256": {
            "control": control_summary.get("realized_lr_trace_sha256"),
            "candidate": candidate_summary.get("realized_lr_trace_sha256"),
        },
    }
    if (
        schedule_identity["realized_lr_audit_steps"]["control"] != int(row["steps"])
        or schedule_identity["realized_lr_audit_steps"]["candidate"] != int(row["steps"])
        or not schedule_identity["realized_lr_trace_sha256"]["control"]
        or schedule_identity["realized_lr_trace_sha256"]["control"]
        != schedule_identity["realized_lr_trace_sha256"]["candidate"]
    ):
        raise RuntimeError(f"realized LR schedule identity failed: {schedule_identity}")
    identity_keys = (
        "train_token_sample_sha256", "val_token_sample_sha256",
        "first_batch_index_sha256", "validation_index_sha256",
    )
    identity = {
        key: {
            "control": control_config.get(key),
            "candidate": candidate_config.get(key),
            "equal": control_config.get(key) == candidate_config.get(key),
        }
        for key in identity_keys
    }
    if not all(value["equal"] for value in identity.values()):
        raise RuntimeError(f"paired data/order identity failed: {identity}")
    endpoint = int(row["steps"])
    step1000_lead = control_eval[1000] - candidate_eval[1000]
    endpoint_lead = control_eval[endpoint] - candidate_eval[endpoint]
    control_bound_allocation = bound_allocation(
        control_summary, control_wall_clock, control_hardware, "control"
    )
    candidate_bound_allocation = bound_allocation(
        candidate_summary, candidate_wall_clock, candidate_hardware, "candidate"
    )
    hardware_standard = matched_hardware(control_hardware, candidate_hardware)
    control_elapsed = float(control_wall_clock["elapsed_seconds"])
    candidate_elapsed = float(candidate_wall_clock["elapsed_seconds"])
    endpoint_time_ratio = candidate_elapsed / control_elapsed
    result = {
        "schema": "tiller_matched_endpoint_result_v1",
        "status": "complete" if step1000_lead >= 0.0 else "invalid_not_interrupted",
        "matrix_index": row["matrix_index"],
        "source_manifest_row_index": row["source_manifest_row_index"],
        "phase": row["phase"],
        "model": row["model"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "control": row["control_name"],
        "candidate": row["candidate_name"],
        "candidate_optimizer": row["candidate_optimizer"],
        "steps": endpoint,
        "control_step1000_loss": control_eval[1000],
        "candidate_step1000_loss": candidate_eval[1000],
        "step1000_absolute_lead": step1000_lead,
        "control_endpoint_loss": control_eval[endpoint],
        "candidate_endpoint_loss": candidate_eval[endpoint],
        "absolute_endpoint_lead": endpoint_lead,
        "exact_matched_hardware_end_to_end_total_time_ratio": endpoint_time_ratio,
        "control_end_to_end_total_seconds": control_elapsed,
        "candidate_end_to_end_total_seconds": candidate_elapsed,
        "control_training_loop_total_seconds": float(control_summary["total_seconds"]),
        "candidate_training_loop_total_seconds": float(candidate_summary["total_seconds"]),
        "passes_discovery_1_20_time_gate": endpoint_time_ratio <= 1.20,
        "passes_final_1_05_time_gate": endpoint_time_ratio <= 1.05,
        "lr_wd_and_all_shared_argparse_hyperparameters_identical": True,
        "realized_lr_schedule_identity": schedule_identity,
        "control_allocation_identity": control_bound_allocation,
        "candidate_allocation_identity": candidate_bound_allocation,
        "matched_hardware_standard": hardware_standard,
        "data_and_order_identity": identity,
        "source_freeze_manifest_sha256": sha256(PACKAGE / "SOURCE_FREEZE.sha256"),
        "control_hardware_audit": {
            "path": str(args.control_hardware),
            "sha256": sha256(args.control_hardware),
        },
        "candidate_hardware_audit": {
            "path": str(args.hardware),
            "sha256": sha256(args.hardware),
        },
        "control_jsonl": {"path": str(args.control), "sha256": sha256(args.control)},
        "candidate_jsonl": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        "control_wall_clock": {
            "path": str(args.control_wall_clock),
            "sha256": sha256(args.control_wall_clock),
        },
        "candidate_wall_clock": {
            "path": str(args.candidate_wall_clock),
            "sha256": sha256(args.candidate_wall_clock),
        },
    }
    if result["status"] != "complete":
        raise RuntimeError("negative step-1,000 row was not interrupted")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n"
    )
    print(rendered, end="")


if __name__ == "__main__":
    main()
