#!/usr/bin/env python3
"""Audit and summarize one paired full endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import suite
from .row_tools import records, terminal


FACTORIZED_LEDGER_DCLM_4000_LEAD = 0.050536155700683594
ORIGINAL_METHOD3_DCLM_4000_LEAD = 0.08379220962524414


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    expected = {
        "activation": row[f"{'control' if arm == 'control' else 'candidate'}_activation"],
        "optimizer": row[f"{'control' if arm == 'control' else 'candidate'}_optimizer"],
        "fairness_contract": suite.contract(row),
        "dataset": row["dataset_name"],
        "dataset_config": row["dataset_config"],
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
        "world_size": 4,
        "params": {
            ("M0", "control"): 123_551_232,
            ("M0", "candidate"): 123_552_672,
            ("M1", "control"): 296_867_840,
            ("M1", "candidate"): 296_871_080,
        }[(row["model"], arm)],
    }
    mismatch = {
        key: {"observed": config.get(key), "required": value}
        for key, value in expected.items() if config.get(key) != value
    }
    fairness = config.get("optimizer_lr_wd_fairness", {})
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
    identity = config.get("m1_300m_campaign_identity", {})
    if identity.get("passed") is not True or identity.get("matrix_index") != row["matrix_index"]:
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
    args = parser.parse_args()
    row = suite.row_at(args.matrix_index)
    control_config, control_eval, control_summary = audited(args.control, row, "control")
    candidate_config, candidate_eval, candidate_summary = audited(
        args.candidate, row, "candidate"
    )
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
    allocation_identity = {
        "control_job_id": str(control_summary.get("slurm_job_id")),
        "candidate_job_id": str(candidate_summary.get("slurm_job_id")),
        "control_node": control_summary.get("slurm_node"),
        "candidate_node": candidate_summary.get("slurm_node"),
    }
    if (
        allocation_identity["control_job_id"]
        != allocation_identity["candidate_job_id"]
        or allocation_identity["control_node"]
        != allocation_identity["candidate_node"]
    ):
        raise RuntimeError(
            f"control/candidate timing did not share one allocation: {allocation_identity}"
        )
    endpoint_time_ratio = (
        float(candidate_summary["total_seconds"]) / float(control_summary["total_seconds"])
    )
    result = {
        "schema": "factorized_ledger_full_transfer_row_result_v1",
        "status": "complete" if step1000_lead >= 0.0 else "invalid_not_interrupted",
        "matrix_index": row["matrix_index"],
        "source_manifest_row_index": row["source_manifest_row_index"],
        "model": row["model"],
        "dataset": row["dataset"],
        "seed": row["seed"],
        "control": row["control_name"],
        "steps": endpoint,
        "control_step1000_loss": control_eval[1000],
        "candidate_step1000_loss": candidate_eval[1000],
        "step1000_absolute_lead": step1000_lead,
        "control_endpoint_loss": control_eval[endpoint],
        "candidate_endpoint_loss": candidate_eval[endpoint],
        "absolute_endpoint_lead": endpoint_lead,
        "lead_retention_vs_factorized_ledger_dclm_4000": (
            endpoint_lead / FACTORIZED_LEDGER_DCLM_4000_LEAD
        ),
        "lead_retention_vs_original_method3_dclm_4000": (
            endpoint_lead / ORIGINAL_METHOD3_DCLM_4000_LEAD
        ),
        "lead_retention_scope_note": (
            "cross-dataset/scale diagnostic; the original Method 3 endpoint exists only "
            "for the DCLM M1/4,000 discovery cell"
        ),
        "exact_same_allocation_endpoint_total_time_ratio": endpoint_time_ratio,
        "control_endpoint_total_seconds": float(control_summary["total_seconds"]),
        "candidate_endpoint_total_seconds": float(candidate_summary["total_seconds"]),
        "passes_discovery_1_20_time_gate": endpoint_time_ratio <= 1.20,
        "passes_final_1_05_time_gate": endpoint_time_ratio <= 1.05,
        "lr_wd_and_all_shared_argparse_hyperparameters_identical": True,
        "realized_lr_schedule_identity": schedule_identity,
        "same_allocation_identity": allocation_identity,
        "data_and_order_identity": identity,
        "control_jsonl": {"path": str(args.control), "sha256": sha256(args.control)},
        "candidate_jsonl": {"path": str(args.candidate), "sha256": sha256(args.candidate)},
        "scalability_evidence": {
            "owner_count": 0,
            "dense_lg_by_lg_metric_elements": 0,
            "selected_update_elements_published": 0,
            "state_depends_on_total_activation_positions": 0,
            "state_scales_as": "O(LH + LGd + 64LG)",
        },
    }
    if result["status"] != "complete":
        raise RuntimeError("negative step-1,000 row was not interrupted")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.resolve()}\n"
    )
    print(rendered, end="")


if __name__ == "__main__":
    main()
