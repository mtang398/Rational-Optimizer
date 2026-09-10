#!/usr/bin/env python3
"""Build the TILLER evaluation matrix from the repository manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
SOURCE_MANIFEST = PACKAGE / "activation_optimizer_manifest.csv"
RESULTS = PACKAGE.parent / "results"
EXACT_OPTIMIZER_KEY = "tiller_v1"
TWO_STAGE_OPTIMIZER_KEY = "tiller_then_muon_v1"
SUITE_100M = "12l_768d_100m_tokens_3050_steps"
SUITE_300M_SMALL = "12l_768d_300m_tokens_9150_steps"
SUITE_100M_LARGE = "18l_1024d_100m_tokens_3050_steps"

def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected true/false, found {value!r}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def matched_controls() -> dict[tuple[str, str, int], dict[str, object]]:
    controls: dict[tuple[str, str, int], dict[str, object]] = {}
    for phase, model, folder, tokens, steps in (
        (SUITE_100M, "12l_768d", "adamw", 100_000_000, 3_050),
        (SUITE_300M_SMALL, "12l_768d", "muon", 300_000_000, 9_150),
        (SUITE_100M_LARGE, "18l_1024d", "muon", 100_000_000, 3_050),
    ):
        path = RESULTS / folder / "runs.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        source = f"experiments/results/{folder}/runs.csv"
        for row in rows:
            if not (
                row["model_scale"] == model
                and row["source_phase"] == phase
                and int(row["train_tokens"]) == tokens
                and int(row["steps_required"]) == steps
                and row["activation"] == "silu"
                and row["optimizer"] == folder
            ):
                continue
            key = (phase, row["dataset"], int(row["seed"]))
            if key in controls:
                raise RuntimeError(f"duplicate matched control: {key}")
            controls[key] = {
                "source_result": source,
                "source_run_index": int(row["run_index"]),
                "status": row["status"],
                "endpoint_validation_loss": optional_float(
                    row["final_validation_loss"]
                ),
                "step1000_validation_loss": optional_float(
                    row["step1000_validation_loss"]
                ),
                "time_scope": row["time_scope"] or None,
                "total_seconds": optional_float(row["total_seconds"]),
            }
    return controls


def selected_rows() -> list[dict[str, str]]:
    with SOURCE_MANIFEST.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    selected = [
        row
        for row in source
        if (
            row["phase"] == SUITE_100M
            and row["method"] == "silu_adamw"
        )
        or (
            row["phase"] in {SUITE_300M_SMALL, SUITE_100M_LARGE}
            and row["method"] == "silu_muon"
        )
    ]
    expected = {
        SUITE_100M: (
            {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
            {"1337", "2027", "3407"},
            15,
        ),
        SUITE_300M_SMALL: (
            {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
            {"1337", "2027", "3407"},
            15,
        ),
        SUITE_100M_LARGE: (
            {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
            {"1337", "2027", "3407"},
            15,
        ),
    }
    for phase, (datasets, seeds, count) in expected.items():
        rows = [row for row in selected if row["phase"] == phase]
        if len(rows) != count:
            raise RuntimeError(f"{phase}: expected {count} rows, found {len(rows)}")
        if {row["dataset"] for row in rows} != datasets:
            raise RuntimeError(f"{phase}: dataset inventory changed")
        if {row["seed"] for row in rows} != seeds:
            raise RuntimeError(f"{phase}: seed inventory changed")
    return selected


def build_row(
    index: int,
    row: dict[str, str],
    controls: dict[tuple[str, str, int], dict[str, object]],
    *,
    candidate_optimizer: str = EXACT_OPTIMIZER_KEY,
) -> dict[str, object]:
    if candidate_optimizer not in {EXACT_OPTIMIZER_KEY, TWO_STAGE_OPTIMIZER_KEY}:
        raise ValueError(f"unknown candidate optimizer: {candidate_optimizer}")
    if candidate_optimizer == TWO_STAGE_OPTIMIZER_KEY and row["phase"] != SUITE_100M_LARGE:
        raise ValueError("the two-stage candidate is defined only for the 18-layer 100M suite")
    uses_adamw_control = row["phase"] == SUITE_100M
    control_key = (row["phase"], row["dataset"], int(row["seed"]))
    control = controls.get(control_key)
    if control is None:
        control = {
            "source_result": (
                "experiments/results/adamw/runs.csv"
                if uses_adamw_control else "experiments/results/muon/runs.csv"
            ),
            "source_run_index": None,
            "status": "pending",
            "endpoint_validation_loss": None,
            "step1000_validation_loss": None,
            "time_scope": None,
            "total_seconds": None,
        }
    result = {
        "matrix_index": index,
        "source_manifest_row_index": int(row["row_index"]),
        "source_manifest_row_id": row["row_id"],
        "phase": row["phase"],
        "model": row["model"],
        "dataset": row["dataset"],
        "dataset_name": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_revision": row["dataset_revision"],
        "dataset_streaming": as_bool(row["dataset_streaming"]),
        "dataset_text_column": row["text_column"],
        "train_split": row["train_split"],
        "validation_split": row["val_split"],
        "train_skip_documents": int(row["train_skip_documents"]),
        "validation_skip_documents": int(row["validation_skip_documents"]),
        "train_skip_tokens": int(row["train_skip_tokens"]),
        "validation_skip_tokens": int(row["val_skip_tokens"]),
        "cache_dir": row["cache_dir"],
        "max_train_tokens": int(row["train_tokens"]),
        "max_val_tokens": int(row["val_tokens"]),
        "steps": int(row["steps"]),
        "layers": int(row["layers"]),
        "d_model": int(row["d_model"]),
        "heads": int(row["heads"]),
        "ffn_dim": int(row["ffn_dim"]),
        "seq_len": int(row["seq_len"]),
        "batch_size": int(row["batch_size"]),
        "grad_accum": int(row["grad_accum"]),
        "global_tokens_per_step": int(row["global_tokens_per_step"]),
        "eval_interval": int(row["eval_interval"]),
        "eval_batches": int(row["eval_batches"]),
        "log_interval": int(row["log_interval"]),
        "seed": int(row["seed"]),
        "lr": float(row["lr"]),
        "min_lr": float(row["min_lr"]),
        "warmup_steps": int(row["warmup_steps"]),
        "weight_decay": float(row["weight_decay"]),
        "beta1": float(row["beta1"]),
        "beta2": float(row["beta2"]),
        "eps": float(row["eps"]),
        "grad_clip": float(row["grad_clip"]),
        "muon_momentum": float(row["muon_momentum"]),
        "muon_ns_steps": int(row["muon_ns_steps"]),
        "muon_adjust_lr_fn": row["muon_adjust_lr_fn"],
        "init_std": float(row["init_std"]),
        "rational_init": row["rational_init"],
        "post_rational_init": row["post_rational_init"],
        "rational_group_size": int(row["rational_group_size"]),
        "rational_max_groups": int(row["rational_max_groups"]),
        "probe_batch_size": int(row["probe_batch_size"]),
        "matrix_spectrum_interval": int(row["matrix_spectrum_interval"]),
        "telemetry_rlb_stat_every": int(row["telemetry_rlb_stat_every"]),
        "sam_rho": float(row["sam_rho"]),
        "sam_adaptive": as_bool(row["sam_adaptive"]),
        "tokenizer": row["tokenizer"],
        "tokenizer_revision": row["tokenizer_revision"],
        "control_activation": "silu",
        "control_optimizer": "adamw" if uses_adamw_control else "muon",
        "control_name": "SwiGLU+AdamW" if uses_adamw_control else "SwiGLU+Muon",
        "candidate_activation": "rlb_fused_global_rational",
        "candidate_optimizer": candidate_optimizer,
        "candidate_name": (
            "TILLER" if candidate_optimizer == EXACT_OPTIMIZER_KEY else "TILLER→Muon"
        ),
        "quality_action": "selected_method_endpoint",
        "timing_condition": (
            "exclusive allocation; selected 4x RTX A6000 as two NVLink pairs "
            "with NCCL P2P"
        ),
        "matched_control": control,
    }
    if candidate_optimizer == TWO_STAGE_OPTIMIZER_KEY:
        result.update(
            switch_after_step=1_000,
            optimizer_state_transition="preserve_compatible_state",
            learning_rate_schedule="single_full_horizon_cosine",
        )
    return result


def build_payload() -> dict[str, object]:
    controls = matched_controls()
    source_rows = selected_rows()
    matrix = [
        build_row(index, row, controls)
        for index, row in enumerate(source_rows)
    ]
    for row in source_rows:
        if row["phase"] == SUITE_100M_LARGE:
            matrix.append(build_row(
                len(matrix), row, controls, candidate_optimizer=TWO_STAGE_OPTIMIZER_KEY
            ))
    return {
        "schema": "tiller_evaluation_matrix_v1",
        "objective": "TILLER and TILLER-to-Muon evaluation across the published model and token-budget suites",
        "source_manifest": "experiments/protocol/activation_optimizer_manifest.csv",
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "matrix_rows": len(matrix),
        "12l_100m_suite_rows": sum(row["phase"] == SUITE_100M for row in matrix),
        "12l_300m_suite_rows": sum(row["phase"] == SUITE_300M_SMALL for row in matrix),
        "18l_100m_suite_rows": sum(row["phase"] == SUITE_100M_LARGE for row in matrix),
        "18l_100m_full_tiller_rows": sum(
            row["phase"] == SUITE_100M_LARGE
            and row["candidate_optimizer"] == EXACT_OPTIMIZER_KEY for row in matrix
        ),
        "18l_100m_tiller_then_muon_rows": sum(
            row["candidate_optimizer"] == TWO_STAGE_OPTIMIZER_KEY for row in matrix
        ),
        "rows": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "matrix.json")
    args = parser.parse_args()
    payload = build_payload()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} ({len(payload['rows'])} rows)")


if __name__ == "__main__":
    main()
