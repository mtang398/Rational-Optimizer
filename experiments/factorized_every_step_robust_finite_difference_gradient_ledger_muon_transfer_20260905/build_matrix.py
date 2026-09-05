#!/usr/bin/env python3
"""Derive the locked full-method transfer matrix from the GitHub controls."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
SOURCE_REPO = ROOT
SOURCE_MANIFEST = SOURCE_REPO / "experiments/manifests/iclr26_main_manifest.csv"
SOURCE_COMMIT = "dd97429cd9458c359dfe0d7e576d81aafe334bec"
EXACT_OPTIMIZER_KEY = "factorized_every_step_rfd_gradient_ledger_muon_v1"

M0_CONTROLS = {
    "dclm": {"step1000_mean": 4.9398, "endpoint_mean": 4.4056, "endpoint_std": 0.0099},
    "fineweb_edu": {"step1000_mean": 4.8348, "endpoint_mean": 4.2375, "endpoint_std": 0.0086},
    "fineweb": {"step1000_mean": 5.0366, "endpoint_mean": 4.4758, "endpoint_std": 0.0097},
    "dolma_sample": {"step1000_mean": 5.0809, "endpoint_mean": 4.4862, "endpoint_std": 0.0012},
    "c4_en": {"step1000_mean": 5.0231, "endpoint_mean": 4.4469, "endpoint_std": 0.0160},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observed_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"authoritative repository changed: {observed_commit} != {SOURCE_COMMIT}"
        )
    with SOURCE_MANIFEST.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    selected = [
        row for row in source
        if row["method"] == "silu_adamw"
        and row["phase"] in {"E1_m0_100m", "E3_m1_300m"}
    ]
    expected = {
        "E1_m0_100m": {
            "datasets": {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
            "seeds": {"1337", "2027", "3407"},
            "count": 15,
        },
        "E3_m1_300m": {
            "datasets": {"dclm", "fineweb_edu", "c4_en"},
            "seeds": {"1337", "2027", "3407"},
            "count": 9,
        },
    }
    for phase, contract in expected.items():
        rows = [row for row in selected if row["phase"] == phase]
        if len(rows) != contract["count"]:
            raise RuntimeError(f"{phase} row count changed")
        if {row["dataset"] for row in rows} != contract["datasets"]:
            raise RuntimeError(f"{phase} dataset inventory changed")
        if {row["seed"] for row in rows} != contract["seeds"]:
            raise RuntimeError(f"{phase} seed inventory changed")

    matrix = []
    for index, row in enumerate(selected):
        m0 = row["phase"] == "E1_m0_100m"
        item = {
            "matrix_index": index,
            "source_manifest_row_index": int(row["row_index"]),
            "source_manifest_row_id": row["row_id"],
            "phase": row["phase"],
            "model": row["model"],
            "dataset": row["dataset"],
            "dataset_name": row["dataset_name"],
            "dataset_config": row["dataset_config"],
            "dataset_streaming": True,
            "dataset_text_column": row["text_column"],
            "train_split": row["train_split"],
            "validation_split": row["val_split"],
            "train_skip_documents": 0,
            "validation_skip_documents": 0,
            "train_skip_tokens": 0,
            "validation_skip_tokens": int(row["val_skip_tokens"]),
            "cache_dir": f"experiments/cache/tokens_iclr26_main/{row['dataset']}",
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
            "log_interval": 10,
            "seed": int(row["seed"]),
            "lr": float(row["lr"]),
            "min_lr": float(row["min_lr"]),
            "warmup_steps": 200,
            "weight_decay": float(row["weight_decay"]),
            "beta1": 0.90,
            "beta2": 0.95,
            "eps": 1.0e-8,
            "grad_clip": 1.0,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "init_std": 0.02,
            "rational_init": "silu",
            "post_rational_init": "identity",
            "rational_group_size": 256,
            "rational_max_groups": 32,
            # M0 follows GitHub E1 exactly. M1 follows the already-frozen
            # 9,150-step transfer protocol, generalized only over E3 data/seeds.
            "probe_batch_size": 1 if m0 else 0,
            "matrix_spectrum_interval": 250 if m0 else 0,
            # The exact established M0 and M1 endpoint launchers both record
            # the trainer-default cadence of four.  It is diagnostic only and
            # is identical between the paired arms.
            "telemetry_rlb_stat_every": 4,
            "sam_rho": 0.0,
            "sam_adaptive": False,
            "control_activation": "silu",
            "control_optimizer": "adamw" if m0 else "muon",
            "control_name": "SwiGLU+AdamW" if m0 else "SwiGLU+Muon",
            "candidate_activation": "rlb_fused_global_rational",
            "candidate_optimizer": EXACT_OPTIMIZER_KEY,
            "candidate_name": (
                "Factorized Every-Step Robust Finite-Difference "
                "Gradient-Ledger Muon"
            ),
            "quality_action": "fresh_matched_control_then_fresh_candidate",
            "timing_condition": (
                "exclusive 4x RTX A6000 on an NVLink-capable node; no named-node "
                "or exclusion pin; NCCL P2P enabled; paired endpoints share allocation"
            ),
            "m0_published_control": M0_CONTROLS[row["dataset"]] if m0 else None,
        }
        matrix.append(item)
    payload = {
        "schema": "factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_matrix_v1",
        "objective": (
            "fresh Factorized Every-Step Robust Finite-Difference Gradient-Ledger "
            "Muon and matched control on every locked M0/M1 dataset and seed"
        ),
        "source_repo": str(SOURCE_REPO),
        "source_commit": SOURCE_COMMIT,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "matrix_rows": len(matrix),
        "m0_rows": sum(row["model"] == "M0" for row in matrix),
        "m1_rows": sum(row["model"] == "M1" for row in matrix),
        "rows": matrix,
    }
    output = PACKAGE / "matrix.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} ({len(matrix)} rows)")


if __name__ == "__main__":
    main()
