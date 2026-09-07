#!/usr/bin/env python3
"""Verify the public experiment package without importing training dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent

try:
    from . import build_source_freeze
except ImportError:  # Support direct execution from the repository root.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.protocol import build_source_freeze  # type: ignore[no-redef]


RESULTS = ROOT / "experiments" / "results"
MANIFEST = PACKAGE / "activation_optimizer_manifest.csv"
MATRIX = PACKAGE / "matrix.json"
TOKEN_FINGERPRINTS = PACKAGE / "token_fingerprints.json"
FREEZE = PACKAGE / "SOURCE_FREEZE.sha256"

GRAIN_ID = "rlb_fused_global_rational"
ACTIVATION_DISPLAY = {"silu": "SwiGLU", GRAIN_ID: "GRAIN"}
SMALL_MODEL = "12l_768d"
LARGE_MODEL = "18l_1024d"
PREFLIGHT_SUITE = "12l_768d_preflight_2621440_tokens_80_steps"
SUITE_100M = "12l_768d_100m_tokens_3050_steps"
SUITE_300M_SMALL = "12l_768d_300m_tokens_9150_steps"
SUITE_300M_LARGE = "18l_1024d_300m_tokens_9150_steps"

MANIFEST_FIELDS = [
    "row_index", "row_id", "phase", "dataset", "dataset_name",
    "dataset_config", "dataset_revision", "dataset_streaming", "text_column",
    "train_split", "val_split", "train_skip_documents",
    "validation_skip_documents", "train_skip_tokens", "val_skip_tokens",
    "cache_dir", "model", "layers", "d_model", "heads", "ffn_dim",
    "seq_len", "batch_size", "grad_accum", "global_tokens_per_step",
    "train_tokens", "val_tokens", "steps", "eval_interval", "eval_batches",
    "log_interval", "seed", "method", "activation", "optimizer", "lr",
    "min_lr", "weight_decay", "warmup_steps", "beta1", "beta2", "eps",
    "grad_clip", "muon_momentum", "muon_ns_steps", "muon_adjust_lr_fn",
    "init_std", "rational_init", "post_rational_init",
    "rational_group_size", "rational_max_groups", "probe_batch_size",
    "matrix_spectrum_interval", "telemetry_rlb_stat_every", "sam_rho",
    "sam_adaptive", "tokenizer", "tokenizer_revision", "extra_args",
]

RUN_FIELDS = [
    "run_index", "model_scale", "dataset", "seed", "train_tokens",
    "steps_required", "steps_completed", "status", "activation",
    "activation_display_name", "optimizer", "optimizer_display_name",
    "method", "method_display_name", "step1000_validation_loss",
    "final_validation_loss", "final_validation_perplexity",
    "matched_control_step1000_validation_loss",
    "matched_control_validation_loss", "lead_at_step1000_vs_matched_control",
    "lead_vs_matched_control", "time_scope", "total_seconds",
    "matched_control_total_seconds", "total_time_ratio_vs_matched_control",
    "training_loop_total_seconds", "mean_seconds_per_step",
    "tokens_per_second", "stopped_early", "lr_wd_fairness_passed",
    "realized_lr_trace_sha256", "slurm_job_id", "slurm_restart_count",
    "slurm_node", "timing_attempt_id", "source_phase", "source_row_index",
    "source_row_id", "source_jsonl", "source_jsonl_sha256",
]

SUMMARY_FIELDS = [
    "model_scale", "dataset", "train_tokens", "steps_required", "activation",
    "activation_display_name", "optimizer", "optimizer_display_name", "method",
    "method_display_name", "runs_total", "runs_reached_required_step",
    "runs_stopped_early", "runs_non_finite", "endpoint_seed_count",
    "validation_loss_mean", "validation_loss_sample_std",
    "matched_control_seed_count", "matched_control_validation_loss_mean",
    "lead_vs_matched_control_mean", "lead_vs_matched_control_sample_std",
    "exact_time_ratio_seed_count", "exact_total_time_ratio_mean",
    "exact_total_time_ratio_sample_std", "total_seconds_mean",
    "total_seconds_sample_std", "mean_seconds_per_step",
    "tokens_per_second_mean", "status",
]

CHECKPOINT_FIELDS = [
    "run_index", "model_scale", "dataset", "seed", "activation",
    "activation_display_name", "optimizer", "optimizer_display_name", "method",
    "method_display_name", "step", "validation_loss", "validation_perplexity",
    "active_seconds_at_validation",
]

OPTIMIZERS = {
    "adamw": ("adamw", "AdamW", "adamw"),
    "muon": ("muon", "Muon", "muon"),
    "lion": ("lion", "Lion", "lion"),
    "soap": ("soap_adamw", "SOAP", "soap"),
    "ademamix": ("ademamix", "ADeMaMix", "ademamix"),
    "came": ("adafactor_came", "CAME", "came"),
    "schedule_free_adamw": (
        "schedule_free_adamw", "Schedule-Free AdamW", "schedulefree"
    ),
    "tiller": ("tiller_v1", "TILLER", "tiller"),
}

PHASES = {
    PREFLIGHT_SUITE: {
        "count": 10,
        "datasets": {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
        "seeds": {"1337"},
        "optimizers": {"adamw"},
        "model": SMALL_MODEL,
        "train_tokens": "2621440",
        "steps": "80",
    },
    SUITE_100M: {
        "count": 210,
        "datasets": {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
        "seeds": {"1337", "2027", "3407"},
        "optimizers": {
            "adamw", "muon", "lion", "soap_adamw", "ademamix",
            "adafactor_came", "schedule_free_adamw",
        },
        "model": SMALL_MODEL,
        "train_tokens": "100000000",
        "steps": "3050",
    },
    SUITE_300M_SMALL: {
        "count": 210,
        "datasets": {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
        "seeds": {"1337", "2027", "3407"},
        "optimizers": {
            "adamw", "muon", "lion", "soap_adamw", "ademamix",
            "adafactor_came", "schedule_free_adamw",
        },
        "model": SMALL_MODEL,
        "train_tokens": "300000000",
        "steps": "9150",
    },
    SUITE_300M_LARGE: {
        "count": 210,
        "datasets": {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
        "seeds": {"1337", "2027", "3407"},
        "optimizers": {
            "adamw", "muon", "lion", "soap_adamw", "ademamix",
            "adafactor_came", "schedule_free_adamw",
        },
        "model": LARGE_MODEL,
        "train_tokens": "300000000",
        "steps": "9150",
    },
}

OPTIMIZER_METHOD_SUFFIX = {
    "adamw": "adamw",
    "muon": "muon",
    "lion": "lion",
    "soap_adamw": "soap",
    "ademamix": "ademamix",
    "adafactor_came": "came",
    "schedule_free_adamw": "schedulefree",
}

MODEL_FIELDS = {
    SMALL_MODEL: {
        "layers": "12", "d_model": "768", "heads": "12",
        "ffn_dim": "2048", "seq_len": "256", "batch_size": "16",
        "grad_accum": "2", "global_tokens_per_step": "32768",
    },
    LARGE_MODEL: {
        "layers": "18", "d_model": "1024", "heads": "16",
        "ffn_dim": "3072", "seq_len": "256", "batch_size": "8",
        "grad_accum": "4", "global_tokens_per_step": "32768",
    },
}

DATASET_REVISIONS = {
    "dclm": "a3b142c183aebe5af344955ae20836eb34dcf69b",
    "fineweb_edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "dolma_sample": "7f48140530a023e9ea4c5cfb141160922727d4d3",
    "c4_en": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
}
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXACT_LR_WD_CONTRACT = "exact_lr_wd_v1"

MATRIX_TO_MANIFEST = {
    "phase": "phase",
    "model": "model",
    "dataset": "dataset",
    "dataset_name": "dataset_name",
    "dataset_config": "dataset_config",
    "dataset_revision": "dataset_revision",
    "dataset_streaming": "dataset_streaming",
    "dataset_text_column": "text_column",
    "train_split": "train_split",
    "validation_split": "val_split",
    "train_skip_documents": "train_skip_documents",
    "validation_skip_documents": "validation_skip_documents",
    "train_skip_tokens": "train_skip_tokens",
    "validation_skip_tokens": "val_skip_tokens",
    "cache_dir": "cache_dir",
    "max_train_tokens": "train_tokens",
    "max_val_tokens": "val_tokens",
    "steps": "steps",
    "layers": "layers",
    "d_model": "d_model",
    "heads": "heads",
    "ffn_dim": "ffn_dim",
    "seq_len": "seq_len",
    "batch_size": "batch_size",
    "grad_accum": "grad_accum",
    "global_tokens_per_step": "global_tokens_per_step",
    "eval_interval": "eval_interval",
    "eval_batches": "eval_batches",
    "log_interval": "log_interval",
    "seed": "seed",
    "lr": "lr",
    "min_lr": "min_lr",
    "warmup_steps": "warmup_steps",
    "weight_decay": "weight_decay",
    "beta1": "beta1",
    "beta2": "beta2",
    "eps": "eps",
    "grad_clip": "grad_clip",
    "muon_momentum": "muon_momentum",
    "muon_ns_steps": "muon_ns_steps",
    "muon_adjust_lr_fn": "muon_adjust_lr_fn",
    "init_std": "init_std",
    "rational_init": "rational_init",
    "post_rational_init": "post_rational_init",
    "rational_group_size": "rational_group_size",
    "rational_max_groups": "rational_max_groups",
    "probe_batch_size": "probe_batch_size",
    "matrix_spectrum_interval": "matrix_spectrum_interval",
    "telemetry_rlb_stat_every": "telemetry_rlb_stat_every",
    "sam_rho": "sam_rho",
    "sam_adaptive": "sam_adaptive",
    "tokenizer": "tokenizer",
    "tokenizer_revision": "tokenizer_revision",
}

INTEGER_MANIFEST_FIELDS = {
    "train_skip_documents", "validation_skip_documents", "train_skip_tokens",
    "val_skip_tokens", "train_tokens", "val_tokens", "steps", "layers",
    "d_model", "heads", "ffn_dim", "seq_len", "batch_size", "grad_accum",
    "global_tokens_per_step", "eval_interval", "eval_batches", "log_interval",
    "seed", "warmup_steps", "muon_ns_steps", "rational_group_size",
    "rational_max_groups", "probe_batch_size", "matrix_spectrum_interval",
    "telemetry_rlb_stat_every",
}
FLOAT_MANIFEST_FIELDS = {
    "lr", "min_lr", "weight_decay", "beta1", "beta2", "eps", "grad_clip",
    "muon_momentum", "init_std", "sam_rho",
}
BOOL_MANIFEST_FIELDS = {"dataset_streaming", "sam_adaptive"}

RUN_STATUSES = {"complete", "incomplete", "pending", "stopped_early", "non_finite"}
SUMMARY_STATUSES = RUN_STATUSES | {"endpoint_loss_unavailable"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
WORKSPACE_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/" + r"(?:home|Users|workspace|workspaces|scratch|mnt)/[^\s\"'<>]+"
)


class VerificationError(RuntimeError):
    """Raised when a public reproducibility invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    require(path.is_file(), f"missing {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, f"unexpected schema in {path.relative_to(ROOT)}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"extra columns in {path.relative_to(ROOT)}")
    require(all(None not in row.values() for row in rows), f"short row in {path.relative_to(ROOT)}")
    return rows


def _as_int(value: str, context: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise VerificationError(f"{context}: expected integer, found {value!r}") from exc


def _as_float(value: str, context: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise VerificationError(f"{context}: expected number, found {value!r}") from exc


def _as_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise VerificationError(f"{context}: expected true/false, found {value!r}")


def _manifest_value(row: dict[str, str], field: str) -> Any:
    value = row[field]
    if field in INTEGER_MANIFEST_FIELDS:
        return _as_int(value, f"{row['row_id']}/{field}")
    if field in FLOAT_MANIFEST_FIELDS:
        return _as_float(value, f"{row['row_id']}/{field}")
    if field in BOOL_MANIFEST_FIELDS:
        return _as_bool(value, f"{row['row_id']}/{field}")
    return value


def verify_manifest() -> list[dict[str, str]]:
    rows = read_csv(MANIFEST, MANIFEST_FIELDS)
    require(len(rows) == 640, f"manifest must contain 640 rows, found {len(rows)}")
    require(
        [row["row_index"] for row in rows] == [str(index) for index in range(640)],
        "manifest row_index must be the ordered range 0..639",
    )
    ids = [row["row_id"] for row in rows]
    require(len(ids) == len(set(ids)), "manifest row_id values are not unique")
    require(
        Counter(row["phase"] for row in rows)
        == Counter({phase: spec["count"] for phase, spec in PHASES.items()}),
        "manifest phase counts changed",
    )

    observed_cells: set[tuple[str, str, str, str, str, str]] = set()
    grouped: dict[
        tuple[str, str, str, str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in rows:
        phase = row["phase"]
        require(phase in PHASES, f"unknown phase in {row['row_id']}")
        spec = PHASES[phase]
        require(row["dataset"] in spec["datasets"], f"unexpected dataset in {row['row_id']}")
        require(row["seed"] in spec["seeds"], f"unexpected seed in {row['row_id']}")
        require(row["optimizer"] in spec["optimizers"], f"unexpected optimizer in {row['row_id']}")
        require(row["model"] == spec["model"], f"wrong model in {row['row_id']}")
        require(row["train_tokens"] == spec["train_tokens"], f"wrong token budget in {row['row_id']}")
        require(row["steps"] == spec["steps"], f"wrong endpoint in {row['row_id']}")
        require(
            row["dataset_revision"] == DATASET_REVISIONS[row["dataset"]],
            f"unpinned dataset revision in {row['row_id']}",
        )
        require(
            row["tokenizer_revision"] == TOKENIZER_REVISION,
            f"unpinned tokenizer revision in {row['row_id']}",
        )
        for field, expected in MODEL_FIELDS[row["model"]].items():
            require(row[field] == expected, f"wrong {field} in {row['row_id']}")

        if row["method"].startswith("rlb_"):
            require(row["activation"] == GRAIN_ID, f"GRAIN ID mismatch in {row['row_id']}")
        elif row["method"].startswith("silu_"):
            require(row["activation"] == "silu", f"SwiGLU ID mismatch in {row['row_id']}")
        else:
            raise VerificationError(f"unknown activation method in {row['row_id']}")

        suffix = OPTIMIZER_METHOD_SUFFIX[row["optimizer"]]
        require(
            row["method"] in {f"silu_{suffix}", f"rlb_{suffix}"},
            f"method/optimizer mismatch in {row['row_id']}",
        )
        key = (
            phase, row["dataset"], row["model"], row["train_tokens"],
            row["seed"], row["optimizer"],
        )
        grouped[key].append(row)
        observed_cells.add(key)

    expected_cells = {
        (phase, dataset, str(spec["model"]), str(spec["train_tokens"]), seed, optimizer)
        for phase, spec in PHASES.items()
        for dataset in spec["datasets"]
        for seed in spec["seeds"]
        for optimizer in spec["optimizers"]
    }
    require(observed_cells == expected_cells, "manifest matched-cell inventory changed")

    permitted_differences = {"row_index", "row_id", "method", "activation"}
    for key, pair in grouped.items():
        require(len(pair) == 2, f"matched cell {key} contains {len(pair)} rows")
        require(
            {row["activation"] for row in pair} == {"silu", GRAIN_ID},
            f"matched cell {key} does not contain both activation arms",
        )
        left, right = pair
        mismatches = {
            field: (left[field], right[field])
            for field in MANIFEST_FIELDS
            if field not in permitted_differences and left[field] != right[field]
        }
        require(not mismatches, f"matched cell {key} differs in shared fields: {mismatches}")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_token_fingerprints() -> dict[str, Any]:
    require(TOKEN_FINGERPRINTS.is_file(), "token_fingerprints.json is missing")
    try:
        payload = json.loads(TOKEN_FINGERPRINTS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError("token_fingerprints.json is not valid UTF-8 JSON") from exc
    require(
        payload.get("schema") == "rationalopt_token_fingerprints_v2",
        "token-fingerprint schema changed",
    )
    require(payload.get("tokenizer") == "gpt2", "token-fingerprint tokenizer changed")
    require(
        payload.get("tokenizer_revision") == TOKENIZER_REVISION,
        "token-fingerprint tokenizer revision changed",
    )
    cells = payload.get("cells")
    require(isinstance(cells, list) and len(cells) == 10, "expected ten token-fingerprint cells")
    expected = {
        (dataset, train_tokens, validation_tokens)
        for dataset in PHASES[SUITE_100M]["datasets"]
        for train_tokens, validation_tokens in (
            (100_000_000, 4_000_000),
            (300_000_000, 8_000_000),
        )
    }
    observed: set[tuple[str, int, int]] = set()
    hash_fields = (
        "train_token_content_sha256",
        "validation_token_content_sha256",
        "train_token_sample_sha256",
        "validation_token_sample_sha256",
    )
    for index, cell in enumerate(cells):
        require(isinstance(cell, dict), f"token-fingerprint cell {index} is not an object")
        key = (
            str(cell.get("dataset")),
            _as_int(
                str(cell.get("train_tokens", "")),
                f"token-fingerprint cell {index}/train_tokens",
            ),
            _as_int(
                str(cell.get("validation_tokens", "")),
                f"token-fingerprint cell {index}/validation_tokens",
            ),
        )
        require(key in expected, f"unexpected token-fingerprint cell: {key}")
        require(key not in observed, f"duplicate token-fingerprint cell: {key}")
        observed.add(key)
        for field in hash_fields:
            require(
                isinstance(cell.get(field), str)
                and HEX64.fullmatch(cell[field]) is not None,
                f"token-fingerprint cell {key} has malformed {field}",
            )
    require(observed == expected, "token-fingerprint cell inventory changed")
    return payload


def verify_matrix(manifest_rows: list[dict[str, str]]) -> dict[str, Any]:
    require(MATRIX.is_file(), "matrix.json is missing")
    try:
        payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError("matrix.json is not valid UTF-8 JSON") from exc
    require(isinstance(payload, dict), "matrix payload must be an object")
    rows = payload.get("rows")
    require(isinstance(rows, list), "matrix rows must be a list")
    require(payload.get("schema") == "tiller_evaluation_matrix_v1", "matrix schema changed")
    require(payload.get("source_manifest") == "experiments/protocol/activation_optimizer_manifest.csv", "matrix source path changed")
    require(payload.get("source_manifest_sha256") == _sha256(MANIFEST), "matrix source-manifest hash is stale")
    require(payload.get("matrix_rows") == 45 == len(rows), "matrix must contain 45 rows")
    require(payload.get("12l_100m_suite_rows") == 15, "matrix must contain 15 12-layer/100M-token rows")
    require(payload.get("12l_300m_suite_rows") == 15, "matrix must contain 15 12-layer/300M-token rows")
    require(payload.get("18l_300m_suite_rows") == 15, "matrix must contain 15 18-layer/300M-token rows")
    require([row.get("matrix_index") for row in rows] == list(range(45)), "matrix_index must be the ordered range 0..44")

    expected_inventory = {
        (SUITE_100M, SMALL_MODEL, dataset, seed)
        for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
        for seed in (1337, 2027, 3407)
    } | {
        (SUITE_300M_SMALL, SMALL_MODEL, dataset, seed)
        for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
        for seed in (1337, 2027, 3407)
    } | {
        (SUITE_300M_LARGE, LARGE_MODEL, dataset, seed)
        for dataset in ("dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en")
        for seed in (1337, 2027, 3407)
    }
    require(
        {(row.get("phase"), row.get("model"), row.get("dataset"), row.get("seed")) for row in rows}
        == expected_inventory,
        "matrix dataset/seed inventory changed",
    )

    by_id = {row["row_id"]: row for row in manifest_rows}
    by_index = {int(row["row_index"]): row for row in manifest_rows}
    control_tables = {
        SUITE_100M: read_csv(RESULTS / "adamw" / "runs.csv", RUN_FIELDS),
        SUITE_300M_SMALL: read_csv(RESULTS / "muon" / "runs.csv", RUN_FIELDS),
        SUITE_300M_LARGE: read_csv(RESULTS / "muon" / "runs.csv", RUN_FIELDS),
    }
    arm_fields = {
        "control_activation", "control_optimizer", "control_name",
        "candidate_activation", "candidate_optimizer", "candidate_name",
    }
    for row in rows:
        require(isinstance(row, dict), "each matrix row must be an object")
        context = f"matrix row {row.get('matrix_index')}"
        require(set(MATRIX_TO_MANIFEST) <= set(row), f"{context}: shared fields are missing")
        source_id = row.get("source_manifest_row_id")
        source_index = row.get("source_manifest_row_index")
        require(source_id in by_id, f"{context}: source row id is unknown")
        require(source_index in by_index, f"{context}: source row index is unknown")
        source = by_id[source_id]
        require(source is by_index[source_index], f"{context}: source id/index disagree")

        is_100m_suite = row["phase"] == SUITE_100M
        expected_method = "silu_adamw" if is_100m_suite else "silu_muon"
        require(source["method"] == expected_method, f"{context}: wrong matched control source")
        for matrix_field, manifest_field in MATRIX_TO_MANIFEST.items():
            expected = _manifest_value(source, manifest_field)
            require(
                row[matrix_field] == expected,
                f"{context}: {matrix_field} differs from source manifest",
            )

        expected_control = "adamw" if is_100m_suite else "muon"
        expected_control_name = "SwiGLU+AdamW" if is_100m_suite else "SwiGLU+Muon"
        require(row.get("control_activation") == "silu", f"{context}: wrong control activation")
        require(row.get("control_optimizer") == expected_control, f"{context}: wrong control optimizer")
        require(row.get("control_name") == expected_control_name, f"{context}: wrong control name")
        require(row.get("candidate_activation") == GRAIN_ID, f"{context}: wrong GRAIN trainer ID")
        require(row.get("candidate_optimizer") == "tiller_v1", f"{context}: wrong TILLER key")
        require(row.get("candidate_name") == "TILLER", f"{context}: wrong TILLER display name")

        matched = row.get("matched_control")
        require(isinstance(matched, dict), f"{context}: exact matched control is missing")
        require(
            set(matched) == {
                "source_result", "source_run_index", "endpoint_validation_loss",
                "step1000_validation_loss", "time_scope", "total_seconds", "status",
            },
            f"{context}: matched-control schema changed",
        )
        control_folder = "adamw" if is_100m_suite else "muon"
        expected_result_path = f"experiments/results/{control_folder}/runs.csv"
        require(
            matched["source_result"] == expected_result_path,
            f"{context}: matched-control result source changed",
        )
        controls = [
            item for item in control_tables[row["phase"]]
            if item["model_scale"] == row["model"]
            and item["dataset"] == row["dataset"]
            and item["seed"] == str(row["seed"])
            and item["train_tokens"] == str(row["max_train_tokens"])
            and item["steps_required"] == str(row["steps"])
            and item["activation"] == "silu"
            and item["optimizer"] == expected_control
        ]
        require(len(controls) == 1, f"{context}: matched-control run index is not unique")
        control = controls[0]
        require(matched["status"] == control["status"], f"{context}: matched control status changed")
        for result_field, expected_value in (
            ("model_scale", row["model"]),
            ("dataset", row["dataset"]),
            ("seed", str(row["seed"])),
            ("train_tokens", str(row["max_train_tokens"])),
            ("steps_required", str(row["steps"])),
            ("activation", "silu"),
            ("optimizer", expected_control),
        ):
            require(
                control[result_field] == expected_value,
                f"{context}: exact control {result_field} changed",
            )
        require(
            matched["source_run_index"] is not None
            and int(matched["source_run_index"]) == int(control["run_index"]),
            f"{context}: control result row differs",
        )
        expected_step1000 = (
            None if control["step1000_validation_loss"] == ""
            else float(control["step1000_validation_loss"])
        )
        expected_total = (
            None if control["total_seconds"] == ""
            else float(control["total_seconds"])
        )
        expected_endpoint = (
            None if control["final_validation_loss"] == ""
            else float(control["final_validation_loss"])
        )
        require(
            matched["endpoint_validation_loss"] == expected_endpoint,
            f"{context}: matched endpoint differs from its result row",
        )
        require(
            matched["step1000_validation_loss"] == expected_step1000,
            f"{context}: matched step-1000 value differs from its exact result row",
        )
        require(
            matched["time_scope"] == (control["time_scope"] or None)
            and matched["total_seconds"] == expected_total,
            f"{context}: matched timing differs from its exact result row",
        )

        unexpected_arm_fields = {
            key for key in row
            if (key.startswith("control_") or key.startswith("candidate_"))
            and key not in arm_fields
        }
        require(
            not unexpected_arm_fields,
            f"{context}: arm-specific shared overrides found: {sorted(unexpected_arm_fields)}",
        )
        require(row["lr"] == 3e-4, f"{context}: LR changed")
        require(row["min_lr"] == 3e-5, f"{context}: minimum LR changed")
        require(row["weight_decay"] == 0.1, f"{context}: weight decay changed")
        require(row["warmup_steps"] == 200, f"{context}: warmup changed")
        require(
            (row["beta1"], row["beta2"], row["eps"], row["grad_clip"])
            == (0.9, 0.95, 1e-8, 1.0),
            f"{context}: shared optimizer fields changed",
        )
        if is_100m_suite:
            expected_budget = (3050, 100_000_000, 1, 250)
        elif row["model"] == SMALL_MODEL:
            expected_budget = (9150, 300_000_000, 1, 250)
        else:
            expected_budget = (9150, 300_000_000, 0, 0)
        require(
            (
                row["steps"], row["max_train_tokens"], row["probe_batch_size"],
                row["matrix_spectrum_interval"],
            ) == expected_budget,
            f"{context}: model-scale budget changed",
        )
    return payload


def _load_suite_without_training_dependencies() -> types.ModuleType:
    """Load argument assembly with a tiny trainer stub, never importing torch."""
    suite_path = PACKAGE / "suite.py"
    require(suite_path.is_file(), "suite.py is missing")
    package_stub = types.ModuleType("training")
    trainer_stub = types.ModuleType("training.train")
    trainer_stub.TILLER_LR_WD_CONTRACT = EXACT_LR_WD_CONTRACT
    trainer_stub.TILLER_EXPERIMENT_IDENTITY = "tiller_matrix_v1"
    trainer_stub.audit_optimizer_lr_wd_fairness = lambda *args, **kwargs: None
    package_stub.train = trainer_stub  # type: ignore[attr-defined]

    sentinel = object()
    previous = {
        name: sys.modules.get(name, sentinel)
        for name in ("training", "training.train", "_rationalopt_verified_suite")
    }
    sys.modules["training"] = package_stub
    sys.modules["training.train"] = trainer_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_rationalopt_verified_suite", suite_path
        )
        require(spec is not None and spec.loader is not None, "cannot load suite.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            f"suite.py cannot assemble runs from matrix.json: {exc}"
        ) from exc
    finally:
        for name, old_value in previous.items():
            if old_value is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value  # type: ignore[assignment]


def verify_launcher_pairing(matrix_payload: dict[str, Any]) -> None:
    """Ensure the real launcher gives both arms one identical shared argv."""
    suite = _load_suite_without_training_dependencies()
    arm_specific_flags = ("--activation", "--optimizer", "--run-name")
    output_root = Path("experiments/runs/reproducibility_check")

    def normalized(argv: list[str], context: str) -> list[str]:
        require(bool(argv), f"{context}: empty training argv")
        result = [str(value) for value in argv]
        result[0] = "<arm-run-name>"
        for flag in arm_specific_flags:
            require(result.count(flag) == 1, f"{context}: expected one {flag}")
            index = result.index(flag)
            require(index + 1 < len(result), f"{context}: {flag} lacks a value")
            result[index + 1] = "<arm-specific>"
        return result

    for row in matrix_payload["rows"]:
        context = f"matrix row {row['matrix_index']}"
        try:
            control = suite.training_argv(row, "control", output_root)
            candidate = suite.training_argv(row, "candidate", output_root)
        except Exception as exc:
            raise VerificationError(f"{context}: training argv assembly failed: {exc}") from exc
        require(
            normalized(control, f"{context}/control")
            == normalized(candidate, f"{context}/candidate"),
            f"{context}: control and TILLER launcher arguments differ beyond identity",
        )
        for arm, argv in (("control", control), ("candidate", candidate)):
            require(
                argv.count("--fairness-contract") == 1,
                f"{context}/{arm}: expected one LR/WD contract flag",
            )
            contract_index = argv.index("--fairness-contract")
            require(
                argv[contract_index + 1] == EXACT_LR_WD_CONTRACT,
                f"{context}/{arm}: wrong LR/WD contract",
            )
            require(
                argv.count("--experiment-identity") == 1,
                f"{context}/{arm}: expected one experiment identity flag",
            )
            identity_index = argv.index("--experiment-identity")
            require(
                argv[identity_index + 1] == "tiller_matrix_v1",
                f"{context}/{arm}: wrong experiment identity",
            )


def verify_manifest_launcher_contract(manifest_rows: list[dict[str, str]]) -> None:
    """Check that every baseline row enables its own exact LR/WD audit."""

    try:
        from . import run_activation_row
    except ImportError:  # Support direct execution from the repository root.
        from experiments.protocol import run_activation_row  # type: ignore[no-redef]

    output_root = Path("experiments/runs/reproducibility_check")
    fields = {
        "--fairness-contract": EXACT_LR_WD_CONTRACT,
        "--lr": None,
        "--min-lr": None,
        "--weight-decay": None,
    }
    for row in manifest_rows:
        context = f"manifest row {row['row_index']}"
        argv = run_activation_row.training_arguments(row, output_root)
        expected = dict(fields)
        expected.update(
            {
                "--lr": row["lr"],
                "--min-lr": row["min_lr"],
                "--weight-decay": row["weight_decay"],
            }
        )
        for flag, value in expected.items():
            require(argv.count(flag) == 1, f"{context}: expected one {flag}")
            index = argv.index(flag)
            require(index + 1 < len(argv), f"{context}: {flag} lacks a value")
            require(argv[index + 1] == value, f"{context}: {flag} differs from its row")


def verify_stage_launchers() -> None:
    activation = (PACKAGE / "run_activation_optimizer_sweep.sbatch").read_text()
    tiller = (PACKAGE / "run_quality_row.sbatch").read_text()
    require(
        "#SBATCH --array=0-29%2" in activation,
        "activation launcher default must be a 30-row stage with concurrency two",
    )
    require(
        'stage="${1:-muon}"' in activation
        and 'suite="${2:-18l_1024d_300m_tokens_9150_steps}"' in activation,
        "activation launcher lacks meaningful stage/suite selection",
    )
    require(
        "manifest-index" in activation and "sha256sum -c" in activation,
        "activation launcher does not bind rows to the frozen manifest source",
    )
    require(
        "#SBATCH --array=0-14%2" in tiller,
        "TILLER launcher default must be a 15-row suite with concurrency two",
    )
    require(
        'suite="${1:-18l_1024d_300m_tokens_9150_steps}"' in tiller
        and "matrix-index" in tiller,
        "TILLER launcher lacks meaningful suite selection",
    )
    require(
        "--dependency=" not in activation and "--dependency=" not in tiller,
        "stage dependencies belong in explicit submission control, not row launchers",
    )


def _truth(value: str, context: str, *, allow_blank: bool = True) -> bool | None:
    if allow_blank and value == "":
        return None
    return _as_bool(value, context)


def _optional_number(value: str, context: str) -> float | None:
    if value == "":
        return None
    return _as_float(value, context)


def _verify_loss_lead(
    row: dict[str, str],
    *,
    candidate_field: str,
    control_field: str,
    lead_field: str,
    context: str,
) -> None:
    candidate = _optional_number(row[candidate_field], f"{context}/{candidate_field}")
    control = _optional_number(row[control_field], f"{context}/{control_field}")
    lead = _optional_number(row[lead_field], f"{context}/{lead_field}")
    if (
        candidate is None
        or control is None
        or not math.isfinite(candidate)
        or not math.isfinite(control)
    ):
        require(lead is None, f"{context}: {lead_field} lacks two finite losses")
        return
    require(lead is not None, f"{context}: {lead_field} is missing")
    require(
        math.isclose(lead, control - candidate, rel_tol=1e-12, abs_tol=1e-12),
        f"{context}: {lead_field} is not control loss minus candidate loss",
    )


def _result_identity(row: dict[str, str], folder: str, context: str) -> None:
    optimizer, optimizer_display, suffix = OPTIMIZERS[folder]
    require(row["optimizer"] == optimizer, f"{context}: optimizer key does not match folder")
    require(row["optimizer_display_name"] == optimizer_display, f"{context}: optimizer display name changed")
    activation = row["activation"]
    require(activation in ACTIVATION_DISPLAY, f"{context}: unknown activation key")
    activation_display = ACTIVATION_DISPLAY[activation]
    require(row["activation_display_name"] == activation_display, f"{context}: activation display mapping changed")
    if folder == "tiller":
        expected_method = "tiller"
        expected_display = "TILLER"
    else:
        prefix = "silu" if activation == "silu" else "rlb"
        expected_method = f"{prefix}_{suffix}"
        expected_display = f"{activation_display} + {optimizer_display}"
    require(row["method"] == expected_method, f"{context}: method key changed")
    require(row["method_display_name"] == expected_display, f"{context}: method display mapping changed")


def _unique(rows: Iterable[dict[str, str]], fields: tuple[str, ...], context: str) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(set(keys)), f"{context}: duplicate result cell")


def _nonfinite_text(value: str) -> bool:
    if not value:
        return False
    try:
        return not math.isfinite(float(value))
    except ValueError:
        return False


def _finite_run_values(
    rows: list[dict[str, str]], field: str, context: str
) -> list[float]:
    values: list[float] = []
    for index, row in enumerate(rows):
        text = row[field]
        if text == "":
            continue
        value = _as_float(text, f"{context}/{field}/run{index}")
        if math.isfinite(value):
            values.append(value)
    return values


def _verify_summary_number(
    row: dict[str, str], field: str, expected: float | None, context: str
) -> None:
    observed_text = row[field]
    if expected is None:
        require(observed_text == "", f"{context}: {field} must be blank")
        return
    require(observed_text != "", f"{context}: {field} is missing")
    observed = _as_float(observed_text, f"{context}/{field}")
    require(math.isfinite(observed), f"{context}: {field} is non-finite")
    require(
        math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{context}: {field} is not recomputed from runs "
        f"(found {observed!r}, expected {expected!r})",
    )


def _recomputed_status(
    *, total: int, reached: int, stopped: int, non_finite: int, endpoints: int,
    pending: int,
) -> str:
    if non_finite:
        return "non_finite"
    if stopped:
        return "stopped_early"
    if reached < total:
        return "pending" if pending == total else "incomplete"
    if endpoints < total:
        return "endpoint_loss_unavailable"
    return "complete"


def _result_suite(row: dict[str, str], context: str) -> str:
    coordinates = (
        row["model_scale"], row["train_tokens"], row["steps_required"]
    )
    suites = {
        (SMALL_MODEL, "100000000", "3050"): SUITE_100M,
        (SMALL_MODEL, "300000000", "9150"): SUITE_300M_SMALL,
        (LARGE_MODEL, "300000000", "9150"): SUITE_300M_LARGE,
    }
    require(coordinates in suites, f"{context}: result is outside the published suites")
    return suites[coordinates]


def verify_results() -> dict[str, dict[str, list[dict[str, str]]]]:
    require(RESULTS.is_dir(), "experiments/results is missing")
    manifest_rows = read_csv(MANIFEST, MANIFEST_FIELDS)
    manifest_by_id = {row["row_id"]: row for row in manifest_rows}
    manifest_by_index = {row["row_index"]: row for row in manifest_rows}
    matrix_payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    tiller_sources = {
        (
            row["phase"], row["model"], row["dataset"], str(row["seed"])
        ): (
            row["source_manifest_row_id"], str(row["source_manifest_row_index"])
        )
        for row in matrix_payload["rows"]
    }
    tiller_controls = {
        (
            row["phase"], row["model"], row["dataset"], str(row["seed"])
        ): row["matched_control"]
        for row in matrix_payload["rows"]
    }
    directories = {path.name for path in RESULTS.iterdir() if path.is_dir()}
    require(directories == set(OPTIMIZERS), "result optimizer directory inventory changed")
    require((RESULTS / "README.md").is_file(), "experiments/results/README.md is missing")

    loaded: dict[str, dict[str, list[dict[str, str]]]] = {}
    for folder in sorted(OPTIMIZERS):
        directory = RESULTS / folder
        expected_files = {"README.md", "runs.csv", "summary.csv", "checkpoints.csv"}
        actual_files = {path.name for path in directory.iterdir() if path.is_file()}
        require(
            actual_files == expected_files,
            f"results/{folder}: file inventory changed",
        )
        require((directory / "README.md").is_file(), f"results/{folder}/README.md is missing")
        runs = read_csv(directory / "runs.csv", RUN_FIELDS)
        summaries = read_csv(directory / "summary.csv", SUMMARY_FIELDS)
        require(runs, f"results/{folder}/runs.csv is empty")
        require(summaries, f"results/{folder}/summary.csv is empty")
        _unique(
            runs,
            ("model_scale", "dataset", "seed", "train_tokens", "steps_required", "activation", "optimizer"),
            f"results/{folder}/runs.csv",
        )
        _unique(
            summaries,
            ("model_scale", "dataset", "train_tokens", "steps_required", "activation", "optimizer"),
            f"results/{folder}/summary.csv",
        )

        run_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for line, row in enumerate(runs, start=2):
            context = f"results/{folder}/runs.csv:{line}"
            _result_identity(row, folder, context)
            require(row["model_scale"] in MODEL_FIELDS, f"{context}: unknown model shape")
            require(row["dataset"] in {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"}, f"{context}: unknown dataset")
            expected_suite = _result_suite(row, context)
            require(row["source_phase"] in {"", expected_suite}, f"{context}: source suite disagrees with result coordinates")
            run_index = _as_int(row["run_index"], f"{context}/run_index")
            seed = _as_int(row["seed"], f"{context}/seed")
            tokens = _as_int(row["train_tokens"], f"{context}/train_tokens")
            required_steps = _as_int(row["steps_required"], f"{context}/steps_required")
            completed_steps = _as_int(row["steps_completed"], f"{context}/steps_completed")
            require(run_index >= 0 and seed >= 0 and tokens > 0, f"{context}: invalid run coordinates")
            require(0 <= completed_steps <= required_steps, f"{context}: invalid step count")
            require(row["status"] in RUN_STATUSES, f"{context}: unknown status")
            if row["status"] == "complete":
                require(completed_steps == required_steps, f"{context}: complete run has not reached endpoint")
            if row["status"] == "pending":
                require(completed_steps == 0, f"{context}: pending run has completed steps")
                require(row["final_validation_loss"] == "", f"{context}: pending run has endpoint loss")
            if row["status"] == "incomplete":
                require(completed_steps < required_steps, f"{context}: incomplete run reached endpoint")
            stopped = _truth(row["stopped_early"], f"{context}/stopped_early")
            _truth(row["lr_wd_fairness_passed"], f"{context}/lr_wd_fairness_passed")
            if row["status"] == "stopped_early":
                require(stopped is True, f"{context}: stopped-early status lacks marker")
            if _nonfinite_text(row["final_validation_loss"]):
                require(row["status"] == "non_finite", f"{context}: non-finite endpoint is not labelled")
            for field in (
                "step1000_validation_loss", "final_validation_loss",
                "final_validation_perplexity",
                "matched_control_step1000_validation_loss",
                "matched_control_validation_loss",
                "lead_at_step1000_vs_matched_control",
                "lead_vs_matched_control", "total_seconds",
                "matched_control_total_seconds",
                "total_time_ratio_vs_matched_control",
                "training_loop_total_seconds", "mean_seconds_per_step",
                "tokens_per_second",
            ):
                _optional_number(row[field], f"{context}/{field}")
            _verify_loss_lead(
                row,
                candidate_field="step1000_validation_loss",
                control_field="matched_control_step1000_validation_loss",
                lead_field="lead_at_step1000_vs_matched_control",
                context=context,
            )
            _verify_loss_lead(
                row,
                candidate_field="final_validation_loss",
                control_field="matched_control_validation_loss",
                lead_field="lead_vs_matched_control",
                context=context,
            )
            require(
                row["time_scope"] in {"", "training_loop", "end_to_end_process"},
                f"{context}: unknown time scope",
            )
            if row["total_time_ratio_vs_matched_control"]:
                total = _as_float(row["total_seconds"], f"{context}/total_seconds")
                control_total = _as_float(
                    row["matched_control_total_seconds"],
                    f"{context}/matched_control_total_seconds",
                )
                ratio = _as_float(
                    row["total_time_ratio_vs_matched_control"],
                    f"{context}/total_time_ratio_vs_matched_control",
                )
                require(
                    row["time_scope"] == "end_to_end_process"
                    and row["matched_control_total_seconds"] != "",
                    f"{context}: exact time ratio lacks paired process timing",
                )
                require(
                    control_total > 0.0
                    and math.isclose(
                        ratio, total / control_total,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    f"{context}: exact time ratio is not candidate/control",
                )
            for hash_field in ("realized_lr_trace_sha256", "source_jsonl_sha256"):
                require(
                    row[hash_field] == "" or HEX64.fullmatch(row[hash_field]) is not None,
                    f"{context}: malformed {hash_field}",
                )
            require(not Path(row["source_jsonl"]).is_absolute(), f"{context}: source_jsonl must be relative")
            source_id = row["source_row_id"]
            source_index = row["source_row_index"]
            require(source_id in manifest_by_id, f"{context}: unknown source_row_id")
            require(source_index in manifest_by_index, f"{context}: unknown source_row_index")
            source = manifest_by_id[source_id]
            require(
                source is manifest_by_index[source_index],
                f"{context}: source row id/index disagree",
            )
            for result_field, source_field in (
                ("model_scale", "model"), ("dataset", "dataset"),
                ("seed", "seed"), ("train_tokens", "train_tokens"),
                ("steps_required", "steps"),
            ):
                require(
                    row[result_field] == source[source_field],
                    f"{context}: {result_field} disagrees with source manifest",
                )
            if folder == "tiller":
                tiller_key = (
                    expected_suite, row["model_scale"], row["dataset"], row["seed"]
                )
                require(
                    (source_id, source_index)
                    == tiller_sources[tiller_key],
                    f"{context}: TILLER source is not its matrix control row",
                )
                matched = tiller_controls[tiller_key]
                expected_endpoint = matched["endpoint_validation_loss"]
                require(
                    row["matched_control_validation_loss"]
                    == ("" if expected_endpoint is None else str(expected_endpoint)),
                    f"{context}: TILLER endpoint control is not the exact same-seed row",
                )
                expected_step1000 = matched["step1000_validation_loss"]
                require(
                    row["matched_control_step1000_validation_loss"]
                    == ("" if expected_step1000 is None else str(expected_step1000)),
                    f"{context}: TILLER step-1000 control differs from its exact row",
                )
                expected_control_seconds = matched["total_seconds"]
                require(
                    row["matched_control_total_seconds"]
                    == ("" if expected_control_seconds is None else str(expected_control_seconds)),
                    f"{context}: TILLER control timing differs from its exact row",
                )
            else:
                for field in ("activation", "optimizer", "method"):
                    require(
                        row[field] == source[field],
                        f"{context}: {field} disagrees with source manifest",
                    )
            key = (
                row["model_scale"], row["dataset"], row["train_tokens"],
                row["steps_required"], row["activation"], row["optimizer"],
            )
            run_groups[key].append(row)

        summary_groups: dict[tuple[str, ...], dict[str, str]] = {}
        for line, row in enumerate(summaries, start=2):
            context = f"results/{folder}/summary.csv:{line}"
            _result_identity(row, folder, context)
            require(row["model_scale"] in MODEL_FIELDS, f"{context}: unknown model shape")
            _result_suite(row, context)
            require(row["status"] in SUMMARY_STATUSES, f"{context}: unknown status")
            key = (
                row["model_scale"], row["dataset"], row["train_tokens"],
                row["steps_required"], row["activation"], row["optimizer"],
            )
            counts = {
                field: _as_int(row[field], f"{context}/{field}")
                for field in (
                    "runs_total", "runs_reached_required_step", "runs_stopped_early",
                    "runs_non_finite", "endpoint_seed_count",
                    "matched_control_seed_count", "exact_time_ratio_seed_count",
                )
            }
            require(counts["runs_total"] > 0, f"{context}: empty aggregate cell")
            require(
                all(0 <= value <= counts["runs_total"] for field, value in counts.items() if field != "runs_total"),
                f"{context}: aggregate counts are inconsistent",
            )
            for field in (
                "validation_loss_mean", "validation_loss_sample_std",
                "matched_control_validation_loss_mean", "lead_vs_matched_control_mean",
                "lead_vs_matched_control_sample_std",
                "exact_total_time_ratio_mean", "exact_total_time_ratio_sample_std",
                "total_seconds_mean", "total_seconds_sample_std",
                "mean_seconds_per_step", "tokens_per_second_mean",
            ):
                value = _optional_number(row[field], f"{context}/{field}")
                require(value is None or math.isfinite(value), f"{context}: non-finite summary value")
            if row["status"] == "complete":
                require(counts["runs_reached_required_step"] == counts["runs_total"], f"{context}: complete aggregate has unfinished runs")
            if row["status"] == "incomplete":
                require(counts["runs_reached_required_step"] < counts["runs_total"], f"{context}: incomplete aggregate is complete")
            if row["status"] == "non_finite":
                require(counts["runs_non_finite"] > 0, f"{context}: non-finite aggregate has no non-finite runs")
            if row["status"] == "stopped_early":
                require(counts["runs_stopped_early"] > 0, f"{context}: stopped aggregate has no stopped runs")
            if row["status"] == "endpoint_loss_unavailable":
                require(counts["endpoint_seed_count"] == 0, f"{context}: endpoint is actually available")
            summary_groups[key] = row

        require(set(run_groups) == set(summary_groups), f"results/{folder}: run/summary cells differ")
        for key, group in run_groups.items():
            summary = summary_groups[key]
            context = f"results/{folder}/summary/{key}"
            required_steps = _as_int(summary["steps_required"], context)
            endpoint_values = [
                value
                for run in group
                if run["status"] == "complete"
                and (value := _optional_number(
                    run["final_validation_loss"],
                    f"{context}/final_validation_loss",
                )) is not None
                and math.isfinite(value)
            ]
            matched_controls = _finite_run_values(
                group, "matched_control_validation_loss", context
            )
            exact_time_ratios = _finite_run_values(
                group, "total_time_ratio_vs_matched_control", context
            )
            expected_counts = {
                "runs_total": len(group),
                "runs_reached_required_step": sum(
                    _as_int(run["steps_completed"], "steps_completed") >= required_steps
                    for run in group
                ),
                "runs_stopped_early": sum(
                    run["stopped_early"].strip().lower() == "true" for run in group
                ),
                "runs_non_finite": sum(
                    _nonfinite_text(run["final_validation_loss"]) for run in group
                ),
                "endpoint_seed_count": len(endpoint_values),
                "matched_control_seed_count": len(matched_controls),
                "exact_time_ratio_seed_count": len(exact_time_ratios),
            }
            for field, expected in expected_counts.items():
                require(
                    _as_int(summary[field], f"{context}/{field}") == expected,
                    f"results/{folder}: {field} disagrees with run records in {key}",
                )

            expected_status = _recomputed_status(
                total=expected_counts["runs_total"],
                reached=expected_counts["runs_reached_required_step"],
                stopped=expected_counts["runs_stopped_early"],
                non_finite=expected_counts["runs_non_finite"],
                endpoints=expected_counts["endpoint_seed_count"],
                pending=sum(run["status"] == "pending" for run in group),
            )
            require(
                summary["status"] == expected_status,
                f"{context}: status is {summary['status']!r}, expected {expected_status!r}",
            )

            total_seconds = _finite_run_values(group, "total_seconds", context)
            mean_step_seconds = _finite_run_values(
                group, "mean_seconds_per_step", context
            )
            throughputs = _finite_run_values(group, "tokens_per_second", context)
            leads = _finite_run_values(
                group, "lead_vs_matched_control", context
            )
            expected_numbers = {
                "validation_loss_mean": (
                    statistics.fmean(endpoint_values) if endpoint_values else None
                ),
                "validation_loss_sample_std": (
                    statistics.stdev(endpoint_values)
                    if len(endpoint_values) >= 2 else None
                ),
                "matched_control_validation_loss_mean": (
                    statistics.fmean(matched_controls) if matched_controls else None
                ),
                "lead_vs_matched_control_mean": (
                    statistics.fmean(leads) if leads else None
                ),
                "lead_vs_matched_control_sample_std": (
                    statistics.stdev(leads) if len(leads) >= 2 else None
                ),
                "exact_total_time_ratio_mean": (
                    statistics.fmean(exact_time_ratios)
                    if exact_time_ratios else None
                ),
                "exact_total_time_ratio_sample_std": (
                    statistics.stdev(exact_time_ratios)
                    if len(exact_time_ratios) >= 2 else None
                ),
                "total_seconds_mean": (
                    statistics.fmean(total_seconds) if total_seconds else None
                ),
                "total_seconds_sample_std": (
                    statistics.stdev(total_seconds)
                    if len(total_seconds) >= 2 else None
                ),
                "mean_seconds_per_step": (
                    statistics.fmean(mean_step_seconds) if mean_step_seconds else None
                ),
                "tokens_per_second_mean": (
                    statistics.fmean(throughputs) if throughputs else None
                ),
            }
            for field, expected in expected_numbers.items():
                _verify_summary_number(summary, field, expected, context)

        checkpoints = read_csv(directory / "checkpoints.csv", CHECKPOINT_FIELDS)
        _unique(checkpoints, ("run_index", "step"), f"results/{folder}/checkpoints.csv")
        runs_by_index = {row["run_index"]: row for row in runs}
        require(
            len(runs_by_index) == len(runs),
            f"results/{folder}: run_index values are not unique",
        )
        checkpoints_by_coordinate: dict[tuple[str, str], dict[str, str]] = {}
        run_indices_with_checkpoints: set[str] = set()
        for line, row in enumerate(checkpoints, start=2):
            context = f"results/{folder}/checkpoints.csv:{line}"
            _result_identity(row, folder, context)
            require(row["run_index"] in runs_by_index, f"{context}: unknown run_index")
            run = runs_by_index[row["run_index"]]
            for field in (
                "model_scale", "dataset", "seed", "activation", "optimizer", "method"
            ):
                require(row[field] == run[field], f"{context}: {field} disagrees with run")
            step = _as_int(row["step"], f"{context}/step")
            require(
                0 <= step <= _as_int(run["steps_completed"], f"{context}/steps_completed"),
                f"{context}: checkpoint exceeds completed trajectory",
            )
            for field in (
                "validation_loss", "validation_perplexity",
                "active_seconds_at_validation",
            ):
                value = _as_float(row[field], f"{context}/{field}")
                require(math.isfinite(value), f"{context}: non-finite checkpoint value")
            require(
                _as_float(row["active_seconds_at_validation"], context) >= 0,
                f"{context}: negative active time",
            )
            checkpoints_by_coordinate[(row["run_index"], row["step"])] = row
            run_indices_with_checkpoints.add(row["run_index"])

        for run in runs:
            run_index = run["run_index"]
            step1000 = run["step1000_validation_loss"]
            if step1000:
                checkpoint = checkpoints_by_coordinate.get((run_index, "1000"))
                require(checkpoint is not None, f"results/{folder}: step-1000 loss lacks checkpoint")
                require(
                    math.isclose(
                        _as_float(step1000, "step1000_validation_loss"),
                        _as_float(checkpoint["validation_loss"], "checkpoint validation_loss"),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    f"results/{folder}: step-1000 loss disagrees with checkpoint",
                )
            if run["status"] == "complete" and run["final_validation_loss"]:
                endpoint_step = run["steps_required"]
                checkpoint = checkpoints_by_coordinate.get((run_index, endpoint_step))
                if run_index in run_indices_with_checkpoints:
                    require(checkpoint is not None, f"results/{folder}: endpoint loss lacks checkpoint")
                    require(
                        math.isclose(
                            _as_float(run["final_validation_loss"], "final_validation_loss"),
                            _as_float(checkpoint["validation_loss"], "checkpoint validation_loss"),
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        f"results/{folder}: endpoint loss disagrees with checkpoint",
                    )

        loaded[folder] = {
            "runs": runs,
            "summary": summaries,
            "checkpoints": checkpoints,
        }

    tiller_runs = loaded["tiller"]["runs"]
    for folder, tables in loaded.items():
        if folder == "tiller":
            continue
        require(len(tables["runs"]) == 90, f"results/{folder}: expected 90 run rows")
        require(len(tables["summary"]) == 30, f"results/{folder}: expected 30 summary rows")
        for model, tokens, steps in (
            (SMALL_MODEL, "100000000", "3050"),
            (SMALL_MODEL, "300000000", "9150"),
            (LARGE_MODEL, "300000000", "9150"),
        ):
            subset = [
                row for row in tables["runs"]
                if row["model_scale"] == model
                and row["train_tokens"] == tokens
                and row["steps_required"] == steps
            ]
            require(len(subset) == 30, f"results/{folder}: incomplete {tokens}-token inventory")
            require(
                {row["dataset"] for row in subset}
                == {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
                f"results/{folder}: {tokens}-token dataset inventory changed",
            )
            require(
                {row["activation"] for row in subset} == {"silu", GRAIN_ID},
                f"results/{folder}: {tokens}-token activation arms changed",
            )
    for model, tokens, steps in (
        (SMALL_MODEL, "100000000", "3050"),
        (SMALL_MODEL, "300000000", "9150"),
        (LARGE_MODEL, "300000000", "9150"),
    ):
        subset = [
            row for row in tiller_runs
            if row["model_scale"] == model
            and row["train_tokens"] == tokens
            and row["steps_required"] == steps
        ]
        require(len(subset) == 15, f"TILLER {tokens}-token result inventory changed")
        require(
            {row["dataset"] for row in subset}
            == {"dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"},
            f"TILLER {tokens}-token dataset inventory changed",
        )
        require(
            {row["status"] for row in subset} <= RUN_STATUSES,
            f"TILLER {model}/{tokens}-token status inventory changed",
        )

    primary_rows = [
        row
        for tables in loaded.values()
        for row in tables["runs"]
        if row["model_scale"] == LARGE_MODEL
        and row["train_tokens"] == "300000000"
        and row["steps_required"] == "9150"
    ]
    require(len(primary_rows) == 225, "primary 18-layer result inventory changed")
    primary_status = Counter(row["status"] for row in primary_rows)
    require(
        primary_status == Counter({"complete": 7, "incomplete": 1, "pending": 217}),
        f"primary 18-layer result status changed: {dict(primary_status)}",
    )

    return loaded


def verify_public_layout() -> tuple[str, ...]:
    inventory = build_source_freeze.public_inventory(ROOT)
    public = set(inventory)
    public_directories = {
        str(parent)
        for relative in inventory
        for parent in Path(relative).parents
        if str(parent) not in {".", "paper"}
    }
    for directory in sorted(public_directories):
        require(
            f"{directory}/README.md" in public,
            f"public directory lacks README.md: {directory}/",
        )
    for directory in ("activation", "optimizer_design", "training", "experiments"):
        require((ROOT / directory).is_dir(), f"missing root directory {directory}/")
        require(f"{directory}/README.md" in public, f"{directory}/README.md is not public")
    require((ROOT / "paper").is_dir(), "missing root directory paper/")
    paper_files = {path for path in public if path.startswith("paper/")}
    require(paper_files == {"paper/.gitkeep"}, "paper/ must publicly contain only .gitkeep")

    optimizer_files = {path for path in public if path.startswith("optimizer_design/")}
    expected_optimizer_files = {
        "optimizer_design/README.md",
        "optimizer_design/__init__.py",
        "optimizer_design/tiller.py",
        "optimizer_design/_tiller/README.md",
        "optimizer_design/_tiller/__init__.py",
        "optimizer_design/_tiller/core.py",
    }
    require(
        optimizer_files == expected_optimizer_files,
        "public optimizer tree must contain only the TILLER API and implementation: "
        f"{sorted(optimizer_files ^ expected_optimizer_files)}",
    )
    return inventory


def verify_no_absolute_workspace_paths(inventory: tuple[str, ...] | None = None) -> None:
    if inventory is None:
        inventory = build_source_freeze.public_inventory(ROOT)
    exact_root = str(ROOT)
    for relative in inventory:
        path = ROOT / relative
        payload = path.read_bytes()
        if b"\x00" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        require(exact_root not in text, f"absolute repository path embedded in {relative}")
        match = WORKSPACE_PATH.search(text)
        require(match is None, f"absolute workspace path embedded in {relative}: {match.group(0) if match else ''}")


def verify_public_suite_names(inventory: tuple[str, ...] | None = None) -> None:
    """Keep reader-facing artifacts on descriptive model and suite names."""

    if inventory is None:
        inventory = build_source_freeze.public_inventory(ROOT)
    retired = re.compile(r"\b(?:M[01]|E[0-3])(?:_|\b)")
    reader_files = {
        relative
        for relative in inventory
        if relative.endswith("README.md")
        or relative.startswith("experiments/results/")
        or relative in {
            "experiments/protocol/activation_optimizer_manifest.csv",
            "experiments/protocol/matrix.json",
        }
    }
    for relative in sorted(reader_files):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        match = retired.search(text)
        require(match is None, f"retired suite alias remains in {relative}")


def parse_freeze(payload: str) -> dict[str, str]:
    require(bool(payload), "source freeze is empty")
    entries: dict[str, str] = {}
    previous = ""
    for number, line in enumerate(payload.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        require(match is not None, f"malformed source-freeze line {number}")
        digest, relative = match.groups()
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe source-freeze path on line {number}")
        require(relative not in entries, f"duplicate source-freeze path: {relative}")
        require(previous < relative if previous else True, "source-freeze paths are not sorted")
        entries[relative] = digest
        previous = relative
    return entries


def verify_source_freeze() -> dict[str, str]:
    require(FREEZE.is_file(), "SOURCE_FREEZE.sha256 is missing")
    entries = parse_freeze(FREEZE.read_text(encoding="utf-8"))
    expected = set(build_source_freeze.frozen_paths(ROOT))
    require(set(entries) == expected, "source-freeze file inventory is stale")
    for relative, digest in entries.items():
        require(_sha256(ROOT / relative) == digest, f"source-freeze hash mismatch: {relative}")
        require(not relative.startswith("experiments/results/"), "generated results entered source freeze")
        require(Path(relative).suffix.lower() not in build_source_freeze.EXCLUDED_SUFFIXES, f"binary entered source freeze: {relative}")
    return entries


def verify_repository(*, quick: bool = False) -> list[tuple[str, int]]:
    manifest = verify_manifest()
    matrix = verify_matrix(manifest)
    fingerprints = verify_token_fingerprints()
    verify_manifest_launcher_contract(manifest)
    verify_launcher_pairing(matrix)
    verify_stage_launchers()
    inventory = verify_public_layout()
    verify_no_absolute_workspace_paths(inventory)
    verify_public_suite_names(inventory)
    checks = [
        ("activation/optimizer manifest", len(manifest)),
        ("TILLER transfer matrix", len(matrix["rows"])),
        ("token fingerprint cells", len(fingerprints["cells"])),
        ("public files", len(inventory)),
    ]
    if quick:
        return checks
    results = verify_results()
    freeze = verify_source_freeze()
    checks.extend(
        (
            ("optimizer result directories", len(results)),
            ("frozen source files", len(freeze)),
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="check runnable configuration and public layout without result or hash scans",
    )
    args = parser.parse_args()
    try:
        checks = verify_repository(quick=args.quick)
    except (VerificationError, build_source_freeze.FreezeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    for label, count in checks:
        print(f"PASS {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
