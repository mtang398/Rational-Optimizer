#!/usr/bin/env python3
"""Rebuild compact experiment tables from repository-produced run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[1]
DEFAULT_MANIFEST = PACKAGE / "activation_optimizer_manifest.csv"
DEFAULT_MATRIX = PACKAGE / "matrix.json"
FACTORIAL_PHASE = "18l_1024d_300m_tokens_9150_steps"

GRAIN_ID = "rlb_fused_global_rational"
PREFLIGHT_SUITE = "12l_768d_preflight_2621440_tokens_80_steps"
ACTIVATION_NAMES = {"silu": "SwiGLU", GRAIN_ID: "GRAIN"}
OPTIMIZER_NAMES = {
    "adamw": "AdamW",
    "muon": "Muon",
    "lion": "Lion",
    "soap_adamw": "SOAP",
    "ademamix": "ADeMaMix",
    "adafactor_came": "CAME",
    "schedule_free_adamw": "Schedule-Free AdamW",
    "tiller_v1": "TILLER",
}
OUTPUT_DIRECTORIES = {
    "adamw": "adamw",
    "muon": "muon",
    "lion": "lion",
    "soap_adamw": "soap",
    "ademamix": "ademamix",
    "adafactor_came": "came",
    "schedule_free_adamw": "schedule_free_adamw",
    "tiller_v1": "tiller",
}

RUN_FIELDS = [
    "run_index",
    "model_scale",
    "dataset",
    "seed",
    "train_tokens",
    "steps_required",
    "steps_completed",
    "status",
    "activation",
    "activation_display_name",
    "optimizer",
    "optimizer_display_name",
    "method",
    "method_display_name",
    "step1000_validation_loss",
    "final_validation_loss",
    "final_validation_perplexity",
    "matched_control_step1000_validation_loss",
    "matched_control_validation_loss",
    "lead_at_step1000_vs_matched_control",
    "lead_vs_matched_control",
    "time_scope",
    "total_seconds",
    "matched_control_total_seconds",
    "total_time_ratio_vs_matched_control",
    "training_loop_total_seconds",
    "mean_seconds_per_step",
    "tokens_per_second",
    "stopped_early",
    "lr_wd_fairness_passed",
    "realized_lr_trace_sha256",
    "slurm_job_id",
    "slurm_restart_count",
    "slurm_node",
    "timing_attempt_id",
    "source_phase",
    "source_row_index",
    "source_row_id",
    "source_jsonl",
    "source_jsonl_sha256",
]

SUMMARY_FIELDS = [
    "model_scale",
    "dataset",
    "train_tokens",
    "steps_required",
    "activation",
    "activation_display_name",
    "optimizer",
    "optimizer_display_name",
    "method",
    "method_display_name",
    "runs_total",
    "runs_reached_required_step",
    "runs_stopped_early",
    "runs_non_finite",
    "endpoint_seed_count",
    "validation_loss_mean",
    "validation_loss_sample_std",
    "matched_control_seed_count",
    "matched_control_validation_loss_mean",
    "lead_vs_matched_control_mean",
    "lead_vs_matched_control_sample_std",
    "exact_time_ratio_seed_count",
    "exact_total_time_ratio_mean",
    "exact_total_time_ratio_sample_std",
    "total_seconds_mean",
    "total_seconds_sample_std",
    "mean_seconds_per_step",
    "tokens_per_second_mean",
    "status",
]

CHECKPOINT_FIELDS = [
    "run_index",
    "model_scale",
    "dataset",
    "seed",
    "activation",
    "activation_display_name",
    "optimizer",
    "optimizer_display_name",
    "method",
    "method_display_name",
    "step",
    "validation_loss",
    "validation_perplexity",
    "active_seconds_at_validation",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY.resolve()))
    except ValueError:
        return str(path.resolve())


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_records(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while tolerating only a partial final line from an active run."""

    lines = path.read_text().splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise RuntimeError(f"invalid JSONL record {index + 1} in {path}")
        if not isinstance(record, dict):
            raise RuntimeError(f"non-object JSONL record {index + 1} in {path}")
        records.append(record)
    return records


def unique_event(records: Iterable[dict[str, Any]], event: str) -> dict[str, Any] | None:
    matches = [record for record in records if record.get("event") == event]
    if len(matches) > 1:
        raise RuntimeError(f"found {len(matches)} {event!r} records")
    return matches[0] if matches else None


def evaluations(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if record.get("event") != "eval":
            continue
        step = int(record["step"])
        if step in result:
            raise RuntimeError(f"duplicate evaluation at step {step}")
        result[step] = record
    return result


def process_time(path: Path) -> tuple[float | None, str]:
    sidecar = path.with_suffix(".wall_clock.json")
    if not sidecar.is_file():
        return None, "training_loop"
    payload = json.loads(sidecar.read_text())
    if (
        payload.get("schema") != "rationalopt_process_wall_clock_v1"
        or payload.get("return_code") != 0
        or finite(payload.get("elapsed_seconds")) is None
    ):
        raise RuntimeError(f"invalid process wall-clock sidecar: {sidecar}")
    return float(payload["elapsed_seconds"]), "end_to_end_process"


def factorial_process_time(
    path: Path,
    *,
    staged_row: dict[str, Any],
    artifact: Path,
    source_freeze: Path,
    stage: str = "01_muon",
) -> float | None:
    """Read the process wall clock emitted by the staged factorial launcher."""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    expected_artifact_sha256 = sha256(artifact)
    expected_source_freeze_sha256 = sha256(source_freeze)
    if (
        payload.get("schema") != "tiller_endpoint_process_wall_clock_v1"
        or payload.get("process_exit_status") != 0
        or isinstance(payload.get("process_exit_status"), bool)
        or finite(payload.get("elapsed_seconds")) is None
        or isinstance(payload.get("elapsed_seconds"), bool)
        or float(payload["elapsed_seconds"]) <= 0
        or int(payload.get("matrix_index", -1)) != int(staged_row["matrix_index"])
        or payload.get("row_id") != staged_row["row_id"]
        or payload.get("stage") != stage
        or Path(str(payload.get("artifact_path", ""))).resolve() != artifact.resolve()
        or payload.get("artifact_exists") is not True
        or payload.get("artifact_bytes") != artifact.stat().st_size
        or payload.get("artifact_sha256") != expected_artifact_sha256
        or payload.get("source_freeze_sha256") != expected_source_freeze_sha256
    ):
        raise RuntimeError(f"invalid factorial process wall clock: {path}")
    return float(payload["elapsed_seconds"])


def raw_run(
    *,
    index: int,
    model: str,
    dataset: str,
    seed: int,
    train_tokens: int,
    steps: int,
    activation: str,
    optimizer: str,
    method: str,
    phase: str,
    source_row_index: int,
    source_row_id: str,
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base = {
        field: "" for field in RUN_FIELDS
    }
    base.update(
        {
            "run_index": index,
            "model_scale": model,
            "dataset": dataset,
            "seed": seed,
            "train_tokens": train_tokens,
            "steps_required": steps,
            "steps_completed": 0,
            "status": "pending",
            "activation": activation,
            "activation_display_name": ACTIVATION_NAMES[activation],
            "optimizer": optimizer,
            "optimizer_display_name": OPTIMIZER_NAMES[optimizer],
            "method": method,
            "method_display_name": (
                "TILLER"
                if optimizer == "tiller_v1"
                else f"{ACTIVATION_NAMES[activation]} + {OPTIMIZER_NAMES[optimizer]}"
            ),
            "source_phase": phase,
            "source_row_index": source_row_index,
            "source_row_id": source_row_id,
        }
    )
    if not path.is_file():
        return base, []

    records = safe_records(path)
    config = unique_event(records, "config")
    summary = unique_event(records, "summary")
    evals = evaluations(records)
    endpoint = evals.get(steps)
    observed_steps = [
        int(record["step"])
        for record in records
        if isinstance(record.get("step"), (int, float))
    ]
    completed = (
        int(summary.get("completed_steps", 0))
        if summary
        else max(observed_steps, default=0)
    )
    stopped = bool(summary.get("stopped_early")) if summary else any(
        record.get("event") == "stopped_early" for record in records
    )
    endpoint_loss = finite(endpoint.get("val_loss")) if endpoint else None
    endpoint_ppl = finite(endpoint.get("val_ppl")) if endpoint else None
    loop_seconds = finite(summary.get("total_seconds")) if summary else None
    outer_seconds, scope = process_time(path)
    total_seconds = outer_seconds if outer_seconds is not None else loop_seconds
    status = "incomplete"
    if stopped:
        status = "stopped_early"
    elif summary and completed == steps and endpoint and endpoint_loss is None:
        status = "non_finite"
    elif summary and completed == steps and endpoint_loss is not None:
        status = "complete"
    fairness = (config or {}).get("optimizer_lr_wd_fairness", {})
    base.update(
        {
            "steps_completed": completed,
            "status": status,
            "step1000_validation_loss": (
                finite(evals[1000].get("val_loss")) if 1000 in evals else ""
            ),
            "final_validation_loss": endpoint_loss if endpoint_loss is not None else "",
            "final_validation_perplexity": endpoint_ppl if endpoint_ppl is not None else "",
            "time_scope": scope if total_seconds is not None else "",
            "total_seconds": total_seconds if total_seconds is not None else "",
            "training_loop_total_seconds": loop_seconds if loop_seconds is not None else "",
            "mean_seconds_per_step": (
                finite(summary.get("mean_seconds_per_step")) if summary else ""
            ),
            "tokens_per_second": (
                finite(summary.get("tokens_per_second")) if summary else ""
            ),
            "stopped_early": stopped,
            "lr_wd_fairness_passed": fairness.get("passed", ""),
            "realized_lr_trace_sha256": (
                summary.get("realized_lr_trace_sha256", "") if summary else ""
            ),
            "slurm_job_id": (summary or config or {}).get("slurm_job_id", ""),
            "slurm_restart_count": (summary or config or {}).get("slurm_restart_count", ""),
            "slurm_node": (summary or config or {}).get("slurm_node", ""),
            "timing_attempt_id": (summary or config or {}).get("timing_attempt_id", ""),
            "source_jsonl": display_path(path),
            "source_jsonl_sha256": sha256(path),
        }
    )
    checkpoints = [
        {
            "run_index": index,
            "model_scale": model,
            "dataset": dataset,
            "seed": seed,
            "activation": activation,
            "activation_display_name": ACTIVATION_NAMES[activation],
            "optimizer": optimizer,
            "optimizer_display_name": OPTIMIZER_NAMES[optimizer],
            "method": method,
            "method_display_name": base["method_display_name"],
            "step": step,
            "validation_loss": finite(record.get("val_loss")) or "",
            "validation_perplexity": finite(record.get("val_ppl")) or "",
            "active_seconds_at_validation": (
                finite(record.get("active_seconds_at_val_loss")) or ""
            ),
        }
        for step, record in sorted(evals.items())
    ]
    return base, checkpoints


def pair(candidate: dict[str, Any], control: dict[str, Any]) -> None:
    step1000_candidate = finite(candidate["step1000_validation_loss"])
    step1000_control = finite(control["step1000_validation_loss"])
    if step1000_control is not None:
        candidate["matched_control_step1000_validation_loss"] = step1000_control
    if step1000_candidate is not None and step1000_control is not None:
        candidate["lead_at_step1000_vs_matched_control"] = (
            step1000_control - step1000_candidate
        )
    if control["status"] != "complete":
        return
    candidate["matched_control_validation_loss"] = control["final_validation_loss"]
    control_time = finite(control["total_seconds"])
    if control_time is not None:
        candidate["matched_control_total_seconds"] = control_time
    if candidate["status"] != "complete":
        return
    candidate["lead_vs_matched_control"] = (
        float(control["final_validation_loss"]) - float(candidate["final_validation_loss"])
    )
    candidate_time = finite(candidate["total_seconds"])
    if (
        candidate_time is not None
        and control_time is not None
        and candidate["time_scope"] == "end_to_end_process"
        and control["time_scope"] == "end_to_end_process"
    ):
        candidate["total_time_ratio_vs_matched_control"] = candidate_time / control_time


def manifest_runs(
    manifest: Path, run_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    with manifest.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoints: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in source:
        if row["phase"] == PREFLIGHT_SUITE:
            continue
        path = (
            run_root
            / row["phase"]
            / row["dataset"]
            / row["row_id"]
            / f"{row['activation']}.jsonl"
        )
        run, eval_rows = raw_run(
            index=int(row["row_index"]),
            model=row["model"],
            dataset=row["dataset"],
            seed=int(row["seed"]),
            train_tokens=int(row["train_tokens"]),
            steps=int(row["steps"]),
            activation=row["activation"],
            optimizer=row["optimizer"],
            method=row["method"],
            phase=row["phase"],
            source_row_index=int(row["row_index"]),
            source_row_id=row["row_id"],
            path=path,
        )
        output[row["optimizer"]].append(run)
        checkpoints[row["optimizer"]].extend(eval_rows)
        key = (
            row["phase"], row["model"], row["dataset"], row["seed"],
            row["train_tokens"], row["steps"], row["optimizer"],
        )
        pairs[key][row["activation"]] = run
    for arms in pairs.values():
        if "silu" in arms and GRAIN_ID in arms:
            pair(arms[GRAIN_ID], arms["silu"])
    return output, checkpoints


def tiller_runs(
    matrix: Path, run_root: Path, analysis_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(matrix.read_text())
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for source in payload["rows"]:
        index = int(source["matrix_index"])
        candidate_path = (
            run_root
            / f"{source['phase']}-{source['dataset']}-seed{source['seed']}-tiller"
            / f"{source['candidate_activation']}.jsonl"
        )
        run, eval_rows = raw_run(
            index=index,
            model=source["model"],
            dataset=source["dataset"],
            seed=int(source["seed"]),
            train_tokens=int(source["max_train_tokens"]),
            steps=int(source["steps"]),
            activation=source["candidate_activation"],
            optimizer="tiller_v1",
            method="tiller",
            phase=source["phase"],
            source_row_index=int(source["source_manifest_row_index"]),
            source_row_id=source["source_manifest_row_id"],
            path=candidate_path,
        )
        result_path = analysis_root / f"row-{index:02d}" / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            required = {
                "schema": "tiller_matched_endpoint_result_v1",
                "status": "complete",
                "matrix_index": index,
            }
            mismatch = {
                key: (result.get(key), value)
                for key, value in required.items()
                if result.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"invalid paired result {result_path}: {mismatch}")
            run.update(
                {
                    "step1000_validation_loss": result["candidate_step1000_loss"],
                    "final_validation_loss": result["candidate_endpoint_loss"],
                    "matched_control_step1000_validation_loss": result[
                        "control_step1000_loss"
                    ],
                    "matched_control_validation_loss": result["control_endpoint_loss"],
                    "lead_at_step1000_vs_matched_control": result[
                        "step1000_absolute_lead"
                    ],
                    "lead_vs_matched_control": result["absolute_endpoint_lead"],
                    "time_scope": "end_to_end_process",
                    "total_seconds": result["candidate_end_to_end_total_seconds"],
                    "matched_control_total_seconds": result[
                        "control_end_to_end_total_seconds"
                    ],
                    "total_time_ratio_vs_matched_control": result[
                        "exact_matched_hardware_end_to_end_total_time_ratio"
                    ],
                    "training_loop_total_seconds": result[
                        "candidate_training_loop_total_seconds"
                    ],
                    "status": "complete",
                }
            )
        rows.append(run)
        checkpoints.extend(eval_rows)
    return rows, checkpoints


def factorial_muon_runs(
    manifest: Path, campaign_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Import the staged 18-layer Muon activation grid into the public schema."""

    matrix_path = campaign_root / "matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError(f"factorial matrix is absent: {matrix_path}")
    matrix = json.loads(matrix_path.read_text())
    factorial_rows = {
        (
            str(row["dataset"]), int(row["seed"]), str(row["activation"]),
            str(row["optimizer"]),
        ): row
        for row in matrix["rows"]
        if row["optimizer"] == "muon"
    }
    if len(factorial_rows) != 30:
        raise RuntimeError(
            f"expected 30 factorial Muon rows, found {len(factorial_rows)}"
        )

    with manifest.open(newline="") as handle:
        public_rows = [
            row for row in csv.DictReader(handle)
            if row["phase"] == FACTORIAL_PHASE and row["optimizer"] == "muon"
        ]
    if len(public_rows) != 30:
        raise RuntimeError(
            f"expected 30 public 18-layer Muon rows, found {len(public_rows)}"
        )

    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in public_rows:
        key = (
            source["dataset"], int(source["seed"]), source["activation"], "muon"
        )
        staged = factorial_rows[key]
        run_path = (
            campaign_root / "runs" / staged["row_id"]
            / f"{source['activation']}.jsonl"
        )
        existing = staged.get("existing_result")
        use_existing = isinstance(existing, dict) and existing.get("status") == "complete"
        if use_existing and run_path.is_file():
            records = safe_records(run_path)
            summary = unique_event(records, "summary") or {}
            endpoint = evaluations(records).get(int(source["steps"]), {})
            use_existing = not (
                summary.get("completed_steps") == int(source["steps"])
                and not summary.get("stopped_early", False)
                and finite(endpoint.get("val_loss")) is not None
            )
        historical_path = None
        if (use_existing or not run_path.is_file()) and isinstance(existing, dict):
            candidate = Path(str(existing.get("canonical_path", "")))
            if candidate.is_file() or use_existing:
                historical_path = candidate
        source_path = historical_path or run_path
        run, eval_rows = raw_run(
            index=int(source["row_index"]),
            model=source["model"],
            dataset=source["dataset"],
            seed=int(source["seed"]),
            train_tokens=int(source["train_tokens"]),
            steps=int(source["steps"]),
            activation=source["activation"],
            optimizer="muon",
            method=source["method"],
            phase=source["phase"],
            source_row_index=int(source["row_index"]),
            source_row_id=source["row_id"],
            path=source_path,
        )
        # The compact public tables carry the verified artifact identity and
        # all evaluation records, not cluster-local paths or scheduler labels.
        run["source_jsonl"] = ""
        run["slurm_job_id"] = ""
        run["slurm_restart_count"] = ""
        run["slurm_node"] = ""
        run["timing_attempt_id"] = ""
        if historical_path is not None:
            # Keep the verified digest while omitting a machine-local source path.
            run["source_jsonl_sha256"] = existing["file_sha256"]
        if use_existing:
            endpoint = float(existing["endpoint"])
            loop_seconds = float(existing["total_seconds"])
            run.update(
                {
                    "steps_completed": int(existing["completed_steps"]),
                    "status": "complete",
                    "step1000_validation_loss": float(existing["step1000"]),
                    "final_validation_loss": endpoint,
                    "final_validation_perplexity": math.exp(endpoint),
                    "time_scope": "training_loop",
                    "total_seconds": loop_seconds,
                    "training_loop_total_seconds": loop_seconds,
                    "mean_seconds_per_step": loop_seconds / int(source["steps"]),
                    "tokens_per_second": (
                        int(source["global_tokens_per_step"])
                        / (loop_seconds / int(source["steps"]))
                    ),
                    "stopped_early": False,
                    "lr_wd_fairness_passed": True,
                    "realized_lr_trace_sha256": existing[
                        "realized_lr_trace_sha256"
                    ],
                    "source_jsonl_sha256": existing["file_sha256"],
                }
            )
        wall_clock = None if use_existing else factorial_process_time(
            campaign_root / "results" / "01_muon"
            / f"matrix-{int(staged['matrix_index'])}" / "WALL_CLOCK.json",
            staged_row=staged,
            artifact=run_path,
            source_freeze=campaign_root / "SOURCE_FREEZE.sha256",
        )
        if wall_clock is not None:
            if run["status"] != "complete":
                raise RuntimeError(
                    f"successful wall clock accompanies incomplete row {staged['row_id']}"
                )
            run["time_scope"] = "end_to_end_process"
            run["total_seconds"] = wall_clock
        rows.append(run)
        checkpoints.extend(eval_rows)
        pairs[(source["dataset"], int(source["seed"]))][source["activation"]] = run

    for arms in pairs.values():
        if "silu" in arms and GRAIN_ID in arms:
            pair(arms[GRAIN_ID], arms["silu"])
    return rows, checkpoints


def factorial_tiller_runs(
    matrix: Path,
    campaign_root: Path,
    published_rows: list[dict[str, Any]],
    published_checkpoints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace a published attempt only with a complete, audited factorial run."""

    campaign_root = campaign_root.resolve()
    staged_rows = json.loads((campaign_root / "matrix.json").read_text())["rows"]
    staged_tiller = {
        int(row["matrix_index"]): row for row in staged_rows
        if row["optimizer"] == "tiller_v1"
    }
    public_tiller = {
        int(row["matrix_index"]): row
        for row in json.loads(matrix.read_text())["rows"]
        if row["phase"] == FACTORIAL_PHASE
    }
    if set(staged_tiller) != set(range(210, 225)) or set(public_tiller) != set(range(30, 45)):
        raise RuntimeError("factorial/public TILLER row inventory changed")
    replacements: dict[int, dict[str, Any]] = {}
    new_checkpoints: list[dict[str, Any]] = []
    for staged_index, staged in staged_tiller.items():
        index = staged_index - 180
        source = public_tiller[index]
        required = {
            key: source[key]
            for key in (
                "model", "dataset", "dataset_name", "dataset_config", "seed",
                "layers", "d_model", "heads", "ffn_dim", "seq_len", "steps",
            )
        }
        required.update(
            activation=source["candidate_activation"],
            optimizer=source["candidate_optimizer"],
            train_tokens=source["max_train_tokens"],
            val_tokens=source["max_val_tokens"],
        )
        if any(staged.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"factorial/public TILLER identity mismatch: {staged_index}")
        artifact = campaign_root / "runs" / staged["row_id"] / f"{staged['activation']}.jsonl"
        result_root = campaign_root / "results/02_tiller" / f"matrix-{staged_index}"
        timing_path = result_root / "WALL_CLOCK.json"
        decision_path = result_root / "STEP1000.json"
        if not all(path.is_file() for path in (artifact, timing_path, decision_path)):
            continue
        timing = json.loads(timing_path.read_text())
        decision = json.loads(decision_path.read_text())
        # A stopped or failed retry cannot supersede a completed historical run.
        if timing.get("process_exit_status") != 0 or decision.get("scientific_stop") is True:
            continue
        records = safe_records(artifact)
        summary = unique_event(records, "summary") or {}
        evals = evaluations(records)
        steps = int(source["steps"])
        if (
            summary.get("completed_steps") != steps
            or summary.get("stopped_early") is not False
            or finite(evals.get(steps, {}).get("val_loss")) is None
        ):
            continue
        config = unique_event(records, "config") or {}
        expected_config = {
            key: staged[key]
            for key in ("seed", "layers", "d_model", "heads", "ffn_dim", "seq_len", "steps",
                        "train_tokens", "val_tokens", "activation", "dataset_config")
        }
        expected_config.update(
            dataset=staged["dataset_name"],
            optimizer="factorized_every_step_rfd_gradient_ledger_muon_v1",
            params=staged["expected_parameter_count"],
            train_token_sample_sha256=staged["token_fingerprints"]["train_token_sample_sha256"],
            val_token_sample_sha256=staged["token_fingerprints"]["validation_token_sample_sha256"],
        )
        identity = config.get("m1_300m_campaign_identity", {})
        if (
            any(config.get(key) != value for key, value in expected_config.items())
            or identity.get("passed") is not True
            or identity.get("scientific_cell_key") != staged["scientific_cell_key"]
        ):
            raise RuntimeError(f"factorial TILLER configuration mismatch: {staged_index}")
        controls = [row for row in staged_rows if row["optimizer"] == "muon"
                    and row["activation"] == "silu" and row["dataset"] == staged["dataset"]
                    and int(row["seed"]) == int(staged["seed"])]
        if len(controls) != 1:
            raise RuntimeError(f"factorial TILLER lacks one exact control: {staged_index}")
        control = controls[0]
        control_path = campaign_root / "runs" / control["row_id"] / "silu.jsonl"
        if control.get("execution_action") == "skip_after_full_evidence_validation":
            control_path = Path(control["existing_result"]["canonical_path"])
            if sha256(control_path) != control["existing_result"]["file_sha256"]:
                raise RuntimeError(f"factorial TILLER historical control changed: {staged_index}")
        control_step1000 = evaluations(safe_records(control_path)).get(1000, {}).get("val_loss")
        candidate_step1000 = evals.get(1000, {}).get("val_loss")
        required_decision = {
            "schema": "tiller_exact_matched_step1000_gate_v1",
            "dataset": staged["dataset"], "seed": int(staged["seed"]),
            "candidate_matrix_index": staged_index,
            "control_matrix_index": int(control["matrix_index"]),
            "candidate_path": str(artifact), "control_path": str(control_path),
            "candidate_step1000_loss": candidate_step1000,
            "control_step1000_loss": control_step1000,
        }
        if (
            decision.get("passed") is not True or decision.get("scientific_stop") is not False
            or any(decision.get(key) != value for key, value in required_decision.items())
            or any(isinstance(value, bool) or finite(value) is None for value in (
                candidate_step1000, control_step1000, decision.get("candidate_lead"),
                evals[steps].get("val_loss"),
            ))
            or decision["candidate_lead"] != control_step1000 - candidate_step1000
            or decision["candidate_lead"] < 0
        ):
            raise RuntimeError(f"invalid factorial TILLER step-1000 decision: {staged_index}")
        wall_seconds = factorial_process_time(
            timing_path, staged_row=staged, artifact=artifact,
            source_freeze=campaign_root / "SOURCE_FREEZE.sha256", stage="02_tiller",
        )
        run, eval_rows = raw_run(
            index=index, model=source["model"], dataset=source["dataset"],
            seed=int(source["seed"]), train_tokens=int(source["max_train_tokens"]),
            steps=steps, activation=source["candidate_activation"], optimizer="tiller_v1",
            method="tiller", phase=source["phase"],
            source_row_index=int(source["source_manifest_row_index"]),
            source_row_id=source["source_manifest_row_id"], path=artifact,
        )
        run.update(time_scope="end_to_end_process", total_seconds=wall_seconds)
        for field in ("source_jsonl", "slurm_job_id", "slurm_restart_count", "slurm_node", "timing_attempt_id"):
            run[field] = ""
        replacements[index] = run
        new_checkpoints.extend(eval_rows)
    return (
        [row for row in published_rows if int(row["run_index"]) not in replacements]
        + list(replacements.values()),
        [row for row in published_checkpoints if int(row["run_index"]) not in replacements]
        + new_checkpoints,
    )


def refresh_tiller_controls(
    output_root: Path,
    control_rows: Iterable[dict[str, Any]],
    *,
    matrix: Path = DEFAULT_MATRIX,
    factorial_campaign: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach every newly available exact control to the published TILLER rows."""

    runs_path = output_root / "tiller" / "runs.csv"
    checkpoints_path = output_root / "tiller" / "checkpoints.csv"
    if not runs_path.is_file() or not checkpoints_path.is_file():
        raise RuntimeError("published TILLER tables are absent")
    with runs_path.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    with checkpoints_path.open(newline="") as handle:
        checkpoints = list(csv.DictReader(handle))
    if factorial_campaign is not None:
        candidates, checkpoints = factorial_tiller_runs(
            matrix, factorial_campaign, candidates, checkpoints
        )
    controls = {
        (
            str(row["model_scale"]), str(row["dataset"]), int(row["seed"]),
            int(row["train_tokens"]), int(row["steps_required"]),
            str(row["optimizer"]),
        ): row
        for row in control_rows
        if row["activation"] == "silu" and row["optimizer"] in {"adamw", "muon"}
    }
    for candidate in candidates:
        if factorial_campaign is not None and candidate["source_phase"] != FACTORIAL_PHASE:
            continue
        for field in (
            "matched_control_step1000_validation_loss",
            "matched_control_validation_loss",
            "lead_at_step1000_vs_matched_control",
            "lead_vs_matched_control",
            "matched_control_total_seconds",
            "total_time_ratio_vs_matched_control",
        ):
            candidate[field] = ""
        key = (
            str(candidate["model_scale"]), str(candidate["dataset"]),
            int(candidate["seed"]), int(candidate["train_tokens"]),
            int(candidate["steps_required"]),
            "adamw" if int(candidate["train_tokens"]) == 100_000_000 else "muon",
        )
        control = controls.get(key)
        if control is not None:
            pair(candidate, control)
    return candidates, checkpoints


def sample_std(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = (
        "model_scale", "dataset", "train_tokens", "steps_required", "activation",
        "activation_display_name", "optimizer", "optimizer_display_name", "method",
        "method_display_name",
    )
    for row in rows:
        group_key = (
            str(row["model_scale"]),
            str(row["dataset"]),
            int(row["train_tokens"]),
            int(row["steps_required"]),
            str(row["activation"]),
            str(row["activation_display_name"]),
            str(row["optimizer"]),
            str(row["optimizer_display_name"]),
            str(row["method"]),
            str(row["method_display_name"]),
        )
        grouped[group_key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        values = dict(zip(keys, key))
        endpoints = [
            value for row in group
            if (value := finite(row["final_validation_loss"])) is not None
            and row["status"] == "complete"
        ]
        controls = [
            value for row in group
            if (value := finite(row["matched_control_validation_loss"])) is not None
        ]
        leads = [
            value for row in group
            if (value := finite(row["lead_vs_matched_control"])) is not None
        ]
        ratios = [
            value for row in group
            if (value := finite(row["total_time_ratio_vs_matched_control"])) is not None
        ]
        seconds = [
            value for row in group if (value := finite(row["total_seconds"])) is not None
        ]
        step_seconds = [
            value for row in group
            if (value := finite(row["mean_seconds_per_step"])) is not None
        ]
        throughput = [
            value for row in group
            if (value := finite(row["tokens_per_second"])) is not None
        ]
        reached = sum(
            int(row["steps_completed"]) >= int(row["steps_required"])
            for row in group
        )
        stopped = sum(
            str(row["stopped_early"]).strip().lower() == "true"
            for row in group
        )
        non_finite = sum(row["status"] == "non_finite" for row in group)
        if non_finite:
            status = "non_finite"
        elif stopped:
            status = "stopped_early"
        elif reached < len(group):
            status = (
                "pending"
                if all(row["status"] == "pending" for row in group)
                else "incomplete"
            )
        elif len(endpoints) < len(group):
            status = "endpoint_loss_unavailable"
        else:
            status = "complete"
        values.update(
            {
                "runs_total": len(group),
                "runs_reached_required_step": reached,
                "runs_stopped_early": stopped,
                "runs_non_finite": non_finite,
                "endpoint_seed_count": len(endpoints),
                "validation_loss_mean": mean(endpoints),
                "validation_loss_sample_std": sample_std(endpoints),
                "matched_control_seed_count": len(controls),
                "matched_control_validation_loss_mean": mean(controls),
                "lead_vs_matched_control_mean": mean(leads),
                "lead_vs_matched_control_sample_std": sample_std(leads),
                "exact_time_ratio_seed_count": len(ratios),
                "exact_total_time_ratio_mean": mean(ratios),
                "exact_total_time_ratio_sample_std": sample_std(ratios),
                "total_seconds_mean": mean(seconds),
                "total_seconds_sample_std": sample_std(seconds),
                "mean_seconds_per_step": mean(step_seconds),
                "tokens_per_second_mean": mean(throughput),
                "status": status,
            }
        )
        output.append(values)
    return output


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish(
    output_root: Path,
    optimizer: str,
    rows: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> None:
    directory = output_root / OUTPUT_DIRECTORIES[optimizer]
    rows.sort(
        key=lambda row: (
            str(row["model_scale"]), int(row["train_tokens"]), str(row["dataset"]),
            int(row["seed"]), str(row["activation"]), int(row["run_index"]),
        )
    )
    checkpoints.sort(
        key=lambda row: (
            str(row["model_scale"]), str(row["dataset"]), int(row["seed"]),
            str(row["activation"]), int(row["step"])
        )
    )
    write_csv(directory / "runs.csv", RUN_FIELDS, rows)
    write_csv(directory / "summary.csv", SUMMARY_FIELDS, summaries(rows))
    write_csv(directory / "checkpoints.csv", CHECKPOINT_FIELDS, checkpoints)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--activation-runs", type=Path,
        default=Path("experiments/runs/activation_optimizer"),
    )
    parser.add_argument(
        "--tiller-runs", type=Path, default=Path("experiments/protocol/runs")
    )
    parser.add_argument(
        "--tiller-analysis", type=Path, default=Path("experiments/protocol/results")
    )
    parser.add_argument(
        "--factorial-campaign",
        type=Path,
        help=(
            "optional staged 18-layer campaign root; import its Muon rows and "
            "completed TILLER endpoints into the corresponding public rows"
        ),
    )
    parser.add_argument(
        "--only-optimizer",
        choices=tuple(OUTPUT_DIRECTORIES),
        action="append",
        help="publish only the selected optimizer table (repeatable)",
    )
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results"))
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="materialize manifest rows that have not produced an output file",
    )
    args = parser.parse_args()

    by_optimizer, checkpoint_rows = manifest_runs(args.manifest, args.activation_runs)
    tiller, tiller_checkpoints = tiller_runs(
        args.matrix, args.tiller_runs, args.tiller_analysis
    )
    by_optimizer["tiller_v1"] = tiller
    checkpoint_rows["tiller_v1"] = tiller_checkpoints
    if args.factorial_campaign is not None:
        factorial, factorial_checkpoints = factorial_muon_runs(
            args.manifest, args.factorial_campaign
        )
        published_runs_path = args.output_root / "muon" / "runs.csv"
        published_checkpoints_path = args.output_root / "muon" / "checkpoints.csv"
        if published_runs_path.is_file():
            with published_runs_path.open(newline="") as handle:
                existing_runs = list(csv.DictReader(handle))
        else:
            existing_runs = by_optimizer.get("muon", [])
        if published_checkpoints_path.is_file():
            with published_checkpoints_path.open(newline="") as handle:
                existing_checkpoints = list(csv.DictReader(handle))
        else:
            existing_checkpoints = checkpoint_rows.get("muon", [])
        replaced_indices = {int(item["run_index"]) for item in factorial}
        by_optimizer["muon"] = [
            row for row in existing_runs
            if not (
                row["model_scale"] == "18l_1024d"
                and int(row["train_tokens"]) == 300_000_000
            )
        ] + factorial
        checkpoint_rows["muon"] = [
            row for row in existing_checkpoints
            if int(row["run_index"]) not in replaced_indices
        ] + factorial_checkpoints
        adamw_path = args.output_root / "adamw" / "runs.csv"
        with adamw_path.open(newline="") as handle:
            adamw_controls = list(csv.DictReader(handle))
        refreshed_tiller, refreshed_tiller_checkpoints = refresh_tiller_controls(
            args.output_root, by_optimizer["muon"] + adamw_controls,
            matrix=args.matrix, factorial_campaign=args.factorial_campaign,
        )
        by_optimizer["tiller_v1"] = refreshed_tiller
        checkpoint_rows["tiller_v1"] = refreshed_tiller_checkpoints
    written = 0
    for optimizer in OUTPUT_DIRECTORIES:
        if args.only_optimizer and optimizer not in args.only_optimizer:
            continue
        rows = by_optimizer.get(optimizer, [])
        if not args.include_pending:
            rows = [row for row in rows if row["status"] != "pending"]
        if not rows:
            continue
        valid_indices = {int(row["run_index"]) for row in rows}
        checkpoints = [
            row for row in checkpoint_rows.get(optimizer, [])
            if int(row["run_index"]) in valid_indices
        ]
        publish(args.output_root, optimizer, rows, checkpoints)
        written += 1
        print(f"wrote {OUTPUT_DIRECTORIES[optimizer]}: {len(rows)} runs, {len(checkpoints)} checkpoints")
    if written == 0:
        raise RuntimeError("no run artifacts were found; existing compact results were left unchanged")


if __name__ == "__main__":
    main()
