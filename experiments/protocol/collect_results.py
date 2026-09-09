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
PRIMARY_PHASE = "18l_1024d_100m_tokens_3050_steps"
CANDIDATE_METHODS = {"tiller_v1": "tiller", "tiller_then_muon_v1": "tiller_then_muon"}
CANDIDATE_STAGES = {"tiller_v1": "02_tiller", "tiller_then_muon_v1": "03_tiller_then_muon"}

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
    "tiller_then_muon_v1": "TILLER→Muon",
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
    "tiller_then_muon_v1": "tiller_then_muon",
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
    if payload.get("schema") == "rationalopt_process_wall_clock_v1" and payload.get("return_code") != 0:
        return None, "training_loop"
    if (
        payload.get("schema") != "rationalopt_process_wall_clock_v1"
        or payload.get("return_code") != 0
        or finite(payload.get("elapsed_seconds")) is None
    ):
        raise RuntimeError(f"invalid process wall-clock sidecar: {sidecar}")
    return float(payload["elapsed_seconds"]), "end_to_end_process"



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
                OPTIMIZER_NAMES[optimizer]
                if optimizer in CANDIDATE_METHODS
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
    sidecar = path.with_suffix(".wall_clock.json")
    if status == "complete" and sidecar.is_file():
        if json.loads(sidecar.read_text()).get("return_code") != 0:
            status = "incomplete"
            endpoint_loss = endpoint_ppl = None
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


def validate_primary_artifact(
    path: Path, source: dict[str, Any], *, campaign_root: Path | None,
    candidate: bool = False,
) -> None:
    """Bind fresh 18-layer observations to the independent 3050-step protocol."""
    data = safe_records(path)
    config = unique_event(data, "config")
    if config is None:
        if data:
            raise RuntimeError(f"primary trajectory has observations but no config: {path}")
        return
    expected = {key: int(source[key]) for key in (
        "seed", "layers", "d_model", "heads", "ffn_dim", "seq_len", "steps", "grad_accum",
    )}
    expected.update(
        activation=source["candidate_activation" if candidate else "activation"],
        optimizer=source["candidate_optimizer" if candidate else "optimizer"],
        dataset=source["dataset_name"], dataset_config=source["dataset_config"],
        dataset_revision=source["dataset_revision"], tokenizer=source["tokenizer"],
        tokenizer_revision=source["tokenizer_revision"],
        train_tokens=int(source["max_train_tokens" if candidate else "train_tokens"]),
        val_tokens=int(source["max_val_tokens" if candidate else "val_tokens"]),
        batch_size_per_gpu=int(source["batch_size"]), world_size=4,
        validation_skip_tokens=int(source["validation_skip_tokens" if candidate else "val_skip_tokens"]),
        train_skip_tokens=int(source["train_skip_tokens"]),
    )
    for field, key in (("optimizer_lr", "lr"), ("optimizer_min_lr", "min_lr"),
                       ("optimizer_weight_decay", "weight_decay"), ("optimizer_beta1", "beta1"),
                       ("optimizer_beta2", "beta2"), ("optimizer_eps", "eps"), ("grad_clip", "grad_clip")):
        expected[field] = float(source[key])
    fingerprints = json.loads((PACKAGE / "token_fingerprints.json").read_text())["cells"]
    token_cell = next(item for item in fingerprints
                      if item["dataset"] == source["dataset"] and item["train_tokens"] == 100_000_000)
    expected.update(train_token_sample_sha256=token_cell["train_token_sample_sha256"],
                    val_token_sample_sha256=token_cell["validation_token_sample_sha256"])
    if candidate:
        expected["experiment_identity"] = (
            "tiller_matrix_v1" if source["candidate_optimizer"] == "tiller_v1"
            else "tiller_then_muon_matrix_v1"
        )
        identity = config.get("tiller_experiment_identity", {})
        if identity.get("passed") is not True or any(identity.get(key) != source[key] for key in (
            "matrix_index", "source_manifest_row_index", "source_manifest_row_id",
        )):
            raise RuntimeError(f"primary candidate matrix identity mismatch: {path}")
    else:
        snapshot = PACKAGE if campaign_root is None else campaign_root / "source_muon/experiments/protocol"
        expected.update(
            experiment_identity="none",
            source_manifest_sha256=sha256(snapshot / "activation_optimizer_manifest.csv"),
            source_manifest_row_sha256=hashlib.sha256(json.dumps(
                source, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")).hexdigest(),
            source_manifest_row_id=source["row_id"], source_manifest_row_index=str(source["row_index"]),
            source_freeze_sha256=sha256(snapshot / "SOURCE_FREEZE.sha256"),
        )
    fairness = config.get("optimizer_lr_wd_fairness", {})
    if (any(config.get(key) != value for key, value in expected.items())
            or fairness.get("passed") is not True or fairness.get("contract") != "exact_lr_wd_v1"):
        raise RuntimeError(f"primary trajectory configuration mismatch: {path}")


def validate_primary_result(
    path: Path, result: dict[str, Any], source: dict[str, Any], artifact: Path,
    result_root: Path, campaign_root: Path | None,
) -> None:
    data = safe_records(artifact)
    summary = unique_event(data, "summary") or {}
    evaluation = evaluations(data)
    if (summary.get("completed_steps") != 3050 or summary.get("stopped_early") is not False
            or any(finite(evaluation.get(step, {}).get("val_loss")) is None for step in (1000, 3050))):
        raise RuntimeError(f"primary result lacks a completed raw endpoint: {path}")
    snapshot = PACKAGE if campaign_root is None else campaign_root / "source_tiller/experiments/protocol"
    if result.get("source_freeze_manifest_sha256") != sha256(snapshot / "SOURCE_FREEZE.sha256"):
        raise RuntimeError(f"primary result source freeze mismatch: {path}")
    checksum = path.with_suffix(path.suffix + ".sha256")
    if not checksum.is_file() or checksum.read_text().strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"primary result checksum mismatch: {path}")
    clock_path = result_root / "CANDIDATE_WALL_CLOCK.json"
    clock = json.loads(clock_path.read_text())
    if (clock.get("schema") != "rationalopt_process_wall_clock_v1"
            or clock.get("arm") != "candidate" or clock.get("return_code") != 0
            or finite(clock.get("elapsed_seconds")) is None or float(clock["elapsed_seconds"]) <= 0
            or result.get("candidate_end_to_end_total_seconds") != clock["elapsed_seconds"]):
        raise RuntimeError(f"primary result lacks successful process timing: {path}")
    for key, target in (("candidate_jsonl", artifact), ("candidate_wall_clock", clock_path)):
        identity = result.get(key, {})
        if Path(str(identity.get("path", ""))).resolve() != target.resolve() or identity.get("sha256") != sha256(target):
            raise RuntimeError(f"primary result {key} is not bound to its artifact: {path}")
    if (result.get("candidate_endpoint_loss") != evaluation[3050]["val_loss"]
            or result.get("candidate_step1000_loss") != evaluation[1000]["val_loss"]
            or finite(result.get("step1000_absolute_lead")) is None
            or float(result["step1000_absolute_lead"]) < 0):
        raise RuntimeError(f"primary result losses or step-1000 decision differ: {path}")
    control_path = Path(str(result.get("control_jsonl", {}).get("path", "")))
    if campaign_root is not None:
        expected_control_path = (campaign_root / "runs/activation_optimizer" / source["phase"]
                                 / source["dataset"] / source["source_manifest_row_id"] / "silu.jsonl")
        if control_path.resolve() != expected_control_path.resolve():
            raise RuntimeError(f"primary result uses another control path: {path}")
    if result.get("control_jsonl", {}).get("sha256") != sha256(control_path):
        raise RuntimeError(f"primary result control artifact hash mismatch: {path}")
    baseline_snapshot = PACKAGE if campaign_root is None else campaign_root / "source_muon/experiments/protocol"
    with (baseline_snapshot / "activation_optimizer_manifest.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))[int(source["source_manifest_row_index"])]
    if baseline["row_id"] != source["source_manifest_row_id"]:
        raise RuntimeError(f"primary result control manifest identity mismatch: {path}")
    validate_primary_artifact(control_path, baseline, campaign_root=campaign_root)
    control_data = safe_records(control_path)
    control_summary = unique_event(control_data, "summary") or {}
    control_evals = evaluations(control_data)
    if (control_summary.get("completed_steps") != 3050 or control_summary.get("stopped_early") is not False
            or any(finite(control_evals.get(step, {}).get("val_loss")) is None for step in (1000, 3050))):
        raise RuntimeError(f"primary result control lacks its independent endpoint: {path}")
    control_clock_path = control_path.with_suffix(".wall_clock.json")
    control_clock = json.loads(control_clock_path.read_text())
    if (control_clock.get("schema") != "rationalopt_process_wall_clock_v1"
            or control_clock.get("arm") != "silu_muon" or control_clock.get("return_code") != 0
            or finite(control_clock.get("elapsed_seconds")) is None or float(control_clock["elapsed_seconds"]) <= 0
            or result.get("control_wall_clock", {}).get("sha256") != sha256(control_clock_path)
            or result.get("control_end_to_end_total_seconds") != control_clock["elapsed_seconds"]):
        raise RuntimeError(f"primary result control timing mismatch: {path}")
    if (result.get("control_step1000_loss") != control_evals[1000]["val_loss"]
            or result.get("control_endpoint_loss") != control_evals[3050]["val_loss"]
            or result["step1000_absolute_lead"] != control_evals[1000]["val_loss"] - evaluation[1000]["val_loss"]
            or result.get("absolute_endpoint_lead") != control_evals[3050]["val_loss"] - evaluation[3050]["val_loss"]):
        raise RuntimeError(f"primary result losses do not match its exact control: {path}")


def manifest_runs(
    manifest: Path, run_root: Path, *, campaign_root: Path | None = None
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
        if row["phase"] == PRIMARY_PHASE and path.is_file():
            validate_primary_artifact(path, row, campaign_root=campaign_root)
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
    matrix: Path, run_root: Path, analysis_root: Path, *, campaign_root: Path | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(matrix.read_text())
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for source in payload["rows"]:
        index = int(source["matrix_index"])
        optimizer = source["candidate_optimizer"]
        method = CANDIDATE_METHODS[optimizer]
        candidate_root = run_root
        result_root = analysis_root / f"row-{index:02d}"
        if campaign_root is not None and source["phase"] == PRIMARY_PHASE:
            stage = CANDIDATE_STAGES[optimizer]
            candidate_root = campaign_root / "runs" / stage
            result_root = campaign_root / "results" / stage / f"matrix-{index}"
        candidate_path = (
            candidate_root
            / f"{source['phase']}-{source['dataset']}-seed{source['seed']}-{method}"
            / f"{source['candidate_activation']}.jsonl"
        )
        if source["phase"] == PRIMARY_PHASE and candidate_path.is_file():
            validate_primary_artifact(candidate_path, source, campaign_root=campaign_root, candidate=True)
        run, eval_rows = raw_run(
            index=index,
            model=source["model"],
            dataset=source["dataset"],
            seed=int(source["seed"]),
            train_tokens=int(source["max_train_tokens"]),
            steps=int(source["steps"]),
            activation=source["candidate_activation"],
            optimizer=optimizer,
            method=method,
            phase=source["phase"],
            source_row_index=int(source["source_manifest_row_index"]),
            source_row_id=source["source_manifest_row_id"],
            path=candidate_path,
        )
        result_path = result_root / "RESULT.json"
        if source["phase"] == PRIMARY_PHASE and run["status"] == "complete" and not result_path.is_file():
            run.update(status="incomplete", final_validation_loss="", final_validation_perplexity="")
        screen_path = result_root / "STEP1000_SCREEN.json"
        if screen_path.is_file():
            screen = json.loads(screen_path.read_text())
            if screen.get("matrix_index") != index:
                raise RuntimeError(f"step-1000 screen has the wrong matrix identity: {screen_path}")
            if screen.get("status") == "failed_negative_interrupted":
                if candidate_path.is_file():
                    observed = evaluations(safe_records(candidate_path)).get(1000, {}).get("val_loss")
                    if (finite(observed) is None or screen.get("candidate_step1000_loss") != observed
                            or finite(screen.get("control_step1000_loss")) is None
                            or screen.get("candidate_step1000_lead") != screen["control_step1000_loss"] - observed
                            or screen["candidate_step1000_lead"] >= 0):
                        raise RuntimeError(f"negative screen does not match the candidate trajectory: {screen_path}")
                    run.update(status="stopped_early", stopped_early=True,
                               final_validation_loss="", final_validation_perplexity="")
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            required = {
                "schema": "tiller_matched_endpoint_result_v1",
                "status": "complete",
                "matrix_index": index,
                "model": source["model"], "dataset": source["dataset"],
                "seed": int(source["seed"]), "steps": int(source["steps"]),
            }
            if source["phase"] == PRIMARY_PHASE:
                required.update(phase=PRIMARY_PHASE, candidate_optimizer=optimizer)
            mismatch = {
                key: (result.get(key), value)
                for key, value in required.items()
                if result.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"invalid paired result {result_path}: {mismatch}")
            if source["phase"] == PRIMARY_PHASE:
                validate_primary_result(result_path, result, source, candidate_path,
                                        result_root, campaign_root)
                if run["status"] == "stopped_early":
                    raise RuntimeError(f"stopped candidate also has a complete result: {result_path}")
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


def preserve_published_attempts(
    output_root: Path, optimizer: str, rows: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain legacy 12-layer data and never splice a partial retry into an endpoint."""
    directory = output_root / OUTPUT_DIRECTORIES[optimizer]
    if not (directory / "runs.csv").is_file():
        return rows, checkpoints
    with (directory / "runs.csv").open(newline="") as handle:
        previous = list(csv.DictReader(handle))
    fields = ("run_index", "model_scale", "dataset", "seed", "train_tokens",
              "steps_required", "activation", "optimizer", "source_phase", "source_row_id")
    key = lambda row: tuple(str(row[field]) for field in fields)
    existing = {key(row): row for row in previous}
    retained = set()
    merged = []
    for row in rows:
        old = existing.get(key(row))
        if old is not None and (
            row["model_scale"] == "12l_768d"
            or row["status"] == "pending"
            or (old["status"] == "complete" and row["status"] != "complete")
        ):
            merged.append(old)
            retained.add(int(old["run_index"]))
        else:
            merged.append(row)
    checkpoint_path = directory / "checkpoints.csv"
    with checkpoint_path.open(newline="") as handle:
        prior_checkpoints = list(csv.DictReader(handle))
    return merged, (
        [row for row in checkpoints if int(row["run_index"]) not in retained]
        + [row for row in prior_checkpoints if int(row["run_index"]) in retained]
    )


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
    parser.add_argument("--campaign", type=Path,
                        help="independent 18-layer 100M campaign with source_muon/source_tiller snapshots")
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
    if args.campaign is not None:
        args.activation_runs = args.campaign / "runs/activation_optimizer"
    by_optimizer, checkpoint_rows = manifest_runs(
        args.manifest, args.activation_runs, campaign_root=args.campaign
    )
    tiller, tiller_checkpoints = tiller_runs(
        args.matrix, args.tiller_runs, args.tiller_analysis, campaign_root=args.campaign
    )
    for optimizer in CANDIDATE_METHODS:
        by_optimizer[optimizer] = [row for row in tiller if row["optimizer"] == optimizer]
        checkpoint_rows[optimizer] = [row for row in tiller_checkpoints if row["optimizer"] == optimizer]
    for optimizer in OUTPUT_DIRECTORIES:
        for row in by_optimizer.get(optimizer, []):
            if row["source_phase"] == PRIMARY_PHASE:
                for field in ("source_jsonl", "slurm_job_id", "slurm_restart_count", "slurm_node", "timing_attempt_id"):
                    row[field] = ""
        by_optimizer[optimizer], checkpoint_rows[optimizer] = preserve_published_attempts(
            args.output_root, optimizer, by_optimizer.get(optimizer, []), checkpoint_rows.get(optimizer, [])
        )
    controls = {
        (row["source_phase"], row["dataset"], int(row["seed"]), row["optimizer"]): row
        for rows in by_optimizer.values() for row in rows if row["activation"] == "silu"
    }
    for optimizer, rows in by_optimizer.items():
        for row in rows:
            if row["source_phase"] != PRIMARY_PHASE or row["activation"] != GRAIN_ID:
                continue
            control_optimizer = "muon" if optimizer in CANDIDATE_METHODS else optimizer
            control = controls.get((PRIMARY_PHASE, row["dataset"], int(row["seed"]), control_optimizer))
            if control is not None:
                pair(row, control)
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
