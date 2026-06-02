#!/usr/bin/env python3
"""Validate tiny CUDA/DDP telemetry and optimizer smoke JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_TRAIN_KEYS = {
    "grad_global_norm_before_clip",
    "grad_clip_triggered",
    "grad_clip_threshold",
    "forward_backward_seconds",
    "optimizer_step_seconds",
    "cuda_max_memory_allocated",
    "cuda_max_memory_reserved",
}

REQUIRED_EVAL_KEYS = {
    "probe_logit_rms",
    "probe_logit_delta_rms_since_prev_eval",
    "probe_logit_delta_rms_since_step1",
    "probe_kl_since_prev_eval",
    "probe_kl_since_step1",
}

REQUIRED_RLB_KEYS = {
    "rlb_output_rms_mean_by_layer",
    "rlb_derivative_rms_mean_by_layer",
    "rlb_atom_rms_mean_by_layer",
    "denominator_abs_min_by_layer",
    "denominator_abs_p01_by_layer",
    "w_in_rms_mean_by_layer",
    "w_out_rms_mean_by_layer",
    "log_w_in_over_w_out_by_layer",
    "log_norm_product_by_layer",
}

REQUIRED_MATRIX_POLICY_KEYS = {
    "matrix_policy_muon_mix_mean_by_role",
    "matrix_policy_adam_lr_scale_mean_by_role",
    "matrix_policy_update_rms_by_role",
    "matrix_policy_weight_rms_by_role",
    "matrix_policy_update_to_weight_rms_by_role",
    "matrix_policy_group_scale_mean",
    "matrix_policy_group_scale_std",
    "matrix_policy_pressure_mean",
    "matrix_policy_activity_mean",
}

REQUIRED_SVD_PREFIXES = (
    "svd_entropy_attn_",
    "svd_entropy_rlb_",
)

DEFAULT_REQUIRED_OPTIMIZERS = (
    "adamw",
    "lion",
    "ademamix",
    "schedule_free_adamw",
    "adafactor_came",
    "soap_adamw",
    "rational_matrix_policy_onpolicy",
)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(item is not None for item in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def read_jsonl(path: Path) -> dict[str, Any] | None:
    config: dict[str, Any] | None = None
    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "config":
                config = record
            elif event == "train":
                train.append(record)
            elif event == "eval":
                evals.append(record)
            elif event == "summary":
                summary = record
    if config is None:
        return None
    return {"path": path, "config": config, "train": train, "eval": evals, "summary": summary}


def has_keys(records: list[dict[str, Any]], keys: set[str]) -> set[str]:
    found: set[str] = set()
    for record in records:
        for key in keys:
            if key in record and value_present(record.get(key)):
                found.add(key)
    return found


def has_svd(records: list[dict[str, Any]]) -> bool:
    for record in records:
        for key, value in record.items():
            if key.startswith(REQUIRED_SVD_PREFIXES) and value_present(value):
                return True
    return False


def final_eval(evals: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    for record in reversed(evals):
        loss = record.get("val_loss")
        if finite(loss):
            return int(record.get("step", -1)), float(loss)
    return None, None


def validate_run(run: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    config = run["config"]
    train = run["train"]
    evals = run["eval"]
    summary = run["summary"] or {}
    optimizer = str(config.get("optimizer"))
    activation = str(config.get("activation"))
    steps = int(config.get("steps", -1))
    summary_steps = int(summary.get("steps", -1)) if summary else -1
    row = {
        "path": str(run["path"]),
        "optimizer": optimizer,
        "activation": activation,
        "steps": steps,
        "summary_steps": summary_steps,
        "final_eval_step": None,
        "final_val_loss": None,
        "mean_seconds_per_step": summary.get("mean_seconds_per_step"),
        "tokens_per_second": summary.get("tokens_per_second"),
    }
    errors: list[str] = []
    if summary_steps != steps:
        errors.append(f"summary steps {summary_steps} != config steps {steps}")
    if not train:
        errors.append("no train records")
    if not evals:
        errors.append("no eval records")

    train_found = has_keys(train, REQUIRED_TRAIN_KEYS)
    train_missing = REQUIRED_TRAIN_KEYS - train_found
    if train_missing:
        errors.append("missing train keys: " + ", ".join(sorted(train_missing)))

    eval_found = has_keys(evals, REQUIRED_EVAL_KEYS)
    eval_missing = REQUIRED_EVAL_KEYS - eval_found
    if eval_missing:
        errors.append("missing eval keys: " + ", ".join(sorted(eval_missing)))
    if not has_svd(evals):
        errors.append("missing SVD entropy eval telemetry")

    if activation.startswith("rlb_"):
        rlb_found = has_keys(train, REQUIRED_RLB_KEYS)
        rlb_missing = REQUIRED_RLB_KEYS - rlb_found
        if rlb_missing:
            errors.append("missing RLB train keys: " + ", ".join(sorted(rlb_missing)))

    if optimizer == "rational_matrix_policy_onpolicy":
        matrix_found = has_keys(train, REQUIRED_MATRIX_POLICY_KEYS)
        matrix_missing = REQUIRED_MATRIX_POLICY_KEYS - matrix_found
        if matrix_missing:
            errors.append("missing MatrixPolicy keys: " + ", ".join(sorted(matrix_missing)))

    step, loss = final_eval(evals)
    row["final_eval_step"] = step
    row["final_val_loss"] = loss
    if step != steps:
        errors.append(f"final eval step {step} != config steps {steps}")
    return row, errors


def write_summary(path: Path, rows: list[dict[str, Any]], errors: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ICLR Optimizer Validation Summary\n\n")
        handle.write(f"runs checked: {len(rows)}\n\n")
        if errors:
            handle.write("## Failures\n\n")
            for source, messages in sorted(errors.items()):
                handle.write(f"- `{source}`: {'; '.join(messages)}\n")
            handle.write("\n")
        else:
            handle.write("All validation runs passed required telemetry checks.\n\n")
        handle.write("## Runs\n\n")
        handle.write("| optimizer | activation | steps | final loss | seconds/step | source |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | --- |\n")
        for row in sorted(rows, key=lambda item: (item["optimizer"], item["activation"])):
            loss = "" if row["final_val_loss"] is None else f"{row['final_val_loss']:.6f}"
            sec = row.get("mean_seconds_per_step")
            sec_text = "" if sec is None else f"{float(sec):.4f}"
            handle.write(
                f"| {row['optimizer']} | {row['activation']} | {row['steps']} | {loss} | {sec_text} | `{row['path']}` |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--required-optimizers", nargs="*", default=list(DEFAULT_REQUIRED_OPTIMIZERS))
    args = parser.parse_args()

    runs = []
    for path in sorted(args.run_root.glob("**/*.jsonl")):
        if ".incomplete_" in str(path):
            continue
        run = read_jsonl(path)
        if run is not None:
            runs.append(run)

    rows: list[dict[str, Any]] = []
    errors: dict[str, list[str]] = {}
    seen_by_optimizer: dict[str, int] = defaultdict(int)
    for run in runs:
        row, messages = validate_run(run)
        rows.append(row)
        seen_by_optimizer[row["optimizer"]] += 1
        if messages:
            errors[row["path"]] = messages

    for optimizer in args.required_optimizers:
        if seen_by_optimizer[optimizer] == 0:
            errors[f"optimizer:{optimizer}"] = ["no completed JSONL run found"]

    output_md = args.output_md or (args.run_root / "validation_summary.md")
    write_summary(output_md, rows, errors)
    print(json.dumps({"runs": len(rows), "failures": len(errors), "summary": str(output_md)}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
