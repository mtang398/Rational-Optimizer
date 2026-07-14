#!/usr/bin/env python3
"""Strict E9 validation and result extraction from the frozen manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


SCIENTIFIC_MANIFEST = Path("abalation/manifests/e9_100m_manifest.csv")
SCIENTIFIC_ROOT = Path("abalation/runs/e9_100m")
PREFLIGHT_MANIFEST = Path("abalation/manifests/e9_preflight_manifest.csv")
PREFLIGHT_ROOT = Path("abalation/runs/e9_preflight")
RESULT_ROOT = Path("abalation/results/e9_100m")
TOKENS_PER_STEP = 32768
TARGETS = {
    "dclm": [4.65, 4.55, 4.45],
    "fineweb_edu": [4.50, 4.40, 4.30],
    "fineweb": [4.70, 4.60, 4.50],
    "dolma_sample": [4.70, 4.60, 4.50],
    "c4_en": [4.70, 4.60, 4.50],
}


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    records: list[dict[str, object]] = []
    issues: list[str] = []
    if not path.exists():
        return records, ["missing_jsonl"]
    with path.open(errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.lstrip().startswith("{"):
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                issues.append(f"invalid_json_line_{line_number}")
    return records, issues


def jsonl_path(root: Path, row: dict[str, str]) -> Path:
    return root / row["phase"] / row["dataset"] / row["row_id"] / f"{row['activation']}.jsonl"


def validate_record_set(row: dict[str, str], records: list[dict[str, object]], preflight: bool) -> list[str]:
    issues: list[str] = []
    expected_steps = int(row["steps"])
    configs = [record for record in records if record.get("event") == "config"]
    summaries = [record for record in records if record.get("event") == "summary"]
    trains = [record for record in records if record.get("event") == "train"]
    evals = [record for record in records if record.get("event") == "eval"]
    if len(configs) != 1:
        issues.append(f"config_count={len(configs)}")
    if len(summaries) != 1:
        issues.append(f"summary_count={len(summaries)}")
    for event, event_records in (("train", trains), ("eval", evals)):
        steps = [int(record.get("step", -1)) for record in event_records]
        if len(steps) != len(set(steps)):
            issues.append(f"duplicate_{event}_steps")
    if any(not finite(record.get("loss")) for record in trains):
        issues.append("nonfinite_train_loss")
    if any(not finite(record.get("val_loss")) for record in evals):
        issues.append("nonfinite_val_loss")
    if configs:
        config = configs[0]
        expected_config = {
            "activation": row["activation"],
            "optimizer": row["optimizer"],
            "seed": int(row["seed"]),
            "steps": expected_steps,
            "global_tokens_per_step": TOKENS_PER_STEP,
            "warmup_steps": 200,
            "grad_clip": 1.0,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                issues.append(f"config_{key}={config.get(key)!r}_expected_{expected!r}")
    if summaries:
        summary = summaries[0]
        completed_steps = int(summary.get("completed_steps", -1))
        if preflight and completed_steps != expected_steps:
            issues.append(f"completed_steps={completed_steps}")
        if not preflight and completed_steps != expected_steps and not bool(summary.get("stopped_early")):
            issues.append(f"unterminated_completed_steps={completed_steps}")
        for key in (
            "total_seconds",
            "training_loop_tokens_per_second",
            "grad_clip_trigger_fraction",
            "cuda_run_max_memory_allocated",
            "cuda_run_max_memory_reserved",
        ):
            if not finite(summary.get(key)):
                issues.append(f"missing_or_nonfinite_{key}")
    if preflight:
        expected_train_steps = [1, *range(10, expected_steps + 1, 10)]
        expected_eval_steps = [1, 40, expected_steps]
        if sorted(int(record.get("step", -1)) for record in trains) != expected_train_steps:
            issues.append("unexpected_train_event_steps")
        if sorted(int(record.get("step", -1)) for record in evals) != expected_eval_steps:
            issues.append("unexpected_eval_event_steps")
        if not any(int(record.get("step", -1)) == expected_steps for record in evals):
            issues.append("missing_final_eval")
        for record in evals:
            if not finite(record.get("active_seconds_at_val_loss")):
                issues.append("missing_active_seconds_at_val_loss")
            if not finite(record.get("active_seconds_after_event")):
                issues.append("missing_active_seconds_after_event")
            if not finite(record.get("eval_seconds")):
                issues.append("missing_eval_seconds")
    return list(dict.fromkeys(issues))


def last_train(records: list[dict[str, object]]) -> dict[str, object]:
    trains = [record for record in records if record.get("event") == "train"]
    return max(trains, key=lambda record: int(record.get("step", -1)), default={})


def validate_preflight_semantics(row: dict[str, str], records: list[dict[str, object]]) -> list[str]:
    arm = row["arm_id"]
    config = next((record for record in records if record.get("event") == "config"), {})
    issues: list[str] = []
    common_expected = {
        "e9_row_id": row["row_id"],
        "e9_arm_id": arm,
        "e9_design_version": row["design_version"],
        "train_tokens": int(row["train_tokens"]),
        "val_tokens": int(row["val_tokens"]),
        "seq_len": int(row["seq_len"]),
        "batch_size_per_gpu": int(row["batch_size"]),
        "grad_accum": int(row["grad_accum"]),
        "params": 123_552_672 if arm != "A0" else config.get("params"),
    }
    for key, expected in common_expected.items():
        if config.get(key) != expected:
            issues.append(f"config_{key}={config.get(key)!r}_expected_{expected!r}")
    for key in (
        "e9_manifest_sha256",
        "e9_freeze_sha256",
        "e9_runtime_freeze_sha256",
        "initial_state_sha256",
        "train_token_sample_sha256",
        "val_token_sample_sha256",
        "first_batch_index_sha256",
        "validation_index_sha256",
    ):
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64:
            issues.append(f"missing_or_invalid_{key}")
    gpu_metadata = config.get("gpu_metadata")
    if not isinstance(gpu_metadata, list) or len(gpu_metadata) != 4:
        issues.append("invalid_gpu_metadata")
    elif any("A6000" not in str(item.get("name", "")) for item in gpu_metadata):
        issues.append("non_a6000_gpu_metadata")

    if arm in {"A0", "A1"}:
        return issues

    role_depth_off = arm in {"A2", "A5", "A7", "A9"}
    group_off = arm in {"A2", "A4"}
    muon_zero = arm in {"A2", "A9"}
    pair_off = arm in {"A2", "A8"}
    matrix_expected = {
        "rational_matrix_policy_backbone_optimizer": "adamw",
        "rational_matrix_policy_backbone_beta2": 0.999,
        "rational_matrix_policy_beta2": 0.999,
        "rational_matrix_policy_adam_lr_scale": 3.0,
        "rational_matrix_policy_adam_role_strength": 0.0 if role_depth_off else 1.20,
        "rational_matrix_policy_input_depth_gain": 0.0 if role_depth_off else -0.50,
        "rational_matrix_policy_output_depth_gain": 0.0 if role_depth_off else 1.00,
        "rational_matrix_policy_muon_strength": 0.0 if muon_zero else 0.75,
        "rational_matrix_policy_muon_lr_scale": 0.0 if muon_zero else 1.0,
        "rational_matrix_policy_max_muon": 0.0 if muon_zero else 0.75,
        "rational_matrix_policy_apply_muon_update": arm not in {"A6", "A7"},
        "rational_matrix_policy_group_gain_strength": 0.0 if group_off else 0.20,
        "rational_matrix_policy_group_pressure_strength": 0.0 if group_off else 0.10,
        "rational_matrix_policy_group_activity_damping": 0.0 if group_off else 0.20,
        "rlb_gauge_strength": 0.0 if pair_off else 0.50,
    }
    for key, expected in matrix_expected.items():
        if config.get(key) != expected:
            issues.append(f"config_{key}={config.get(key)!r}_expected_{expected!r}")

    trains = [record for record in records if record.get("event") == "train"]
    train = last_train(records)
    required_train_keys = {
        "matrix_policy_group_policy_enabled",
        "matrix_policy_muon_mix_mean_by_role",
        "matrix_policy_applied_muon_mix_mean_by_role",
        "matrix_policy_muon_mix_by_layer_role",
        "matrix_policy_applied_muon_mix_by_layer_role",
        "matrix_policy_adam_lr_scale_by_layer_role",
        "matrix_policy_realized_adam_lr_scale_by_layer_role",
        "matrix_policy_group_grad_rms_pre_mean_by_role",
        "matrix_policy_group_grad_rms_post_mean_by_role",
        "matrix_policy_group_grad_rms_ratio_mean_by_role",
        "matrix_policy_update_to_weight_rms_by_role",
        "matrix_policy_pair_rescale_enabled",
        "matrix_policy_pair_rescale_applied",
    }
    for record in trains:
        missing = sorted(required_train_keys - record.keys())
        if missing:
            issues.append(f"missing_train_telemetry_step_{record.get('step')}={','.join(missing)}")
        ratios = record.get("matrix_policy_update_to_weight_rms_by_role") or {}
        if not ratios or any(not finite(value) for value in ratios.values()):
            issues.append(f"invalid_update_ratio_step_{record.get('step')}")
        if int(record.get("step", -1)) >= 10:
            for key in (
                "matrix_policy_pair_log_move_abs_mean",
                "matrix_policy_pair_clip_fraction",
                "matrix_policy_pair_target_mismatch_abs_mean",
                "matrix_policy_pair_local_probe_relative_delta_max",
            ):
                if key not in record or not finite(record.get(key)):
                    issues.append(f"missing_{key}_step_{record.get('step')}")
    group_enabled = any(bool(record.get("matrix_policy_group_policy_enabled")) for record in trains)
    if group_enabled != (arm not in {"A2", "A4"}):
        issues.append(f"group_gate_semantics={group_enabled}")

    scheduled_max = max(
        (
            float(value)
            for record in trains
            for value in (record.get("matrix_policy_muon_mix_mean_by_role") or {}).values()
        ),
        default=0.0,
    )
    applied_max = max(
        (
            float(value)
            for record in trains
            for value in (record.get("matrix_policy_applied_muon_mix_mean_by_role") or {}).values()
        ),
        default=0.0,
    )
    if arm in {"A2", "A9"}:
        if scheduled_max != 0.0 or applied_max != 0.0:
            issues.append("muon_schedule_should_be_zero")
    elif arm in {"A6", "A7"}:
        if scheduled_max <= 0.0 or applied_max != 0.0:
            issues.append("suppressed_muon_branch_not_observed")
    elif scheduled_max <= 0.0 or applied_max <= 0.0:
        issues.append("active_muon_branch_not_observed")

    if arm in {"A6", "A7"}:
        attenuated = False
        for record in trains:
            scheduled_map = record.get("matrix_policy_muon_mix_by_layer_role") or {}
            nominal_map = record.get("matrix_policy_adam_lr_scale_by_layer_role") or {}
            realized_map = record.get("matrix_policy_realized_adam_lr_scale_by_layer_role") or {}
            for key, fraction in scheduled_map.items():
                if (
                    float(fraction) > 0.0
                    and key in nominal_map
                    and key in realized_map
                    and float(realized_map[key]) < float(nominal_map[key])
                ):
                    attenuated = True
        if not attenuated:
            issues.append("suppressed_muon_arm_missing_adam_attenuation")

    pair_enabled = any(bool(record.get("matrix_policy_pair_rescale_enabled")) for record in trains)
    pair_applied = any(bool(record.get("matrix_policy_pair_rescale_applied")) for record in trains)
    if arm in {"A2", "A8"}:
        if pair_enabled or pair_applied:
            issues.append("pair_rescale_should_be_disabled")
    elif not pair_enabled or not pair_applied:
        issues.append("pair_rescale_not_observed")

    if arm in {"A2", "A4"}:
        ratios = [
            float(value)
            for record in trains
            for value in (record.get("matrix_policy_group_grad_rms_ratio_mean_by_role") or {}).values()
        ]
        if not ratios:
            issues.append("missing_disabled_group_gate_ratios")
        elif any(abs(value - 1.0) > 1.0e-5 for value in ratios):
            issues.append("disabled_group_gate_changed_gradients")
    if arm in {"A2", "A8"}:
        deltas = [
            float(record["matrix_policy_pair_local_probe_relative_delta_max"])
            for record in trains
            if record.get("matrix_policy_pair_local_probe_relative_delta_max") is not None
        ]
        if not deltas:
            issues.append("missing_disabled_pair_probe_delta")
        elif any(delta != 0.0 for delta in deltas):
            issues.append("disabled_pair_rescale_changed_probe")
    return issues


def normalized_auc(evals: list[dict[str, object]], horizon: int = 3050) -> float | None:
    points = sorted(
        (int(record["step"]), float(record["val_loss"]))
        for record in evals
        if 50 <= int(record.get("step", -1)) <= horizon
        and int(record.get("step", -1)) % 50 == 0
        and finite(record.get("val_loss"))
    )
    expected = list(range(50, horizon + 1, 50))
    if [step for step, _ in points] != expected:
        return None
    area = sum(
        (right_step - left_step) * (left_loss + right_loss) / 2.0
        for (left_step, left_loss), (right_step, right_loss) in zip(points, points[1:])
    )
    return area / float(horizon - 50)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_preflight_validation() -> int:
    rows = list(csv.DictReader(PREFLIGHT_MANIFEST.open(newline="")))
    coverage = []
    configs_by_arm: dict[str, dict[str, object]] = {}
    for row in rows:
        path = jsonl_path(PREFLIGHT_ROOT, row)
        records, issues = read_jsonl(path)
        issues.extend(validate_record_set(row, records, preflight=True))
        if not issues:
            issues.extend(validate_preflight_semantics(row, records))
        config = next((record for record in records if record.get("event") == "config"), None)
        if config is not None:
            configs_by_arm[row["arm_id"]] = config
        coverage.append(
            {
                "row_index": row["row_index"],
                "arm_id": row["arm_id"],
                "row_id": row["row_id"],
                "status": "pass" if not issues else "fail",
                "issues": ";".join(dict.fromkeys(issues)),
                "jsonl": str(path),
            }
        )
    fingerprint_keys = (
        "train_token_sample_sha256",
        "val_token_sample_sha256",
        "first_batch_index_sha256",
        "validation_index_sha256",
    )
    for key in fingerprint_keys:
        values = {config.get(key) for config in configs_by_arm.values()}
        if None in values or len(values) != 1:
            for item in coverage:
                item["status"] = "fail"
                item["issues"] = f"{item['issues']};cross_arm_{key}_mismatch".strip(";")
    rlb_hashes = {
        configs_by_arm[arm].get("initial_state_sha256")
        for arm in [f"A{index}" for index in range(1, 10)]
        if arm in configs_by_arm
    }
    if len(configs_by_arm) != 10 or None in rlb_hashes or len(rlb_hashes) != 1:
        for item in coverage:
            if item["arm_id"] != "A0":
                item["status"] = "fail"
                item["issues"] = f"{item['issues']};rlb_initial_state_sha256_mismatch".strip(";")
    out = Path("abalation/results/e9_preflight/coverage.csv")
    write_csv(out, coverage, ["row_index", "arm_id", "row_id", "status", "issues", "jsonl"])
    failures = [row for row in coverage if row["status"] != "pass"]
    pass_path = Path("abalation/results/e9_preflight/preflight_pass.json")
    if failures:
        pass_path.unlink(missing_ok=True)
    else:
        freeze_path = Path("abalation/submissions/e9_frozen_inputs.json")
        if not freeze_path.is_file():
            raise RuntimeError(f"missing E9 freeze metadata: {freeze_path}")
        frozen = json.loads(freeze_path.read_text())
        evidence_rows = []
        manifest_by_id = {row["row_id"]: row for row in rows}
        for item in coverage:
            row = manifest_by_id[item["row_id"]]
            path = jsonl_path(PREFLIGHT_ROOT, row)
            evidence_rows.append(
                {
                    "arm_id": row["arm_id"],
                    "row_id": row["row_id"],
                    "jsonl_sha256": file_sha256(path),
                }
            )
        evidence = {
            "freeze_sha256": frozen["freeze_sha256"],
            "runtime_sha256": frozen["runtime_sha256"],
            "preflight_manifest_sha256": frozen["files"][str(PREFLIGHT_MANIFEST)],
            "rows": evidence_rows,
        }
        pass_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"E9 preflight: {len(coverage) - len(failures)}/{len(coverage)} arms passed; {out}")
    for row in failures:
        print(f"  {row['arm_id']}: {row['issues']}")
    return 1 if failures else 0


def run_scientific_analysis() -> int:
    manifest_rows = list(csv.DictReader(SCIENTIFIC_MANIFEST.open(newline="")))
    coverage: list[dict[str, object]] = []
    complete: list[dict[str, object]] = []
    for row in manifest_rows:
        path = jsonl_path(SCIENTIFIC_ROOT, row)
        records, issues = read_jsonl(path)
        issues.extend(validate_record_set(row, records, preflight=False))
        config = next((record for record in records if record.get("event") == "config"), {})
        summary = next((record for record in records if record.get("event") == "summary"), {})
        evals = sorted(
            [record for record in records if record.get("event") == "eval"],
            key=lambda record: int(record.get("step", -1)),
        )
        finite_completion = (
            not issues
            and int(summary.get("completed_steps", -1)) == int(row["steps"])
            and not bool(summary.get("stopped_early"))
            and any(int(record.get("step", -1)) == int(row["steps"]) for record in evals)
        )
        status = "complete" if finite_completion else ("terminal_failure" if summary.get("stopped_early") else "invalid")
        coverage.append(
            {
                "row_index": row["row_index"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "arm_id": row["arm_id"],
                "status": status,
                "issues": ";".join(dict.fromkeys(issues)),
                "jsonl": str(path),
            }
        )
        if finite_completion:
            final_eval = next(record for record in evals if int(record.get("step", -1)) == int(row["steps"]))
            complete.append({"manifest": row, "config": config, "summary": summary, "evals": evals, "final": final_eval})

    coverage_fields = ["row_index", "dataset", "seed", "arm_id", "status", "issues", "jsonl"]
    write_csv(RESULT_ROOT / "coverage.csv", coverage, coverage_fields)

    final_rows = []
    runtime_rows = []
    mechanism_rows = []
    by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for run in complete:
        row = run["manifest"]
        summary = run["summary"]
        final_eval = run["final"]
        auc = normalized_auc(run["evals"])
        item = {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "arm_id": row["arm_id"],
            "method": row["method"],
            "final_val_loss": final_eval["val_loss"],
            "final_val_ppl": final_eval["val_ppl"],
            "normalized_val_loss_auc": auc,
        }
        final_rows.append(item)
        by_key[(row["dataset"], row["seed"], row["arm_id"])] = item
        runtime_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "arm_id": row["arm_id"],
                "total_seconds": summary.get("total_seconds"),
                "mean_seconds_per_step": summary.get("mean_seconds_per_step"),
                "tokens_per_second": summary.get("tokens_per_second"),
                "training_loop_tokens_per_second": summary.get("training_loop_tokens_per_second"),
                "peak_allocated_bytes": summary.get("cuda_run_max_memory_allocated"),
                "peak_reserved_bytes": summary.get("cuda_run_max_memory_reserved"),
                "grad_clip_trigger_fraction": summary.get("grad_clip_trigger_fraction"),
                "slurm_job_id": summary.get("slurm_job_id"),
                "slurm_restart_count": summary.get("slurm_restart_count"),
                "slurm_node": summary.get("slurm_node"),
                "timing_attempt_id": summary.get("timing_attempt_id"),
            }
        )
        for record in [item for item in read_jsonl(Path(coverage[int(row["row_index"])]["jsonl"]))[0] if item.get("event") == "train"]:
            selected = {key: value for key, value in record.items() if key.startswith("matrix_policy_") or key.startswith("rlb_")}
            mechanism_rows.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "arm_id": row["arm_id"],
                    "step": record.get("step"),
                    "telemetry_json": json.dumps(selected, sort_keys=True),
                }
            )

    write_csv(
        RESULT_ROOT / "final_loss.csv",
        final_rows,
        ["dataset", "seed", "arm_id", "method", "final_val_loss", "final_val_ppl", "normalized_val_loss_auc"],
    )
    write_csv(
        RESULT_ROOT / "runtime.csv",
        runtime_rows,
        [
            "dataset", "seed", "arm_id", "total_seconds", "mean_seconds_per_step", "tokens_per_second",
            "training_loop_tokens_per_second", "peak_allocated_bytes", "peak_reserved_bytes",
            "grad_clip_trigger_fraction", "slurm_job_id", "slurm_restart_count", "slurm_node", "timing_attempt_id",
        ],
    )
    write_csv(
        RESULT_ROOT / "mechanism_checks.csv",
        mechanism_rows,
        ["dataset", "seed", "arm_id", "step", "telemetry_json"],
    )

    paired = []
    for dataset in TARGETS:
        for seed in ("2479", "5052", "8913"):
            full = by_key.get((dataset, seed, "A3"))
            if full is None or full["normalized_val_loss_auc"] is None:
                continue
            for arm in [f"A{index}" for index in range(10) if index != 3]:
                other = by_key.get((dataset, seed, arm))
                if other is None or other["normalized_val_loss_auc"] is None:
                    continue
                paired.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "comparison_arm": arm,
                        "full_auc": full["normalized_val_loss_auc"],
                        "comparison_auc": other["normalized_val_loss_auc"],
                        "auc_penalty_vs_full": float(other["normalized_val_loss_auc"]) - float(full["normalized_val_loss_auc"]),
                    }
                )
    write_csv(
        RESULT_ROOT / "paired_effects.csv",
        paired,
        ["dataset", "seed", "comparison_arm", "full_auc", "comparison_auc", "auc_penalty_vs_full"],
    )

    target_rows = []
    for run in complete:
        row = run["manifest"]
        for target in TARGETS[row["dataset"]]:
            hit = next((record for record in run["evals"] if float(record["val_loss"]) <= target), None)
            target_rows.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "arm_id": row["arm_id"],
                    "target_loss": target,
                    "hit": hit is not None,
                    "hit_step": "" if hit is None else hit["step"],
                    "hit_tokens": "" if hit is None else int(hit["step"]) * TOKENS_PER_STEP,
                    "active_seconds_at_val_loss": "" if hit is None else hit.get("active_seconds_at_val_loss", ""),
                }
            )
    write_csv(
        RESULT_ROOT / "target_arrival.csv",
        target_rows,
        ["dataset", "seed", "arm_id", "target_loss", "hit", "hit_step", "hit_tokens", "active_seconds_at_val_loss"],
    )

    counts = Counter(row["status"] for row in coverage)
    print(f"E9 scientific coverage: {dict(counts)}; {RESULT_ROOT}")
    return 0 if counts.get("complete", 0) == len(manifest_rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="validate the ten 80-step engineering rows")
    args = parser.parse_args()
    return run_preflight_validation() if args.preflight else run_scientific_analysis()


if __name__ == "__main__":
    raise SystemExit(main())
