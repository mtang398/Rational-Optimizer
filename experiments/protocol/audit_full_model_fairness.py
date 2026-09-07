#!/usr/bin/env python3
"""Construct each published suite cell and audit optimizer LR/WD scalars."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for search_path in (ROOT / "activation", ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from training import train as trainer

from . import suite


def parsed(row, arm):
    trainer.ACTIVE_OPTIMIZERS = sorted(
        set(trainer.ACTIVE_OPTIMIZERS) | {suite.EXACT_OPTIMIZER_KEY}
    )
    old = sys.argv
    try:
        sys.argv = suite.training_argv(row, arm, Path("/tmp/full-method-fairness"))
        return trainer.parse_args()
    finally:
        sys.argv = old


def audit_cell(row):
    candidate = importlib.import_module(suite.CANDIDATE_MODULE)
    reports = []
    for arm in ("control", "candidate"):
        args = parsed(row, arm)
        suite.install_identity(row, arm)
        suite.install_exact_weight_decay_policy(row, candidate if arm == "candidate" else None)
        suite.install_fixed_probe_layout(row)
        model = trainer.CausalTransformer(args, 50_257)
        trainer.apply_rlb_positive_gauge(
            model,
            args.rlb_init_gauge_log_scale,
            args.rlb_init_gauge_seed,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        identity = trainer.audit_tiller_experiment_identity(
            args,
            world_size=4,
            global_tokens=row["global_tokens_per_step"],
            train_token_count=row["max_train_tokens"],
            val_token_count=row["max_val_tokens"],
            parameter_count=parameter_count,
        )
        optimizer = (
            trainer.configure_optimizer(model, args)
            if arm == "control"
            else candidate.configure_candidate_optimizer(model, args)
        )
        fairness = trainer.audit_optimizer_lr_wd_fairness(model, optimizer, args)
        reports.append({
            "suite": row["phase"],
            "model": row["model"],
            "arm": arm,
            "optimizer": args.optimizer,
            "parameter_count": parameter_count,
            "identity": identity,
            "base_lr": fairness["base_lr"],
            "minimum_lr": fairness["minimum_lr"],
            "base_weight_decay": fairness["base_weight_decay"],
            "all_group_lr_scales_are_one": all(
                group["lr_scale"] == 1.0 for group in fairness["groups"]
            ),
            "all_internal_lr_wd_scalars_are_one": all(
                value == 1.0
                for value in fairness["internal_lr_wd_scalars"].values()
            ),
            "covered_parameter_elements": fairness["covered_parameter_elements"],
            "trainable_parameter_elements": fairness["trainable_parameter_elements"],
        })
        candidate._ACTIVE_ROUTER = None
        del optimizer, model
        gc.collect()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = {
        row["phase"]: row for row in suite.matrix_payload()["rows"]
    }
    reports = []
    for phase in (
        "12l_768d_100m_tokens_3050_steps",
        "12l_768d_300m_tokens_9150_steps",
        "18l_1024d_300m_tokens_9150_steps",
    ):
        reports.extend(audit_cell(selected[phase]))
    payload = {
        "schema": "tiller_full_model_fairness_v1",
        "passed": all(
            report["base_lr"] == 3e-4
            and report["minimum_lr"] == 3e-5
            and report["base_weight_decay"] == 0.1
            and report["all_group_lr_scales_are_one"]
            and report["all_internal_lr_wd_scalars_are_one"]
            and report["covered_parameter_elements"]
            == report["trainable_parameter_elements"]
            for report in reports
        ),
        "reports": reports,
    }
    if not payload["passed"]:
        raise RuntimeError(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
