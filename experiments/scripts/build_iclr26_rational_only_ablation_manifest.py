#!/usr/bin/env python3
"""Build the ICLR26 global-rational RLB ablation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_iclr26_main_manifest import FIELDNAMES, MAIN_DATASETS, SEEDS, add_row, write_manifest

RATIONAL_ONLY_ACTIVATION = "rlb_fused_global_rational"
MATRIX_POLICY_ARGS = (
    "--rational-matrix-policy-backbone-optimizer adamw "
    "--rational-matrix-policy-adam-lr-scale 3.0 "
    "--rational-matrix-policy-group-gain-strength 0.20 "
    "--rational-matrix-policy-group-pressure-strength 0.10 "
    "--rational-matrix-policy-group-activity-damping 0.20 "
    "--rational-matrix-policy-group-start 0.02 "
    "--rational-matrix-policy-group-end 0.30 "
    "--rational-matrix-policy-group-min-scale 0.75 "
    "--rational-matrix-policy-group-max-scale 1.35"
)
RATIONAL_ONLY_MATRIX_POLICY = {
    "method": "rlb_rational_only_matrixpolicy",
    "activation": RATIONAL_ONLY_ACTIVATION,
    "optimizer": "rational_matrix_policy_onpolicy",
    "lr": "0.0003",
    "min_lr": "0.00003",
    "weight_decay": "0.10",
    "extra_args": MATRIX_POLICY_ARGS,
}


PHASES = [
    ("E1_rational_only_100m", 100_000_000),
    ("E2_rational_only_300m", 300_000_000),
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for phase, train_tokens in PHASES:
        for dataset in MAIN_DATASETS:
            for seed in SEEDS:
                add_row(
                    rows,
                    phase=phase,
                    dataset=dataset,
                    model_name="M0",
                    train_tokens=train_tokens,
                    seed=seed,
                    method=RATIONAL_ONLY_MATRIX_POLICY,
                )
    for idx, row in enumerate(rows):
        row["row_index"] = str(idx)
    return rows


def verify(rows: list[dict[str, str]]) -> None:
    expected = len(PHASES) * len(MAIN_DATASETS) * len(SEEDS)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    for row in rows:
        missing = [field for field in FIELDNAMES if field not in row]
        if missing:
            raise SystemExit(f"missing fields in {row.get('row_id', '<unknown>')}: {missing}")
        if row["activation"] != RATIONAL_ONLY_ACTIVATION:
            raise SystemExit(f"wrong activation in {row['row_id']}: {row['activation']}")
        if row["optimizer"] != "rational_matrix_policy_onpolicy":
            raise SystemExit(f"wrong optimizer in {row['row_id']}: {row['optimizer']}")
        if row["method"] != "rlb_rational_only_matrixpolicy":
            raise SystemExit(f"wrong method in {row['row_id']}: {row['method']}")


def print_summary(rows: list[dict[str, str]]) -> None:
    print("rows", len(rows))
    for phase, _ in PHASES:
        count = sum(1 for row in rows if row["phase"] == phase)
        print(f"{phase}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/manifests/iclr26_rational_only_ablation_manifest.csv"))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    write_manifest(rows, args.output)
    if args.print_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
