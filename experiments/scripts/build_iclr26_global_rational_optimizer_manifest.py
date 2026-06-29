#!/usr/bin/env python3
"""Build the ICLR26 global-rational/no-local-atom RLB optimizer-control manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_iclr26_main_manifest import FIELDNAMES, MAIN_DATASETS, METHODS, SEEDS, add_row, write_manifest

GLOBAL_RATIONAL_ACTIVATION = "rlb_fused_global_rational"
PHASES = [
    ("E1_global_rational_optimizers_100m", 100_000_000),
    ("E2_global_rational_optimizers_300m", 300_000_000),
]
EXCLUDED_METHODS = {"rlb_matrixpolicy_original"}


def global_rational_methods() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for method in METHODS:
        name = method["method"]
        if not name.startswith("rlb_") or name in EXCLUDED_METHODS:
            continue
        item = dict(method)
        item["activation"] = GLOBAL_RATIONAL_ACTIVATION
        out.append(item)
    return out


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    methods = global_rational_methods()
    for phase, train_tokens in PHASES:
        for dataset in MAIN_DATASETS:
            for seed in SEEDS:
                for method in methods:
                    add_row(
                        rows,
                        phase=phase,
                        dataset=dataset,
                        model_name="M0",
                        train_tokens=train_tokens,
                        seed=seed,
                        method=method,
                    )
    for idx, row in enumerate(rows):
        row["row_index"] = str(idx)
    return rows


def verify(rows: list[dict[str, str]]) -> None:
    methods = global_rational_methods()
    expected = len(PHASES) * len(MAIN_DATASETS) * len(SEEDS) * len(methods)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} rows, found {len(rows)}")
    expected_methods = {method["method"] for method in methods}
    for row in rows:
        missing = [field for field in FIELDNAMES if field not in row]
        if missing:
            raise SystemExit(f"missing fields in {row.get('row_id', '<unknown>')}: {missing}")
        if row["method"] not in expected_methods:
            raise SystemExit(f"unexpected method in {row['row_id']}: {row['method']}")
        if row["activation"] != GLOBAL_RATIONAL_ACTIVATION:
            raise SystemExit(f"wrong activation in {row['row_id']}: {row['activation']}")
        if row["optimizer"] == "rational_matrix_policy_onpolicy":
            raise SystemExit(f"MatrixPolicy belongs to the completed replacement manifest, not this optimizer-control manifest: {row['row_id']}")


def print_summary(rows: list[dict[str, str]]) -> None:
    print("rows", len(rows))
    for phase, _ in PHASES:
        count = sum(1 for row in rows if row["phase"] == phase)
        print(f"{phase}: {count}")
    methods = sorted({row["method"] for row in rows})
    print("methods", ", ".join(methods))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/manifests/iclr26_global_rational_optimizer_controls_manifest.csv"))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    write_manifest(rows, args.output)
    if args.print_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
