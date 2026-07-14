#!/usr/bin/env python3
"""Build MatrixPolicy role/depth V5 and V6 ablation manifests."""

from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path("experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv")
DEST = Path("abalation/manifests/role_depth_v5_v6_e1_e2_manifest.csv")


VARIANTS = {
    "role_depth_v5": {
        "method": "rlb_matrixpolicy_role_depth_v5",
        "extra": (
            "--rational-matrix-policy-adam-role-strength-final 0.40 "
            "--rational-matrix-policy-adam-role-decay-start 0.20 "
            "--rational-matrix-policy-adam-role-decay-end 0.36 "
            "--rational-matrix-policy-muon-strength 0.65 "
            "--rational-matrix-policy-max-muon 0.65 "
            "--rational-matrix-policy-decay-start 0.18 "
            "--rational-matrix-policy-decay-end 0.32"
        ),
        "phases": {
            "E1_rational_only_100m": "E1_matrixpolicy_ablation_role_depth_v5_100m",
            "E2_rational_only_300m": "E2_matrixpolicy_ablation_role_depth_v5_300m",
        },
    },
    "role_depth_v6": {
        "method": "rlb_matrixpolicy_role_depth_v6",
        "extra": (
            "--rational-matrix-policy-adam-role-strength-final 0.40 "
            "--rational-matrix-policy-adam-role-decay-start 0.20 "
            "--rational-matrix-policy-adam-role-decay-end 0.36 "
            "--rational-matrix-policy-muon-strength 0.50 "
            "--rational-matrix-policy-max-muon 0.50 "
            "--rational-matrix-policy-decay-start 0.16 "
            "--rational-matrix-policy-decay-end 0.28 "
            "--rational-matrix-policy-adam-stat-strength 0.10 "
            "--rational-matrix-policy-adam-pressure-balance 0.05 "
            "--rational-matrix-policy-adam-stat-start 0.16 "
            "--rational-matrix-policy-adam-stat-end 0.34"
        ),
        "phases": {
            "E1_rational_only_100m": "E1_matrixpolicy_ablation_role_depth_v6_100m",
            "E2_rational_only_300m": "E2_matrixpolicy_ablation_role_depth_v6_300m",
        },
    },
}


def make_variant(row: dict[str, str], variant: str, row_index: int) -> dict[str, str]:
    spec = VARIANTS[variant]
    out = dict(row)
    source_phase = row["phase"]
    phase = spec["phases"][source_phase]
    method = spec["method"]
    out["row_index"] = str(row_index)
    out["phase"] = phase
    out["method"] = method
    out["extra_args"] = f"{row['extra_args']} {spec['extra']}".strip()
    out["row_id"] = row["row_id"].replace("rlb_rational_only_matrixpolicy", method)
    out["row_id"] = out["row_id"].replace(source_phase, phase)
    return out


def main() -> None:
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise RuntimeError(f"manifest has no header: {SOURCE}")
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 source rows, found {len(rows)}")

    output_rows: list[dict[str, str]] = []
    for variant in ("role_depth_v5", "role_depth_v6"):
        phase_map = VARIANTS[variant]["phases"]
        selected = [row for row in rows if row["phase"] in phase_map]
        if len(selected) != 30:
            raise RuntimeError(f"expected 30 source rows for {variant}, found {len(selected)}")
        for source_row in selected:
            output_rows.append(make_variant(source_row, variant, len(output_rows)))

    row_ids = [row["row_id"] for row in output_rows]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("generated duplicate row ids")

    DEST.parent.mkdir(parents=True, exist_ok=True)
    with DEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {DEST} with {len(output_rows)} rows")


if __name__ == "__main__":
    main()
