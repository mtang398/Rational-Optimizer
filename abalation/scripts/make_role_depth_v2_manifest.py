#!/usr/bin/env python3
"""Build the MatrixPolicy role/depth V2 ablation manifest."""

from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path("experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv")
DEST = Path("abalation/manifests/role_depth_v2_e1_e2_manifest.csv")


ROLE_DEPTH_V2_EXTRA = (
    "--rational-matrix-policy-adam-role-strength-final 0.40 "
    "--rational-matrix-policy-adam-role-decay-start 0.24 "
    "--rational-matrix-policy-adam-role-decay-end 0.42"
)


PHASES = {
    "E1_rational_only_100m": "E1_matrixpolicy_ablation_role_depth_v2_100m",
    "E2_rational_only_300m": "E2_matrixpolicy_ablation_role_depth_v2_300m",
}


def make_variant(row: dict[str, str], row_index: int) -> dict[str, str]:
    out = dict(row)
    source_phase = row["phase"]
    phase = PHASES[source_phase]
    suffix = "rlb_matrixpolicy_role_depth_v2"
    out["row_index"] = str(row_index)
    out["phase"] = phase
    out["method"] = suffix
    out["extra_args"] = f"{row['extra_args']} {ROLE_DEPTH_V2_EXTRA}".strip()
    out["row_id"] = row["row_id"].replace("rlb_rational_only_matrixpolicy", suffix)
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

    selected = [row for row in rows if row["phase"] in PHASES]
    if len(selected) != 30:
        raise RuntimeError(f"expected 30 E1/E2 MatrixPolicy rows, found {len(selected)}")

    output_rows = [make_variant(row, index) for index, row in enumerate(selected)]
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
