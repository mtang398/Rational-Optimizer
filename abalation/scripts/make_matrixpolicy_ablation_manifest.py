#!/usr/bin/env python3
"""Build the MatrixPolicy ablation manifest without touching main results."""

from __future__ import annotations

import csv
from pathlib import Path


SOURCE = Path("experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv")
DEST = Path("abalation/manifests/matrixpolicy_ablation_e1_e2_manifest.csv")


NO_ROLE_DEPTH_EXTRA = (
    "--rational-matrix-policy-adam-role-strength 0.0 "
    "--rational-matrix-policy-input-depth-gain 0.0 "
    "--rational-matrix-policy-output-depth-gain 0.0 "
    "--rational-matrix-policy-muon-strength 0.0 "
    "--rational-matrix-policy-muon-lr-scale 0.0 "
    "--rational-matrix-policy-final-muon 0.0 "
    "--rational-matrix-policy-min-muon 0.0 "
    "--rational-matrix-policy-max-muon 0.0"
)


PHASES = {
    ("E1_rational_only_100m", "no_role_depth"): "E1_matrixpolicy_ablation_no_role_depth_100m",
    ("E2_rational_only_300m", "no_role_depth"): "E2_matrixpolicy_ablation_no_role_depth_300m",
    ("E1_rational_only_100m", "bypass_muon"): "E1_matrixpolicy_ablation_bypass_muon_100m",
    ("E2_rational_only_300m", "bypass_muon"): "E2_matrixpolicy_ablation_bypass_muon_300m",
}


def replace_backbone_optimizer(extra_args: str, optimizer: str) -> str:
    needle = "--rational-matrix-policy-backbone-optimizer adamw"
    replacement = f"--rational-matrix-policy-backbone-optimizer {optimizer}"
    if needle in extra_args:
        return extra_args.replace(needle, replacement)
    return f"{extra_args} {replacement}".strip()


def make_variant(row: dict[str, str], variant: str, row_index: int) -> dict[str, str]:
    out = dict(row)
    source_phase = row["phase"]
    out["row_index"] = str(row_index)
    out["phase"] = PHASES[(source_phase, variant)]

    if variant == "no_role_depth":
        suffix = "rlb_matrixpolicy_no_role_depth"
        out["method"] = "rlb_matrixpolicy_no_role_depth"
        out["extra_args"] = f"{row['extra_args']} {NO_ROLE_DEPTH_EXTRA}".strip()
    elif variant == "bypass_muon":
        suffix = "rlb_matrixpolicy_bypass_muon"
        out["method"] = "rlb_matrixpolicy_bypass_muon"
        out["extra_args"] = replace_backbone_optimizer(row["extra_args"], "muon")
    else:
        raise ValueError(f"unknown variant: {variant}")

    out["row_id"] = row["row_id"].replace("rlb_rational_only_matrixpolicy", suffix)
    out["row_id"] = out["row_id"].replace(source_phase, out["phase"])
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

    e1 = [row for row in rows if row["phase"] == "E1_rational_only_100m"]
    e2 = [row for row in rows if row["phase"] == "E2_rational_only_300m"]
    if len(e1) != 15 or len(e2) != 15:
        raise RuntimeError(f"expected 15 E1 and 15 E2 rows, found {len(e1)} and {len(e2)}")

    output_rows: list[dict[str, str]] = []
    ordered_blocks = [
        ("no_role_depth", e1),
        ("bypass_muon", e1),
        ("no_role_depth", e2),
        ("bypass_muon", e2),
    ]
    for variant, block_rows in ordered_blocks:
        for source_row in block_rows:
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
