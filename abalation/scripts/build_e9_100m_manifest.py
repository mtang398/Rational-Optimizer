#!/usr/bin/env python3
"""Build the frozen E9 scientific and engineering-preflight manifests."""

from __future__ import annotations

import csv
import random
import shlex
from pathlib import Path


SOURCE = Path("experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv")
SCIENTIFIC_DEST = Path("abalation/manifests/e9_100m_manifest.csv")
PREFLIGHT_DEST = Path("abalation/manifests/e9_preflight_manifest.csv")

DATASETS = ["dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"]
SEEDS = [2479, 5052, 8913]
DESIGN_SEED = 20260710
SCIENTIFIC_STEPS = 3050
PREFLIGHT_STEPS = 80
TOKENS_PER_STEP = 32768


FULL_EXTRA = (
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

NO_GROUP = (
    "--rational-matrix-policy-group-gain-strength 0.0 "
    "--rational-matrix-policy-group-pressure-strength 0.0 "
    "--rational-matrix-policy-group-activity-damping 0.0"
)
NO_ROLE_DEPTH = (
    "--rational-matrix-policy-adam-role-strength 0.0 "
    "--rational-matrix-policy-adam-role-strength-final 0.0 "
    "--rational-matrix-policy-input-depth-gain 0.0 "
    "--rational-matrix-policy-output-depth-gain 0.0"
)
ZERO_MUON_SCHEDULE = (
    "--rational-matrix-policy-muon-strength 0.0 "
    "--rational-matrix-policy-muon-lr-scale 0.0 "
    "--rational-matrix-policy-final-muon 0.0 "
    "--rational-matrix-policy-min-muon 0.0 "
    "--rational-matrix-policy-max-muon 0.0"
)
NO_PAIR_RESCALE = "--rlb-gauge-strength 0.0"
SUPPRESS_MUON_BRANCH = "--no-rational-matrix-policy-apply-muon-update"


ARMS = {
    "A0": {
        "label": "SiLU + AdamW",
        "method": "silu_adamw",
        "activation": "silu",
        "optimizer": "adamw",
        "extra": "",
    },
    "A1": {
        "label": "RLB + AdamW",
        "method": "rlb_adamw",
        "activation": "rlb_fused_global_rational",
        "optimizer": "adamw",
        "extra": "",
    },
    "A2": {
        "label": "RLB + static MatrixPolicy optimizer-recipe shell",
        "method": "rlb_matrixpolicy_static_shell",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_GROUP} {NO_ROLE_DEPTH} {ZERO_MUON_SCHEDULE} {NO_PAIR_RESCALE}",
    },
    "A3": {
        "label": "RLB + MatrixPolicy",
        "method": "rlb_matrixpolicy_full",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": FULL_EXTRA,
    },
    "A4": {
        "label": "MatrixPolicy without group-stat gradient gating",
        "method": "rlb_matrixpolicy_no_group_gate",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_GROUP}",
    },
    "A5": {
        "label": "MatrixPolicy without role/depth factors",
        "method": "rlb_matrixpolicy_no_role_depth",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_ROLE_DEPTH}",
    },
    "A6": {
        "label": "MatrixPolicy with transient Muon branch suppressed at fixed schedule",
        "method": "rlb_matrixpolicy_muon_branch_suppressed",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {SUPPRESS_MUON_BRANCH}",
    },
    "A7": {
        "label": "MatrixPolicy without role/depth factors and with Muon branch suppressed",
        "method": "rlb_matrixpolicy_no_role_depth_muon_branch_suppressed",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_ROLE_DEPTH} {SUPPRESS_MUON_BRANCH}",
    },
    "A8": {
        "label": "MatrixPolicy without reciprocal pair rescaling",
        "method": "rlb_matrixpolicy_no_pair_rescale",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_PAIR_RESCALE}",
    },
    "A9": {
        "label": "MatrixPolicy without role/depth-Muon action block",
        "method": "rlb_matrixpolicy_no_role_depth_muon_block",
        "activation": "rlb_fused_global_rational",
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra": f"{FULL_EXTRA} {NO_ROLE_DEPTH} {ZERO_MUON_SCHEDULE}",
    },
}


def effective_options(extra_args: str) -> dict[str, str | bool]:
    """Return last-wins long-option values for manifest semantic checks."""
    tokens = shlex.split(extra_args)
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise RuntimeError(f"unexpected positional manifest token: {token}")
        if token.startswith("--no-"):
            options[token] = True
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token] = tokens[index + 1]
            index += 2
        else:
            options[token] = True
            index += 1
    return options


def load_templates() -> tuple[dict[str, dict[str, str]], list[str]]:
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    selected = [row for row in rows if row["phase"] == "E1_rational_only_100m"]
    templates: dict[str, dict[str, str]] = {}
    for dataset in DATASETS:
        matches = [row for row in selected if row["dataset"] == dataset]
        if len(matches) != 3:
            raise RuntimeError(f"expected three E1 source rows for {dataset}, found {len(matches)}")
        template = dict(matches[0])
        if template["extra_args"] != FULL_EXTRA:
            raise RuntimeError(f"source A3 flags changed for {dataset}: {template['extra_args']!r}")
        templates[dataset] = template
    return templates, fieldnames


def balanced_block_orders() -> list[list[str]]:
    """Create 15 randomized cyclic orders with position counts differing by at most one."""
    rng = random.Random(DESIGN_SEED)
    base = list(ARMS)
    rng.shuffle(base)
    rotations = list(range(len(base)))
    rng.shuffle(rotations)
    rotations.extend(rng.sample(range(len(base)), 5))
    orders = []
    for rotation in rotations:
        order = base[rotation:] + base[:rotation]
        orders.append(order)
    return orders


def make_row(
    template: dict[str, str],
    *,
    arm_id: str,
    seed: int,
    phase: str,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    row_index: int,
    block_index: int,
    block_position: int,
) -> dict[str, str]:
    spec = ARMS[arm_id]
    preflight_phase = phase == "E9_preflight_80step"
    row = dict(template)
    source_row_index = row["row_index"]
    row.update(
        {
            "row_index": str(row_index),
            "phase": phase,
            "seed": str(seed),
            "method": str(spec["method"]),
            "activation": str(spec["activation"]),
            "optimizer": str(spec["optimizer"]),
            "extra_args": str(spec["extra"]),
            "steps": str(steps),
            "train_tokens": str(2_621_440 if preflight_phase else 100_000_000),
            "eval_interval": str(eval_interval),
            "eval_batches": str(eval_batches),
            "arm_id": arm_id,
            "arm_label": str(spec["label"]),
            "block_index": str(block_index),
            "block_position": str(block_position),
            "exact_train_tokens": str(steps * TOKENS_PER_STEP),
            "design_version": "E9-frozen-2026-07-11",
            "run_kind": "preflight" if preflight_phase else "scientific",
            "source_row_index": source_row_index,
            "randomization_seed": str(DESIGN_SEED),
        }
    )
    if preflight_phase:
        row["val_tokens"] = "200000"
    row["row_id"] = (
        f"{phase}_{row['dataset']}_m0_{steps * TOKENS_PER_STEP}_"
        f"seed{seed}_{arm_id.lower()}_{spec['method']}"
    )
    return row


def validate_rows(rows: list[dict[str, str]], *, preflight: bool) -> None:
    expected = 10 if preflight else 150
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("generated duplicate row IDs")
    if [int(row["row_index"]) for row in rows] != list(range(expected)):
        raise RuntimeError("generated row indices are not contiguous")
    for row in rows:
        arm = row["arm_id"]
        options = effective_options(row["extra_args"])
        if int(row["exact_train_tokens"]) != int(row["steps"]) * TOKENS_PER_STEP:
            raise RuntimeError(f"non-exact token exposure in {row['row_id']}")
        if arm in {"A0", "A1"}:
            if row["optimizer"] != "adamw" or options:
                raise RuntimeError(f"invalid ordinary-AdamW arm {row['row_id']}")
            continue
        if options.get("--rational-matrix-policy-adam-lr-scale") != "3.0":
            raise RuntimeError(f"missing fixed 3x matrix Adam scale in {row['row_id']}")
        suppressed = bool(options.get("--no-rational-matrix-policy-apply-muon-update", False))
        if suppressed != (arm in {"A6", "A7"}):
            raise RuntimeError(f"wrong applied-Muon suppression in {row['row_id']}")
        muon_zero = options.get("--rational-matrix-policy-max-muon") == "0.0"
        if muon_zero != (arm in {"A2", "A9"}):
            raise RuntimeError(f"wrong Muon schedule intervention in {row['row_id']}")
        pair_off = options.get("--rlb-gauge-strength") == "0.0"
        if pair_off != (arm in {"A2", "A8"}):
            raise RuntimeError(f"wrong pair-rescale intervention in {row['row_id']}")
        group_off = options.get("--rational-matrix-policy-group-gain-strength") == "0.0"
        if group_off != (arm in {"A2", "A4"}):
            raise RuntimeError(f"wrong group-gate intervention in {row['row_id']}")
        role_values = (
            options.get("--rational-matrix-policy-adam-role-strength"),
            options.get("--rational-matrix-policy-adam-role-strength-final"),
            options.get("--rational-matrix-policy-input-depth-gain"),
            options.get("--rational-matrix-policy-output-depth-gain"),
        )
        role_off = role_values == ("0.0", "0.0", "0.0", "0.0")
        if role_off != (arm in {"A2", "A5", "A7", "A9"}):
            raise RuntimeError(f"wrong role/depth intervention in {row['row_id']}")
        if arm == "A6" and set(options) - set(effective_options(FULL_EXTRA)) != {
            "--no-rational-matrix-policy-apply-muon-update"
        }:
            raise RuntimeError(f"A6 contains an intervention beyond Muon branch suppression: {row['row_id']}")
        if arm == "A7":
            expected_extra = {
                "--rational-matrix-policy-adam-role-strength",
                "--rational-matrix-policy-adam-role-strength-final",
                "--rational-matrix-policy-input-depth-gain",
                "--rational-matrix-policy-output-depth-gain",
                "--no-rational-matrix-policy-apply-muon-update",
            }
            if set(options) - set(effective_options(FULL_EXTRA)) != expected_extra:
                raise RuntimeError(f"A7 intervention set changed: {row['row_id']}")
    if preflight:
        if {row["arm_id"] for row in rows} != set(ARMS):
            raise RuntimeError("preflight does not contain exactly A0-A9")
        if {int(row["block_position"]) for row in rows} != set(range(10)):
            raise RuntimeError("preflight positions are incomplete")
        if any(int(row["train_tokens"]) != 2_621_440 or int(row["val_tokens"]) != 200_000 for row in rows):
            raise RuntimeError("preflight cache pools changed")
        return

    blocks: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        blocks.setdefault(int(row["block_index"]), []).append(row)
    if set(blocks) != set(range(15)):
        raise RuntimeError("scientific block indices are incomplete")
    for block_index, block in blocks.items():
        if len(block) != 10 or {row["arm_id"] for row in block} != set(ARMS):
            raise RuntimeError(f"block {block_index} does not contain exactly A0-A9")
        if {int(row["block_position"]) for row in block} != set(range(10)):
            raise RuntimeError(f"block {block_index} positions are incomplete")
        if len({(row["dataset"], row["seed"]) for row in block}) != 1:
            raise RuntimeError(f"block {block_index} mixes dataset-seed cells")
    position_counts = {
        (arm, position): sum(
            row["arm_id"] == arm and int(row["block_position"]) == position for row in rows
        )
        for arm in ARMS
        for position in range(10)
    }
    if set(position_counts.values()) - {1, 2}:
        raise RuntimeError("arm-position balance is outside the predeclared 1-2 range")
    if any(int(row["train_tokens"]) != 100_000_000 or int(row["val_tokens"]) != 4_000_000 for row in rows):
        raise RuntimeError("scientific cache pools changed")


def write_manifest(path: Path, rows: list[dict[str, str]], source_fields: list[str]) -> None:
    extra_fields = [
        "arm_id",
        "arm_label",
        "block_index",
        "block_position",
        "exact_train_tokens",
        "design_version",
        "run_kind",
        "source_row_index",
        "randomization_seed",
    ]
    fieldnames = source_fields + [field for field in extra_fields if field not in source_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    templates, source_fields = load_templates()
    orders = balanced_block_orders()

    scientific: list[dict[str, str]] = []
    block_index = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            for position, arm_id in enumerate(orders[block_index]):
                scientific.append(
                    make_row(
                        templates[dataset],
                        arm_id=arm_id,
                        seed=seed,
                        phase="E9_100m",
                        steps=SCIENTIFIC_STEPS,
                        eval_interval=50,
                        eval_batches=10,
                        row_index=len(scientific),
                        block_index=block_index,
                        block_position=position,
                    )
                )
            block_index += 1

    preflight: list[dict[str, str]] = []
    for position, arm_id in enumerate(orders[0]):
        preflight.append(
            make_row(
                templates["dclm"],
                arm_id=arm_id,
                seed=SEEDS[0],
                phase="E9_preflight_80step",
                steps=PREFLIGHT_STEPS,
                eval_interval=40,
                eval_batches=2,
                row_index=len(preflight),
                block_index=0,
                block_position=position,
            )
        )

    validate_rows(scientific, preflight=False)
    validate_rows(preflight, preflight=True)
    write_manifest(SCIENTIFIC_DEST, scientific, source_fields)
    write_manifest(PREFLIGHT_DEST, preflight, source_fields)
    print(f"wrote {SCIENTIFIC_DEST} with {len(scientific)} rows")
    print(f"wrote {PREFLIGHT_DEST} with {len(preflight)} rows")


if __name__ == "__main__":
    main()
