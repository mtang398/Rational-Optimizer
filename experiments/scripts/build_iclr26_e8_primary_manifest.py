#!/usr/bin/env python3
"""Build the five-dataset primary E8 paired LR/WD sensitivity manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

RLB = "rlb_fused_global_rational"

DATASETS = {
    "dclm": {
        "dataset_name": "mlfoundations/dclm-baseline-1.0",
        "dataset_config": "none",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_tokens": 210_000_000,
    },
    "fineweb_edu": {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_tokens": 210_000_000,
    },
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_config": "sample-10BT",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_tokens": 210_000_000,
    },
    "dolma_sample": {
        "dataset_name": "allenai/dolma",
        "dataset_config": "v1_6-sample",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_tokens": 210_000_000,
    },
    "c4_en": {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "text_column": "text",
        "train_split": "train",
        "val_split": "validation",
        "val_skip_tokens": 0,
    },
}

MODEL = {
    "name": "M0",
    "layers": 12,
    "d_model": 768,
    "heads": 12,
    "ffn_dim": 2048,
    "seq_len": 256,
    "batch_size": 16,
    "grad_accum": 2,
}

METHODS = [
    {
        "method": "silu_adamw",
        "activation": "silu",
        "optimizer": "adamw",
        "extra_args": "",
    },
    {
        "method": "silu_muon",
        "activation": "silu",
        "optimizer": "muon",
        "extra_args": "--muon-momentum 0.95 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw",
    },
    {
        "method": "rlb_matrixpolicy_original",
        "activation": RLB,
        "optimizer": "rational_matrix_policy_onpolicy",
        "extra_args": "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-adam-lr-scale 3.0 --rational-matrix-policy-group-gain-strength 0.20 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35",
    },
]

LR_GRID = [
    ("0.0001", "0.00001", "lr1em4"),
    ("0.0002", "0.00002", "lr2em4"),
    ("0.0003", "0.00003", "lr3em4"),
    ("0.0005", "0.00005", "lr5em4"),
]
WD_GRID = [
    ("0.00", "wd0p00"),
    ("0.05", "wd0p05"),
    ("0.10", "wd0p10"),
    ("0.20", "wd0p20"),
]

PHASE = "E8_primary_100m"
SEED = 1337
TRAIN_TOKENS = 100_000_000
VAL_TOKENS = 4_000_000
STEPS = 3_050
EVAL_INTERVAL = 50
EVAL_BATCHES = 10

FIELDNAMES = [
    "row_index",
    "row_id",
    "phase",
    "dataset",
    "dataset_name",
    "dataset_config",
    "text_column",
    "train_split",
    "val_split",
    "val_skip_tokens",
    "model",
    "layers",
    "d_model",
    "heads",
    "ffn_dim",
    "seq_len",
    "batch_size",
    "grad_accum",
    "global_tokens_per_step",
    "train_tokens",
    "val_tokens",
    "steps",
    "eval_interval",
    "eval_batches",
    "seed",
    "method",
    "activation",
    "optimizer",
    "lr",
    "min_lr",
    "weight_decay",
    "extra_args",
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    global_tokens = MODEL["seq_len"] * 4 * MODEL["batch_size"] * MODEL["grad_accum"]
    for dataset, ds in DATASETS.items():
        for lr, min_lr, lr_label in LR_GRID:
            for weight_decay, wd_label in WD_GRID:
                for method in METHODS:
                    row_id = f"{PHASE}_{dataset}_m0_100m_seed{SEED}_{method['method']}_{lr_label}_{wd_label}"
                    rows.append(
                        {
                            "row_index": str(len(rows)),
                            "row_id": row_id,
                            "phase": PHASE,
                            "dataset": dataset,
                            "dataset_name": ds["dataset_name"],
                            "dataset_config": ds["dataset_config"],
                            "text_column": ds["text_column"],
                            "train_split": ds["train_split"],
                            "val_split": ds["val_split"],
                            "val_skip_tokens": str(ds["val_skip_tokens"]),
                            "model": MODEL["name"],
                            "layers": str(MODEL["layers"]),
                            "d_model": str(MODEL["d_model"]),
                            "heads": str(MODEL["heads"]),
                            "ffn_dim": str(MODEL["ffn_dim"]),
                            "seq_len": str(MODEL["seq_len"]),
                            "batch_size": str(MODEL["batch_size"]),
                            "grad_accum": str(MODEL["grad_accum"]),
                            "global_tokens_per_step": str(global_tokens),
                            "train_tokens": str(TRAIN_TOKENS),
                            "val_tokens": str(VAL_TOKENS),
                            "steps": str(STEPS),
                            "eval_interval": str(EVAL_INTERVAL),
                            "eval_batches": str(EVAL_BATCHES),
                            "seed": str(SEED),
                            "lr": lr,
                            "min_lr": min_lr,
                            "weight_decay": weight_decay,
                            **method,
                        }
                    )
    return rows


def verify(rows: list[dict[str, str]]) -> None:
    expected_methods = {row["method"] for row in METHODS}
    if len(rows) != len(DATASETS) * len(LR_GRID) * len(WD_GRID) * len(METHODS):
        raise SystemExit(f"unexpected row count: {len(rows)}")
    cells: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] != PHASE:
            raise SystemExit(f"bad phase in {row['row_id']}")
        if row["activation"].startswith("rlb_") and row["activation"] != RLB:
            raise SystemExit(f"bad RLB activation in {row['row_id']}")
        if int(row["eval_interval"]) > 50:
            raise SystemExit(f"eval interval too sparse in {row['row_id']}")
        if row["method"] == "rlb_adamw":
            raise SystemExit("rlb_adamw must not appear in reduced E8 primary")
        key = (row["dataset"], row["lr"], row["min_lr"], row["weight_decay"])
        cells[key].append(row)

    for key, cell_rows in cells.items():
        present = {row["method"] for row in cell_rows}
        if present != expected_methods:
            raise SystemExit(f"incomplete sensitivity cell {key}: methods={sorted(present)}")
        outer = {(row["lr"], row["min_lr"], row["weight_decay"]) for row in cell_rows}
        if len(outer) != 1:
            raise SystemExit(f"outer config mismatch in {key}: {sorted(outer)}")


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    print("rows", len(rows))
    for key, count in sorted(Counter(row["dataset"] for row in rows).items()):
        print(f"dataset/{key}: {count}")
    for key, count in sorted(Counter(row["method"] for row in rows).items()):
        print(f"method/{key}: {count}")
    for key, count in sorted(Counter((row["lr"], row["weight_decay"]) for row in rows).items()):
        print(f"grid/{key[0]}/wd={key[1]}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/manifests/iclr26_e8_primary_manifest.csv"))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    write_manifest(rows, args.output)
    if args.print_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
