#!/usr/bin/env python3
"""Build the retained ICLR 2026 baseline experiment manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

RLB = "rlb_fused_fixed_strong_ffn"

DATASETS = {
    "dclm": {
        "dataset_name": "mlfoundations/dclm-baseline-1.0",
        "dataset_config": "none",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "fineweb_edu": {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_config": "sample-10BT",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "dolma_sample": {
        "dataset_name": "allenai/dolma",
        "dataset_config": "v1_6-sample",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "c4_en": {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "text_column": "text",
        "train_split": "train",
        "val_split": "validation",
        "val_skip_100m": 0,
        "val_skip_300m": 0,
    },
}

MODELS = {
    "M0": {"layers": 12, "d_model": 768, "heads": 12, "ffn_dim": 2048, "batch_size": 16, "grad_accum": 2},
    "M1": {"layers": 18, "d_model": 1024, "heads": 16, "ffn_dim": 3072, "batch_size": 8, "grad_accum": 4},
}

METHODS = [
    {"method": "silu_adamw", "activation": "silu", "optimizer": "adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": ""},
    {"method": "rlb_adamw", "activation": RLB, "optimizer": "adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": ""},
    {"method": "silu_muon", "activation": "silu", "optimizer": "muon", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--muon-momentum 0.95 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw"},
    {"method": "rlb_muon", "activation": RLB, "optimizer": "muon", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--muon-momentum 0.95 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw"},
    {"method": "silu_lion", "activation": "silu", "optimizer": "lion", "lr": "0.0001", "min_lr": "0.00001", "weight_decay": "0.10", "extra_args": ""},
    {"method": "rlb_lion", "activation": RLB, "optimizer": "lion", "lr": "0.0001", "min_lr": "0.00001", "weight_decay": "0.10", "extra_args": ""},
    {"method": "silu_soap", "activation": "silu", "optimizer": "soap_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--soap-precondition-frequency 50 --no-soap-one-sided"},
    {"method": "rlb_soap", "activation": RLB, "optimizer": "soap_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--soap-precondition-frequency 50 --no-soap-one-sided"},
    {"method": "silu_ademamix", "activation": "silu", "optimizer": "ademamix", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--ademamix-alpha 5.0 --ademamix-beta3 0.9999"},
    {"method": "rlb_ademamix", "activation": RLB, "optimizer": "ademamix", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--ademamix-alpha 5.0 --ademamix-beta3 0.9999"},
    {"method": "silu_came", "activation": "silu", "optimizer": "adafactor_came", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--came-confidence-scale 1.0"},
    {"method": "rlb_came", "activation": RLB, "optimizer": "adafactor_came", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--came-confidence-scale 1.0"},
    {"method": "silu_schedulefree", "activation": "silu", "optimizer": "schedule_free_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--schedule-free-beta1 0.90 --schedule-free-warmup-steps 0"},
    {"method": "rlb_schedulefree", "activation": RLB, "optimizer": "schedule_free_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--schedule-free-beta1 0.90 --schedule-free-warmup-steps 0"},
]

SEEDS = [1337, 2027, 3407]
MAIN_DATASETS = ["dclm", "fineweb_edu", "fineweb", "dolma_sample", "c4_en"]

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


def budget_spec(tokens: int) -> tuple[int, int, int, str]:
    if tokens == 100_000_000:
        return 3_050, 4_000_000, 50, "100m"
    if tokens == 300_000_000:
        return 9_150, 8_000_000, 50, "300m"
    raise ValueError(tokens)


def add_row(rows: list[dict[str, str]], *, phase: str, dataset: str, model_name: str, train_tokens: int, seed: int, method: dict[str, str]) -> None:
    steps, val_tokens, eval_interval, budget_tag = budget_spec(train_tokens)
    ds = DATASETS[dataset]
    model = MODELS[model_name]
    skip_key = "val_skip_300m" if train_tokens == 300_000_000 else "val_skip_100m"
    global_tokens = 256 * 4 * model["batch_size"] * model["grad_accum"]
    row_id = f"{phase}_{dataset}_{model_name.lower()}_{budget_tag}_seed{seed}_{method['method']}"
    row = {
        "row_index": str(len(rows)),
        "row_id": row_id,
        "phase": phase,
        "dataset": dataset,
        "dataset_name": ds["dataset_name"],
        "dataset_config": ds["dataset_config"],
        "text_column": ds["text_column"],
        "train_split": ds["train_split"],
        "val_split": ds["val_split"],
        "val_skip_tokens": str(ds[skip_key]),
        "model": model_name,
        "layers": str(model["layers"]),
        "d_model": str(model["d_model"]),
        "heads": str(model["heads"]),
        "ffn_dim": str(model["ffn_dim"]),
        "seq_len": "256",
        "batch_size": str(model["batch_size"]),
        "grad_accum": str(model["grad_accum"]),
        "global_tokens_per_step": str(global_tokens),
        "train_tokens": str(train_tokens),
        "val_tokens": str(val_tokens),
        "steps": str(steps),
        "eval_interval": str(eval_interval),
        "eval_batches": "10",
        "seed": str(seed),
        **method,
    }
    rows.append(row)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    preflight_methods = [m for m in METHODS if m["method"] in {"silu_adamw", "rlb_adamw"}]
    for dataset in MAIN_DATASETS:
        for method in preflight_methods:
            add_row(rows, phase="E0_preflight", dataset=dataset, model_name="M0", train_tokens=100_000_000, seed=1337, method=method)
            rows[-1]["row_id"] = f"E0_preflight_{dataset}_m0_smoke80_seed1337_{method['method']}"
            rows[-1]["steps"] = "80"
            rows[-1]["train_tokens"] = "2621440"
            rows[-1]["val_tokens"] = "200000"
            rows[-1]["eval_interval"] = "40"
            rows[-1]["eval_batches"] = "2"

    for tokens, phase in [(100_000_000, "E1_m0_100m"), (300_000_000, "E2_m0_300m")]:
        for dataset in MAIN_DATASETS:
            for seed in SEEDS:
                for method in METHODS:
                    add_row(rows, phase=phase, dataset=dataset, model_name="M0", train_tokens=tokens, seed=seed, method=method)

    m1_methods = [m for m in METHODS if m["method"] in {"silu_adamw", "rlb_adamw", "silu_soap", "rlb_soap"}]
    for dataset in ["dclm", "fineweb_edu", "c4_en"]:
        for seed in SEEDS:
            for method in m1_methods:
                add_row(rows, phase="E3_m1_300m", dataset=dataset, model_name="M1", train_tokens=300_000_000, seed=seed, method=method)

    for idx, row in enumerate(rows):
        row["row_index"] = str(idx)
    return rows


def verify(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("manifest is empty")
    for row in rows:
        if int(row["eval_interval"]) > 50:
            raise SystemExit(f"eval interval too sparse in {row['row_id']}")
    required = {m["method"] for m in METHODS}
    parity_phases = {"E0_preflight", "E1_m0_100m", "E2_m0_300m", "E3_m1_300m"}
    main_phases = {"E1_m0_100m", "E2_m0_300m"}

    methods_by_phase = {
        "E0_preflight": {"silu_adamw", "rlb_adamw"},
        "E1_m0_100m": required,
        "E2_m0_300m": required,
        "E3_m1_300m": {"silu_adamw", "rlb_adamw", "silu_soap", "rlb_soap"},
    }

    cells: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] in parity_phases:
            key = (row["phase"], row["dataset"], row["model"], row["train_tokens"], row["seed"])
            cells[key].append(row)

    for key, cell_rows in cells.items():
        phase = key[0]
        present = {row["method"] for row in cell_rows}
        missing = methods_by_phase[phase] - present
        if missing:
            raise SystemExit(f"incomplete matched cell {key}: missing {sorted(missing)}")

        adamw_outer = {
            (row["lr"], row["min_lr"], row["weight_decay"])
            for row in cell_rows
            if row["method"] in {"silu_adamw", "rlb_adamw"}
        }
    if main_phases - {row["phase"] for row in rows}:
        raise SystemExit("manifest is missing a main evidence phase")


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    by_phase = Counter(row["phase"] for row in rows)
    by_phase_dataset = Counter((row["phase"], row["dataset"]) for row in rows)
    print("rows", len(rows))
    for phase, count in sorted(by_phase.items()):
        print(f"{phase}: {count}")
    for (phase, dataset), count in sorted(by_phase_dataset.items()):
        print(f"{phase}/{dataset}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/manifests/iclr26_main_manifest.csv"))
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    write_manifest(rows, args.output)
    if args.print_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
