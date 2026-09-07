#!/usr/bin/env python3
"""Build the activation–optimizer experiment manifest."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

GRAIN = "rlb_fused_global_rational"

SMALL_MODEL = "12l_768d"
LARGE_MODEL = "18l_1024d"
PREFLIGHT_SUITE = "12l_768d_preflight_2621440_tokens_80_steps"
SUITE_100M = "12l_768d_100m_tokens_3050_steps"
SUITE_300M_SMALL = "12l_768d_300m_tokens_9150_steps"
SUITE_300M_LARGE = "18l_1024d_300m_tokens_9150_steps"

DATASETS = {
    "dclm": {
        "dataset_name": "mlfoundations/dclm-baseline-1.0",
        "dataset_config": "none",
        "dataset_revision": "a3b142c183aebe5af344955ae20836eb34dcf69b",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "fineweb_edu": {
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "dataset_revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "fineweb": {
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_config": "sample-10BT",
        "dataset_revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "dolma_sample": {
        "dataset_name": "allenai/dolma",
        "dataset_config": "v1_6-sample",
        "dataset_revision": "7f48140530a023e9ea4c5cfb141160922727d4d3",
        "text_column": "text",
        "train_split": "train",
        "val_split": "train",
        "val_skip_100m": 210_000_000,
        "val_skip_300m": 610_000_000,
    },
    "c4_en": {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "dataset_revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "text_column": "text",
        "train_split": "train",
        "val_split": "validation",
        "val_skip_100m": 0,
        "val_skip_300m": 0,
    },
}

MODELS = {
    SMALL_MODEL: {"layers": 12, "d_model": 768, "heads": 12, "ffn_dim": 2048, "batch_size": 16, "grad_accum": 2},
    LARGE_MODEL: {"layers": 18, "d_model": 1024, "heads": 16, "ffn_dim": 3072, "batch_size": 8, "grad_accum": 4},
}

METHODS = [
    {"method": "silu_adamw", "activation": "silu", "optimizer": "adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": ""},
    {"method": "rlb_adamw", "activation": GRAIN, "optimizer": "adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": ""},
    {"method": "silu_muon", "activation": "silu", "optimizer": "muon", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--muon-momentum 0.95 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw"},
    {"method": "rlb_muon", "activation": GRAIN, "optimizer": "muon", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--muon-momentum 0.95 --muon-ns-steps 5 --muon-adjust-lr-fn match_rms_adamw"},
    {"method": "silu_lion", "activation": "silu", "optimizer": "lion", "lr": "0.0001", "min_lr": "0.00001", "weight_decay": "0.10", "extra_args": ""},
    {"method": "rlb_lion", "activation": GRAIN, "optimizer": "lion", "lr": "0.0001", "min_lr": "0.00001", "weight_decay": "0.10", "extra_args": ""},
    {"method": "silu_soap", "activation": "silu", "optimizer": "soap_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--soap-precondition-frequency 50 --no-soap-one-sided"},
    {"method": "rlb_soap", "activation": GRAIN, "optimizer": "soap_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--soap-precondition-frequency 50 --no-soap-one-sided"},
    {"method": "silu_ademamix", "activation": "silu", "optimizer": "ademamix", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--ademamix-alpha 5.0 --ademamix-beta3 0.9999"},
    {"method": "rlb_ademamix", "activation": GRAIN, "optimizer": "ademamix", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--ademamix-alpha 5.0 --ademamix-beta3 0.9999"},
    {"method": "silu_came", "activation": "silu", "optimizer": "adafactor_came", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--came-confidence-scale 1.0"},
    {"method": "rlb_came", "activation": GRAIN, "optimizer": "adafactor_came", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--came-confidence-scale 1.0"},
    {"method": "silu_schedulefree", "activation": "silu", "optimizer": "schedule_free_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--schedule-free-beta1 0.90 --schedule-free-warmup-steps 0"},
    {"method": "rlb_schedulefree", "activation": GRAIN, "optimizer": "schedule_free_adamw", "lr": "0.0003", "min_lr": "0.00003", "weight_decay": "0.10", "extra_args": "--schedule-free-beta1 0.90 --schedule-free-warmup-steps 0"},
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
    "dataset_revision",
    "dataset_streaming",
    "text_column",
    "train_split",
    "val_split",
    "train_skip_documents",
    "validation_skip_documents",
    "train_skip_tokens",
    "val_skip_tokens",
    "cache_dir",
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
    "log_interval",
    "seed",
    "method",
    "activation",
    "optimizer",
    "lr",
    "min_lr",
    "weight_decay",
    "warmup_steps",
    "beta1",
    "beta2",
    "eps",
    "grad_clip",
    "muon_momentum",
    "muon_ns_steps",
    "muon_adjust_lr_fn",
    "init_std",
    "rational_init",
    "post_rational_init",
    "rational_group_size",
    "rational_max_groups",
    "probe_batch_size",
    "matrix_spectrum_interval",
    "telemetry_rlb_stat_every",
    "sam_rho",
    "sam_adaptive",
    "tokenizer",
    "tokenizer_revision",
    "extra_args",
]


def budget_spec(tokens: int) -> tuple[int, int, int, str]:
    if tokens == 100_000_000:
        return 3_050, 4_000_000, 50, "100m"
    if tokens == 300_000_000:
        return 9_150, 8_000_000, 50, "300m"
    raise ValueError(tokens)


def add_row(rows: list[dict[str, str]], *, phase: str, dataset: str, model_name: str, train_tokens: int, seed: int, method: dict[str, str]) -> None:
    steps, val_tokens, eval_interval, _ = budget_spec(train_tokens)
    ds = DATASETS[dataset]
    model = MODELS[model_name]
    skip_key = "val_skip_300m" if train_tokens == 300_000_000 else "val_skip_100m"
    global_tokens = 256 * 4 * model["batch_size"] * model["grad_accum"]
    row_id = f"{phase}_{dataset}_seed{seed}_{method['method']}"
    row = {
        "row_index": str(len(rows)),
        "row_id": row_id,
        "phase": phase,
        "dataset": dataset,
        "dataset_name": ds["dataset_name"],
        "dataset_config": ds["dataset_config"],
        "dataset_revision": ds["dataset_revision"],
        "dataset_streaming": "true",
        "text_column": ds["text_column"],
        "train_split": ds["train_split"],
        "val_split": ds["val_split"],
        "train_skip_documents": "0",
        "validation_skip_documents": "0",
        "train_skip_tokens": "0",
        "val_skip_tokens": str(ds[skip_key]),
        "cache_dir": f"experiments/cache/tokens_iclr26_main/{dataset}",
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
        "log_interval": "10",
        "seed": str(seed),
        "warmup_steps": "200",
        "beta1": "0.90",
        "beta2": "0.95",
        "eps": "0.00000001",
        "grad_clip": "1.0",
        "muon_momentum": "0.95",
        "muon_ns_steps": "5",
        "muon_adjust_lr_fn": "match_rms_adamw",
        "init_std": "0.02",
        "rational_init": "silu",
        "post_rational_init": "identity",
        "rational_group_size": "256",
        "rational_max_groups": "32",
        "probe_batch_size": "1" if model_name == SMALL_MODEL else "0",
        "matrix_spectrum_interval": "250" if model_name == SMALL_MODEL else "0",
        "telemetry_rlb_stat_every": "4",
        "sam_rho": "0.0",
        "sam_adaptive": "false",
        "tokenizer": "gpt2",
        "tokenizer_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        **method,
    }
    rows.append(row)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    preflight_methods = [m for m in METHODS if m["method"] in {"silu_adamw", "rlb_adamw"}]
    for dataset in MAIN_DATASETS:
        for method in preflight_methods:
            add_row(rows, phase=PREFLIGHT_SUITE, dataset=dataset, model_name=SMALL_MODEL, train_tokens=100_000_000, seed=1337, method=method)
            rows[-1]["row_id"] = f"{PREFLIGHT_SUITE}_{dataset}_seed1337_{method['method']}"
            rows[-1]["steps"] = "80"
            rows[-1]["train_tokens"] = "2621440"
            rows[-1]["val_tokens"] = "200000"
            rows[-1]["eval_interval"] = "40"
            rows[-1]["eval_batches"] = "2"

    for dataset in MAIN_DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                add_row(rows, phase=SUITE_100M, dataset=dataset, model_name=SMALL_MODEL, train_tokens=100_000_000, seed=seed, method=method)

    for dataset in MAIN_DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                add_row(rows, phase=SUITE_300M_SMALL, dataset=dataset, model_name=SMALL_MODEL, train_tokens=300_000_000, seed=seed, method=method)

    for dataset in MAIN_DATASETS:
        for seed in SEEDS:
            for method in METHODS:
                add_row(rows, phase=SUITE_300M_LARGE, dataset=dataset, model_name=LARGE_MODEL, train_tokens=300_000_000, seed=seed, method=method)

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
    parity_phases = {
        PREFLIGHT_SUITE, SUITE_100M, SUITE_300M_SMALL, SUITE_300M_LARGE,
    }
    endpoint_phases = {SUITE_100M, SUITE_300M_SMALL, SUITE_300M_LARGE}

    methods_by_phase = {
        PREFLIGHT_SUITE: {"silu_adamw", "rlb_adamw"},
        SUITE_100M: required,
        SUITE_300M_SMALL: required,
        SUITE_300M_LARGE: required,
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

        for optimizer in {
            row["optimizer"] for row in cell_rows
        }:
            arms = [row for row in cell_rows if row["optimizer"] == optimizer]
            if len(arms) != 2:
                raise SystemExit(
                    f"matched cell {key} has {len(arms)} rows for optimizer {optimizer}"
                )
            left, right = arms
            permitted = {"row_index", "row_id", "method", "activation"}
            mismatch = {
                field: (left[field], right[field])
                for field in FIELDNAMES
                if field not in permitted and left[field] != right[field]
            }
            if mismatch:
                raise SystemExit(
                    f"matched activation arms differ in {key}/{optimizer}: {mismatch}"
                )
    if endpoint_phases - {row["phase"] for row in rows}:
        raise SystemExit("manifest is missing an endpoint suite")


def write_manifest(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, lineterminator="\n"
        )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/protocol/activation_optimizer_manifest.csv"),
    )
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    verify(rows)
    write_manifest(rows, args.output)
    if args.print_summary:
        print_summary(rows)


if __name__ == "__main__":
    main()
