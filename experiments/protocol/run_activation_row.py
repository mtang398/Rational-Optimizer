#!/usr/bin/env python3
"""Run one activation–optimizer manifest row with the shared trainer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.protocol.row_tools import records, terminal


PACKAGE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE / "activation_optimizer_manifest.csv"
SOURCE_FREEZE = PACKAGE / "SOURCE_FREEZE.sha256"
EXACT_LR_WD_CONTRACT = "exact_lr_wd_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_sha256(row: dict[str, str]) -> str:
    payload = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_identity(
    manifest: Path, row: dict[str, str]
) -> dict[str, str]:
    if not SOURCE_FREEZE.is_file():
        raise RuntimeError(f"source freeze is missing: {SOURCE_FREEZE}")
    return {
        "source_manifest_sha256": sha256(manifest),
        "source_manifest_row_sha256": row_sha256(row),
        "source_manifest_row_id": row["row_id"],
        "source_manifest_row_index": row["row_index"],
        "source_freeze_sha256": sha256(SOURCE_FREEZE),
    }


def matches_source_identity(
    path: Path,
    row: dict[str, str],
    identity: dict[str, str],
) -> bool:
    if not terminal(path, int(row["steps"])):
        return False
    configs = [record for record in records(path) if record.get("event") == "config"]
    if len(configs) != 1:
        return False
    config = configs[0]
    if any(str(config.get(key, "")) != value for key, value in identity.items()):
        return False
    if config.get("experiment_identity") != "none":
        return False
    exact = {
        "activation": row["activation"],
        "optimizer": row["optimizer"],
        "dataset": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_revision": row["dataset_revision"],
        "tokenizer": row["tokenizer"],
        "tokenizer_revision": row["tokenizer_revision"],
    }
    if any(str(config.get(key, "")) != value for key, value in exact.items()):
        return False
    integers = {
        "train_tokens": row["train_tokens"],
        "val_tokens": row["val_tokens"],
        "steps": row["steps"],
        "layers": row["layers"],
        "d_model": row["d_model"],
        "heads": row["heads"],
        "ffn_dim": row["ffn_dim"],
        "batch_size_per_gpu": row["batch_size"],
        "grad_accum": row["grad_accum"],
        "global_tokens_per_step": row["global_tokens_per_step"],
        "warmup_steps": row["warmup_steps"],
    }
    try:
        integer_mismatch = any(
            int(config.get(key, -1)) != int(value)
            for key, value in integers.items()
        )
    except (TypeError, ValueError):
        return False
    if integer_mismatch:
        return False
    numbers = {
        "optimizer_lr": row["lr"],
        "optimizer_min_lr": row["min_lr"],
        "optimizer_weight_decay": row["weight_decay"],
        "optimizer_beta1": row["beta1"],
        "optimizer_beta2": row["beta2"],
        "optimizer_eps": row["eps"],
        "grad_clip": row["grad_clip"],
        "init_std": row["init_std"],
    }
    try:
        number_mismatch = any(
            float(config.get(key, "nan")) != float(value)
            for key, value in numbers.items()
        )
    except (TypeError, ValueError):
        return False
    if number_mismatch:
        return False
    fairness = config.get("optimizer_lr_wd_fairness")
    if not isinstance(fairness, dict):
        return False
    try:
        scalars_match = (
            float(fairness.get("base_lr", "nan")) == float(row["lr"])
            and float(fairness.get("minimum_lr", "nan")) == float(row["min_lr"])
            and float(fairness.get("base_weight_decay", "nan"))
            == float(row["weight_decay"])
        )
    except (TypeError, ValueError):
        return False
    return bool(
        fairness.get("contract") == EXACT_LR_WD_CONTRACT
        and fairness.get("passed") is True
        and scalars_match
    )


def archive_existing(path: Path) -> None:
    digest = sha256(path)[:12]
    archive = path.with_name(f"{path.stem}.superseded-{digest}{path.suffix}")
    if archive.exists():
        raise RuntimeError(f"existing archive blocks forced rerun: {archive}")
    path.rename(archive)
    sidecar = path.with_suffix(".wall_clock.json")
    if sidecar.exists():
        sidecar.rename(archive.with_suffix(".wall_clock.json"))


def read_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"row index {index} is outside 0..{len(rows) - 1}")
    row = rows[index]
    if int(row["row_index"]) != index:
        raise RuntimeError("manifest row ordering changed")
    return row


def flag(argv: list[str], name: str, value: str | int | float) -> None:
    argv.extend((name, str(value)))


def training_arguments(row: dict[str, str], output_root: Path) -> list[str]:
    output_dir = output_root / row["phase"] / row["dataset"]
    argv = [
        "training/train.py",
        "--activation", row["activation"],
        "--optimizer", row["optimizer"],
        "--fairness-contract", EXACT_LR_WD_CONTRACT,
        "--run-name", row["row_id"],
        "--dataset-name", row["dataset_name"],
        "--dataset-config", row["dataset_config"],
        "--dataset-revision", row["dataset_revision"],
        "--dataset-text-column", row["text_column"],
        "--train-split", row["train_split"],
        "--validation-split", row["val_split"],
        "--cache-dir", row["cache_dir"],
        "--hf-cache", "experiments/cache/huggingface",
        "--output-dir", str(output_dir),
    ]
    if row["dataset_streaming"].lower() == "true":
        argv.append("--dataset-streaming")
    else:
        argv.append("--no-dataset-streaming")
    if row["sam_adaptive"].lower() == "true":
        argv.append("--sam-adaptive")
    else:
        argv.append("--no-sam-adaptive")
    values = {
        "--train-skip-documents": row["train_skip_documents"],
        "--validation-skip-documents": row["validation_skip_documents"],
        "--train-skip-tokens": row["train_skip_tokens"],
        "--validation-skip-tokens": row["val_skip_tokens"],
        "--max-train-tokens": row["train_tokens"],
        "--max-val-tokens": row["val_tokens"],
        "--steps": row["steps"],
        "--layers": row["layers"],
        "--d-model": row["d_model"],
        "--heads": row["heads"],
        "--ffn-dim": row["ffn_dim"],
        "--seq-len": row["seq_len"],
        "--batch-size": row["batch_size"],
        "--grad-accum": row["grad_accum"],
        "--eval-interval": row["eval_interval"],
        "--eval-batches": row["eval_batches"],
        "--log-interval": row["log_interval"],
        "--seed": row["seed"],
        "--lr": row["lr"],
        "--min-lr": row["min_lr"],
        "--warmup-steps": row["warmup_steps"],
        "--weight-decay": row["weight_decay"],
        "--beta1": row["beta1"],
        "--beta2": row["beta2"],
        "--eps": row["eps"],
        "--grad-clip": row["grad_clip"],
        "--muon-momentum": row["muon_momentum"],
        "--muon-ns-steps": row["muon_ns_steps"],
        "--muon-adjust-lr-fn": row["muon_adjust_lr_fn"],
        "--init-std": row["init_std"],
        "--rational-init": row["rational_init"],
        "--post-rational-init": row["post_rational_init"],
        "--rational-group-size": row["rational_group_size"],
        "--rational-max-groups": row["rational_max_groups"],
        "--probe-batch-size": row["probe_batch_size"],
        "--matrix-spectrum-interval": row["matrix_spectrum_interval"],
        "--telemetry-rlb-stat-every": row["telemetry_rlb_stat_every"],
        "--sam-rho": row["sam_rho"],
        "--tokenizer": row["tokenizer"],
        "--tokenizer-revision": row["tokenizer_revision"],
    }
    for name, value in values.items():
        flag(argv, name, value)
    argv.extend(shlex.split(row["extra_args"]))
    return argv


def result_path(row: dict[str, str], output_root: Path) -> Path:
    return (
        output_root
        / row["phase"]
        / row["dataset"]
        / row["row_id"]
        / f"{row['activation']}.jsonl"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs/activation_optimizer"),
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    row = read_row(args.manifest, args.row_index)
    path = result_path(row, args.output_root)
    identity = source_identity(args.manifest, row)
    if path.is_file():
        if matches_source_identity(path, row, identity) and not args.force:
            print(f"complete: {path}")
            return
        if not args.force:
            raise RuntimeError(
                f"existing trajectory does not match the current row/source identity: {path}"
            )
        archive_existing(path)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.world_size}",
        *training_arguments(row, args.output_root),
    ]
    wall_clock = path.with_suffix(".wall_clock.json")
    timed_command = [
        sys.executable,
        "-m",
        "experiments.protocol.run_timed",
        "--output",
        str(wall_clock),
        "--arm",
        row["method"],
        "--",
        *command,
    ]
    environment = os.environ.copy()
    environment.update(
        {f"RATIONALOPT_{key.upper()}": value for key, value in identity.items()}
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path.cwd() / "activation"), str(Path.cwd()), environment.get("PYTHONPATH")))
    )
    subprocess.run(timed_command, check=True, env=environment)
    if not terminal(path, int(row["steps"])):
        raise RuntimeError(f"row did not produce a complete endpoint: {path}")


if __name__ == "__main__":
    main()
