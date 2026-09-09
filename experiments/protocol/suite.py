#!/usr/bin/env python3
"""Entrypoint and cache views for the complete TILLER evaluation matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from training import train as trainer


PACKAGE = Path(__file__).resolve().parent
MATRIX = PACKAGE / "matrix.json"
EXACT_OPTIMIZER_KEY = "tiller_v1"
TWO_STAGE_OPTIMIZER_KEY = "tiller_then_muon_v1"
CANDIDATE_MODULE = (
    "experiments.protocol."
    "method_entrypoint"
)
WORLD_SIZE = 4
PARAMETER_COUNTS = {
    ("12l_768d", "control"): 123_551_232,
    ("12l_768d", "candidate"): 123_552_672,
    ("18l_1024d", "control"): 296_867_840,
    ("18l_1024d", "candidate"): 296_871_080,
}
_BASE_FAIRNESS_AUDIT = trainer.audit_optimizer_lr_wd_fairness


def matrix_payload() -> dict[str, Any]:
    payload = json.loads(MATRIX.read_text())
    expected_schema = "tiller_evaluation_matrix_v1"
    if payload.get("schema") != expected_schema:
        raise RuntimeError("full-method transfer matrix schema changed")
    if payload.get("matrix_rows") != 60 or len(payload.get("rows", ())) != 60:
        raise RuntimeError("full-method transfer matrix inventory changed")
    return payload


def row_at(index: int) -> dict[str, Any]:
    rows = matrix_payload()["rows"]
    if not 0 <= int(index) < len(rows):
        raise RuntimeError(f"matrix index {index} is outside 0..{len(rows) - 1}")
    row = dict(rows[int(index)])
    if int(row["matrix_index"]) != int(index):
        raise RuntimeError("matrix ordering changed")
    return row


def contract(row: dict[str, Any]) -> str:
    del row
    return trainer.TILLER_LR_WD_CONTRACT


def experiment_identity(row: dict[str, Any]) -> str:
    if row["candidate_optimizer"] == TWO_STAGE_OPTIMIZER_KEY:
        return trainer.TILLER_THEN_MUON_EXPERIMENT_IDENTITY
    return trainer.TILLER_EXPERIMENT_IDENTITY


def run_name(row: dict[str, Any], arm: str) -> str:
    if arm == "control":
        suffix = "control"
    elif arm == "candidate":
        suffixes = {
            EXACT_OPTIMIZER_KEY: "tiller",
            TWO_STAGE_OPTIMIZER_KEY: "tiller_then_muon",
        }
        try:
            suffix = suffixes[row["candidate_optimizer"]]
        except KeyError as exc:
            raise RuntimeError("unrecognized TILLER candidate optimizer") from exc
    else:
        raise RuntimeError(f"unknown arm {arm!r}")
    return (
        f"{row['phase']}-{row['dataset']}-seed{row['seed']}-"
        f"{suffix}"
    )


def jsonl_path(row: dict[str, Any], arm: str, output_root: Path) -> Path:
    if arm == "control":
        return (
            output_root
            / row["phase"]
            / row["dataset"]
            / row["source_manifest_row_id"]
            / f"{row['control_activation']}.jsonl"
        )
    if arm == "candidate":
        return (
            output_root
            / run_name(row, arm)
            / f"{row['candidate_activation']}.jsonl"
        )
    raise RuntimeError(f"unknown arm {arm!r}")


def _flag(argv: list[str], name: str, value: Any) -> None:
    argv.extend((name, str(value)))


def training_argv(
    row: dict[str, Any], arm: str, output_root: Path, *, timing: bool = False
) -> list[str]:
    if arm not in {"control", "candidate"}:
        raise RuntimeError(f"unknown arm {arm!r}")
    activation = row[f"{'control' if arm == 'control' else 'candidate'}_activation"]
    optimizer = row[f"{'control' if arm == 'control' else 'candidate'}_optimizer"]
    name = run_name(row, arm) + ("-timing" if timing else "")
    argv = [name, "--activation", activation, "--optimizer", optimizer]
    values = {
        "--run-name": name,
        "--fairness-contract": contract(row),
        "--experiment-identity": experiment_identity(row),
        "--dataset-name": row["dataset_name"],
        "--dataset-config": row["dataset_config"],
        "--dataset-revision": row["dataset_revision"],
        "--dataset-text-column": row["dataset_text_column"],
        "--train-split": row["train_split"],
        "--validation-split": row["validation_split"],
        "--train-skip-documents": row["train_skip_documents"],
        "--validation-skip-documents": row["validation_skip_documents"],
        "--train-skip-tokens": row["train_skip_tokens"],
        "--validation-skip-tokens": row["validation_skip_tokens"],
        "--cache-dir": row["cache_dir"],
        "--hf-cache": "experiments/cache/huggingface",
        "--output-dir": output_root,
        "--max-train-tokens": row["max_train_tokens"],
        "--max-val-tokens": row["max_val_tokens"],
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
        "--early-stop-min-step": 0,
        "--early-stop-max-val-loss": 0.0,
        "--early-stop-loss-increase": 0.0,
        "--timing-guard-min-step": 0,
        "--timing-guard-max-seconds-per-step": 0.0,
        "--timing-guard-max-optimizer-step-seconds": 0.0,
    }
    for name_flag, value in values.items():
        _flag(argv, name_flag, value)
    argv.extend(("--dataset-streaming", "--no-sam-adaptive"))
    return argv


def expected_args(row: dict[str, Any], arm: str) -> dict[str, Any]:
    return {
        "activation": row[f"{'control' if arm == 'control' else 'candidate'}_activation"],
        "optimizer": row[f"{'control' if arm == 'control' else 'candidate'}_optimizer"],
        "fairness_contract": contract(row),
        "experiment_identity": experiment_identity(row),
        "dataset_name": row["dataset_name"],
        "dataset_config": row["dataset_config"],
        "dataset_revision": row["dataset_revision"],
        "dataset_streaming": True,
        "dataset_text_column": row["dataset_text_column"],
        "train_split": row["train_split"],
        "validation_split": row["validation_split"],
        "train_skip_documents": row["train_skip_documents"],
        "validation_skip_documents": row["validation_skip_documents"],
        "train_skip_tokens": row["train_skip_tokens"],
        "validation_skip_tokens": row["validation_skip_tokens"],
        "max_train_tokens": row["max_train_tokens"],
        "max_val_tokens": row["max_val_tokens"],
        "steps": row["steps"],
        "layers": row["layers"],
        "d_model": row["d_model"],
        "heads": row["heads"],
        "ffn_dim": row["ffn_dim"],
        "seq_len": row["seq_len"],
        "batch_size": row["batch_size"],
        "grad_accum": row["grad_accum"],
        "eval_interval": row["eval_interval"],
        "eval_batches": row["eval_batches"],
        "log_interval": row["log_interval"],
        "seed": row["seed"],
        "lr": row["lr"],
        "min_lr": row["min_lr"],
        "warmup_steps": row["warmup_steps"],
        "weight_decay": row["weight_decay"],
        "beta1": row["beta1"],
        "beta2": row["beta2"],
        "eps": row["eps"],
        "grad_clip": row["grad_clip"],
        "muon_momentum": row["muon_momentum"],
        "muon_ns_steps": row["muon_ns_steps"],
        "muon_adjust_lr_fn": row["muon_adjust_lr_fn"],
        "init_std": row["init_std"],
        "rational_init": row["rational_init"],
        "post_rational_init": row["post_rational_init"],
        "rational_group_size": row["rational_group_size"],
        "rational_max_groups": row["rational_max_groups"],
        "probe_batch_size": row["probe_batch_size"],
        "matrix_spectrum_interval": row["matrix_spectrum_interval"],
        "telemetry_rlb_stat_every": row["telemetry_rlb_stat_every"],
        "sam_rho": row["sam_rho"],
        "sam_adaptive": row["sam_adaptive"],
        "tokenizer": row["tokenizer"],
        "tokenizer_revision": row["tokenizer_revision"],
    }


def verify_args(args: argparse.Namespace, row: dict[str, Any], arm: str) -> None:
    mismatch = {
        key: {"observed": getattr(args, key, None), "required": value}
        for key, value in expected_args(row, arm).items()
        if getattr(args, key, None) != value
    }
    if mismatch:
        raise RuntimeError(f"full-transfer argument mismatch: {mismatch}")


def install_identity(row: dict[str, Any], arm: str) -> None:
    required_contract = contract(row)
    def audit(
        args: argparse.Namespace,
        *,
        world_size: int,
        global_tokens: int,
        train_token_count: int,
        val_token_count: int,
        parameter_count: int,
    ) -> dict[str, Any] | None:
        if (
            args.fairness_contract != required_contract
            or args.experiment_identity != experiment_identity(row)
        ):
            return None
        verify_args(args, row, arm)
        required = {
            "world_size": (int(world_size), WORLD_SIZE),
            "global_tokens_per_step": (
                int(global_tokens), int(row["global_tokens_per_step"])
            ),
            "train_tokens": (int(train_token_count), int(row["max_train_tokens"])),
            "val_tokens": (int(val_token_count), int(row["max_val_tokens"])),
            "parameter_count": (
                int(parameter_count), PARAMETER_COUNTS[(row["model"], arm)]
            ),
        }
        mismatch = {
            key: {"observed": observed, "required": expected}
            for key, (observed, expected) in required.items()
            if observed != expected
        }
        if mismatch:
            raise RuntimeError(f"full-transfer runtime identity mismatch: {mismatch}")
        return {
            "passed": True,
            "matrix_index": int(row["matrix_index"]),
            "source_manifest_row_index": int(row["source_manifest_row_index"]),
            "source_manifest_row_id": row["source_manifest_row_id"],
            "model": row["model"],
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "arm": arm,
            "parameter_count": int(parameter_count),
            "train_token_count": int(train_token_count),
            "val_token_count": int(val_token_count),
        }

    trainer.install_tiller_identity_auditor(audit)


def install_exact_weight_decay_policy(
    row: dict[str, Any], candidate_module: Any | None = None
) -> None:
    """Match the literal parameter-routing policy of the paired control."""

    decay_tied_embedding = row["control_optimizer"] == "adamw"
    if decay_tied_embedding:
        def small_model_fairness_audit(model, optimizer, args):
            original = trainer.is_tied_embedding_parameter_name
            trainer.is_tied_embedding_parameter_name = lambda name: False
            try:
                return _BASE_FAIRNESS_AUDIT(model, optimizer, args)
            finally:
                trainer.is_tied_embedding_parameter_name = original

        trainer.audit_optimizer_lr_wd_fairness = small_model_fairness_audit
    else:
        trainer.audit_optimizer_lr_wd_fairness = _BASE_FAIRNESS_AUDIT

    if candidate_module is not None:
        candidate_module.set_experiment_layout(
            microbatches=int(row["grad_accum"]),
            decay_tied_embedding=decay_tied_embedding,
        )


def _payload_tokens(path: Path) -> tuple[dict[str, Any], "torch.Tensor"]:
    import torch

    if not path.is_file():
        raise RuntimeError(f"required token source is absent: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("tokens")):
        raise RuntimeError(f"token source schema changed: {path}")
    return payload, payload["tokens"]


def install_cache_views(row: dict[str, Any]) -> None:
    import torch

    original = trainer.load_or_tokenize

    def load_or_view(args: argparse.Namespace, split: str, max_tokens: int):
        requested = trainer.token_cache_path(args, split, max_tokens)
        if requested.is_file():
            return original(args, split, max_tokens)
        if int(max_tokens) not in {100_000_000, 4_000_000}:
            raise RuntimeError(f"required token cache is absent: {requested}")
        if split == "train":
            source = trainer.token_cache_path(args, "train", 300_000_000)
            payload, tokens = _payload_tokens(source)
            start, stop = 0, 100_000_000
        elif split == "validation" and args.validation_split == "train":
            source_args = SimpleNamespace(**vars(args))
            source_args.train_skip_tokens = 0
            source = trainer.token_cache_path(source_args, "train", 300_000_000)
            payload, tokens = _payload_tokens(source)
            start, stop = 210_000_000, 214_000_000
        elif split == "validation" and args.validation_split == "validation":
            source = trainer.token_cache_path(args, "validation", 8_000_000)
            payload, tokens = _payload_tokens(source)
            start, stop = 0, 4_000_000
        else:
            raise RuntimeError("unapproved 100M cache-view request")
        metadata = {
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "dataset_streaming": True,
            "dataset_text_column": args.dataset_text_column,
            "skip_documents": 0,
            "tokenizer": args.tokenizer,
        }
        mismatch = {
            key: {"observed": payload.get(key), "required": value}
            for key, value in metadata.items()
            if payload.get(key) != value
        }
        if mismatch or tokens.dtype != torch.int32 or int(tokens.numel()) < stop:
            raise RuntimeError(
                f"100M cache-view source mismatch: metadata={mismatch}, "
                f"dtype={tokens.dtype}, tokens={tokens.numel()}, stop={stop}"
            )
        view = tokens[start:stop]
        if int(view.numel()) != int(max_tokens) or not view.is_contiguous():
            raise RuntimeError("100M cache view has the wrong shape/layout")
        return view

    trainer.load_or_tokenize = load_or_view


def install_fixed_probe_layout(row: dict[str, Any]) -> None:
    from optimizer_design.tiller import configure_microbatch_count

    configure_microbatch_count(int(row["grad_accum"]))


def _selector() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matrix-index", required=True, type=int)
    parser.add_argument("--arm", required=True, choices=("control", "candidate"))
    parser.add_argument("--output-root", required=True, type=Path)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise RuntimeError(f"unapproved external training arguments: {unknown}")
    return args


def main() -> None:
    selector = _selector()
    row = row_at(selector.matrix_index)
    arm = selector.arm
    install_identity(row, arm)
    install_cache_views(row)
    install_exact_weight_decay_policy(row)
    if int(row["steps"]) == 9150:
        from training.exact_resume import install as install_exact_resume

        install_exact_resume(trainer)
    sys.argv = training_argv(row, arm, selector.output_root)
    if arm == "control":
        trainer.main()
        return

    import importlib

    candidate = importlib.import_module(CANDIDATE_MODULE)
    install_exact_weight_decay_policy(row, candidate)
    install_fixed_probe_layout(row)
    candidate.main()


if __name__ == "__main__":
    main()
