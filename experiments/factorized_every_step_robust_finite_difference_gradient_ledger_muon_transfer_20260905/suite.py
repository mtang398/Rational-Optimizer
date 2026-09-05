#!/usr/bin/env python3
"""Fail-closed entrypoint and cache views for the full transfer matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from training import transformer_lm_compare as trainer


PACKAGE = Path(__file__).resolve().parent
MATRIX = PACKAGE / "matrix.json"
EXACT_OPTIMIZER_KEY = "factorized_every_step_rfd_gradient_ledger_muon_v1"
CANDIDATE_MODULE = (
    "experiments.rlb_300m_4000_design_20260731."
    "candidate_entrypoint_factorized_every_step_rfd_gradient_ledger_muon_v1"
)
WORLD_SIZE = 4
_BASE_FAIRNESS_AUDIT = trainer.audit_optimizer_lr_wd_fairness
_BASE_TIED_EMBEDDING_TEST = trainer.is_tied_embedding_parameter_name


def matrix_payload() -> dict[str, Any]:
    payload = json.loads(MATRIX.read_text())
    expected_schema = (
        "factorized_every_step_robust_finite_difference_gradient_ledger_muon_"
        "transfer_matrix_v1"
    )
    if payload.get("schema") != expected_schema:
        raise RuntimeError("full-method transfer matrix schema changed")
    if payload.get("matrix_rows") != 24 or len(payload.get("rows", ())) != 24:
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
    return trainer.UNIT_LR_WD_FAIRNESS_CONTRACT


def run_name(row: dict[str, Any], arm: str) -> str:
    return (
        f"{row['model'].lower()}-{row['dataset']}-seed{row['seed']}-"
        f"{'control' if arm == 'control' else 'factorized-every-step-robust-finite-difference-gradient-ledger-muon'}"
    )


def jsonl_path(row: dict[str, Any], arm: str, output_root: Path) -> Path:
    activation = (
        row["control_activation"] if arm == "control" else row["candidate_activation"]
    )
    return output_root / run_name(row, arm) / f"{activation}.jsonl"


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
        "--dataset-name": row["dataset_name"],
        "--dataset-config": row["dataset_config"],
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
        "dataset_name": row["dataset_name"],
        "dataset_config": row["dataset_config"],
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
    trainer.UNIT_LR_WD_FAIRNESS_CONTRACT = required_contract

    def audit(
        args: argparse.Namespace,
        *,
        world_size: int,
        global_tokens: int,
        train_token_count: int,
        val_token_count: int,
        parameter_count: int,
    ) -> dict[str, Any] | None:
        if args.fairness_contract != required_contract:
            return None
        verify_args(args, row, arm)
        parameter_counts = {
            ("M0", "control"): 123_551_232,
            ("M0", "candidate"): 123_552_672,
            ("M1", "control"): 296_867_840,
            ("M1", "candidate"): 296_871_080,
        }
        required = {
            "world_size": (int(world_size), WORLD_SIZE),
            "global_tokens_per_step": (
                int(global_tokens), int(row["global_tokens_per_step"])
            ),
            "train_tokens": (int(train_token_count), int(row["max_train_tokens"])),
            "val_tokens": (int(val_token_count), int(row["max_val_tokens"])),
            "parameter_count": (
                int(parameter_count), parameter_counts[(row["model"], arm)]
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
            "model": row["model"],
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "arm": arm,
            "parameter_count": int(parameter_count),
            "train_token_count": int(train_token_count),
            "val_token_count": int(val_token_count),
        }

    trainer.audit_m1_300m_campaign_identity = audit


def install_exact_weight_decay_policy(
    row: dict[str, Any], candidate_module: Any | None = None
) -> None:
    """Match each scale's literal control parameter-routing policy.

    The authoritative GitHub M0 AdamW cell applies WD=0.1 to its tied token
    embedding because its ``split_decay_parameters`` predicate only exempts
    low-rank/bias and rational parameters.  The established M1 Muon cell puts
    that tied embedding in AdamW with WD=0.0.  The scalar WD is unchanged;
    this adapter makes the candidate and the fairness audit follow the exact
    scale-specific control routing.
    """

    if row["model"] == "M0":
        def m0_fairness_audit(model, optimizer, args):
            original = trainer.is_tied_embedding_parameter_name
            trainer.is_tied_embedding_parameter_name = lambda name: False
            try:
                return _BASE_FAIRNESS_AUDIT(model, optimizer, args)
            finally:
                trainer.is_tied_embedding_parameter_name = original

        trainer.audit_optimizer_lr_wd_fairness = m0_fairness_audit
    else:
        trainer.audit_optimizer_lr_wd_fairness = _BASE_FAIRNESS_AUDIT

    if candidate_module is None:
        return
    scalable = candidate_module.base.base.scalable_base
    freeze_name = "_FULL_TRANSFER_BASE_PARTITION_PARAMETERS"
    if not hasattr(scalable, freeze_name):
        setattr(scalable, freeze_name, scalable._partition_parameters)
    base_partition = getattr(scalable, freeze_name)
    if row["model"] != "M0":
        scalable._partition_parameters = base_partition
        return

    def m0_partition_parameters(model, blocks):
        structural_ids, muon_named, adam_decay, adam_no_decay = base_partition(
            model, blocks
        )
        tied_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _BASE_TIED_EMBEDDING_TEST(name)
        }
        if len(tied_ids) != 1:
            raise RuntimeError(
                f"M0 exact AdamW routing requires one tied embedding, got {len(tied_ids)}"
            )
        found = [parameter for parameter in adam_no_decay if id(parameter) in tied_ids]
        if len(found) != 1:
            raise RuntimeError("M0 tied embedding was not in the inherited AdamW group")
        adam_no_decay = [
            parameter for parameter in adam_no_decay if id(parameter) not in tied_ids
        ]
        adam_decay = [*adam_decay, *found]
        return structural_ids, muon_named, adam_decay, adam_no_decay

    scalable._partition_parameters = m0_partition_parameters


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
        if row["model"] != "M0":
            raise RuntimeError(f"exact M1 cache is absent: {requested}")
        if int(max_tokens) not in {100_000_000, 4_000_000}:
            raise RuntimeError("unapproved M0 cache-view length")
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
            raise RuntimeError("unapproved M0 cache-view request")
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
                f"M0 cache-view source mismatch: metadata={mismatch}, "
                f"dtype={tokens.dtype}, tokens={tokens.numel()}, stop={stop}"
            )
        view = tokens[start:stop]
        if int(view.numel()) != int(max_tokens) or not view.is_contiguous():
            raise RuntimeError("M0 cache view has the wrong shape/layout")
        return view

    trainer.load_or_tokenize = load_or_view


def install_fixed_probe_layout(row: dict[str, Any]) -> None:
    from optimizer_design import rlb_fixed32_transaction_muon_base as fixed32
    from optimizer_design import rlb_ten_probe_loss_image_muon as probe

    expected = int(row["grad_accum"])
    fixed32.EXPECTED_MICROBATCHES = expected
    probe.EXPECTED_MICROBATCHES = expected
    if fixed32.EXPECTED_MICROBATCHES != expected or probe.EXPECTED_MICROBATCHES != expected:
        raise RuntimeError("fixed-probe microbatch layout did not install")


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
    if row["model"] == "M1":
        from training.fairness_exact_resume import install as install_exact_resume

        install_exact_resume(trainer)
    sys.argv = training_argv(row, arm, selector.output_root)
    if arm == "control":
        trainer.main()
        return

    import importlib

    candidate = importlib.import_module(CANDIDATE_MODULE)
    install_exact_weight_decay_policy(row, candidate)
    install_fixed_probe_layout(row)
    historical = candidate.base.base.historical_base
    historical._verify_frozen_cell = lambda args: verify_args(args, row, arm)
    candidate.main()


if __name__ == "__main__":
    main()
