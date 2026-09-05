#!/usr/bin/env python3
"""Strict internal exact-resume support for fairness-contracted experiments.

The base trainer intentionally rejects arbitrary external checkpoints for a
fairness-contracted run.  This module narrows the exception to a checkpoint
inside the run directory, strengthens the checkpoint contract with every CLI
argument, and repairs the small atomic-commit window between rank files and
the main checkpoint after a zero-grace Slurm preemption.

Installing this module changes no optimizer or model equations.  A fresh run
still starts from the contracted seed.  A restarted run may load only the
checkpoint created by the same run name and complete argument contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


SCHEMA = "fairness_internal_exact_resume_v1"
CHECKPOINT_BASENAME = "trainer_state.pt"


def expected_checkpoint_path(args) -> Path:
    """Return the only resume checkpoint path allowed for ``args``."""

    return (
        Path(args.output_dir)
        / args.run_name
        / ".exact_resume"
        / CHECKPOINT_BASENAME
    ).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"resume contract contains unsupported value: {type(value)!r}")


def complete_argument_digest(args) -> str:
    """Hash the complete parsed argument namespace deterministically."""

    payload = {
        key: _jsonable(value)
        for key, value in sorted(vars(args).items())
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def full_lr_trace_digest(args) -> str:
    """Reconstruct the full contracted LR trace asserted during training."""

    digest = hashlib.sha256()
    for zero_step in range(int(args.steps)):
        if int(args.warmup_steps) > 0 and zero_step < int(args.warmup_steps):
            lr = float(args.lr) * float(zero_step + 1) / float(args.warmup_steps)
        else:
            progress = (zero_step - int(args.warmup_steps)) / max(
                1, int(args.steps) - int(args.warmup_steps)
            )
            progress = min(1.0, max(0.0, progress))
            import math

            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = float(args.min_lr) + cosine * (
                float(args.lr) - float(args.min_lr)
            )
        digest.update(f"{zero_step + 1}:{float(lr).hex()}\n".encode("ascii"))
    return digest.hexdigest()


def _checkpoint_step(path: Path, expected_schema: str) -> int | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if payload.get("schema") != expected_schema:
        return None
    try:
        return int(payload["completed_step"])
    except (KeyError, TypeError, ValueError):
        return None


def recover_interrupted_commit(path: Path, world_size: int) -> int | None:
    """Promote the newest complete checkpoint generation atomically.

    The base saver writes every rank's ``.tmp`` payload, then promotes rank
    payloads, and finally promotes the main payload.  A zero-grace preemption
    can therefore leave a mixture of stable and temporary files.  For each
    main generation we accept it only when every rank has a readable payload
    with the identical completed step, then promote the newest complete set.
    """

    path = path.resolve()
    main_candidates = (path, Path(f"{path}.tmp"))
    complete = []
    for main in main_candidates:
        step = _checkpoint_step(main, "transformer_lm_exact_resume_v1")
        if step is None:
            continue
        rank_choices = []
        for rank in range(int(world_size)):
            stable = Path(f"{path}.rank{rank}.pt")
            temporary = Path(f"{stable}.tmp")
            matching = next(
                (
                    candidate
                    for candidate in (stable, temporary)
                    if _checkpoint_step(
                        candidate, "transformer_lm_exact_resume_rank_v1"
                    )
                    == step
                ),
                None,
            )
            if matching is None:
                rank_choices = []
                break
            rank_choices.append(matching)
        if len(rank_choices) == int(world_size):
            complete.append((step, main, rank_choices))

    if not complete:
        return None
    step, main, ranks = max(complete, key=lambda item: item[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    for rank, source in enumerate(ranks):
        target = Path(f"{path}.rank{rank}.pt")
        if source != target:
            os.replace(source, target)
    if main != path:
        os.replace(main, path)
    return int(step)


def prepare_trajectory(
    checkpoint_path: Path,
    trajectory_path: Path,
    world_size: int,
) -> dict:
    """Make an append-only trajectory consistent with its latest checkpoint.

    Records written after the last durable checkpoint belong to a transaction
    that will be replayed.  They are archived and removed before restart so
    endpoint integration never double-counts the replayed steps.
    """

    checkpoint_path = checkpoint_path.resolve()
    trajectory_path = trajectory_path.resolve()
    step = recover_interrupted_commit(checkpoint_path, int(world_size))
    if not trajectory_path.is_file() or trajectory_path.stat().st_size == 0:
        return {"schema": SCHEMA, "mode": "fresh", "completed_step": step}

    raw_lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    if step is None:
        archive = trajectory_path.with_name(
            f"{trajectory_path.name}.orphaned-before-first-checkpoint-"
            f"{time.time_ns()}"
        )
        os.replace(trajectory_path, archive)
        return {
            "schema": SCHEMA,
            "mode": "fresh_after_archiving_uncheckpointed_partial",
            "completed_step": None,
            "archive": str(archive),
        }

    kept = []
    removed = []
    config_count = 0
    for line in raw_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid JSONL record in exact-resume trajectory: {error}"
            ) from error
        event = record.get("event")
        if event == "config":
            config_count += 1
        record_step = record.get("step")
        should_keep = (
            event not in {"summary", "checkpoint", "stopped_early"}
            and (
                record_step is None
                or int(record_step) <= int(step)
            )
        )
        (kept if should_keep else removed).append(line)
    if config_count != 1:
        raise RuntimeError(
            "exact-resume trajectory must contain exactly one config record"
        )
    if not any(
        int(json.loads(line).get("step", -1)) == int(step)
        for line in kept
    ):
        raise RuntimeError(
            "trajectory does not contain the durable checkpoint boundary"
        )

    if removed:
        archive = trajectory_path.with_name(
            f"{trajectory_path.name}.tail-after-step-{step}-{time.time_ns()}"
        )
        archive.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        temporary = Path(f"{trajectory_path}.resume.tmp")
        temporary.write_text("\n".join(kept) + "\n", encoding="utf-8")
        os.replace(temporary, trajectory_path)
    else:
        archive = None
    return {
        "schema": SCHEMA,
        "mode": "resume",
        "completed_step": int(step),
        "removed_records": len(removed),
        "archive": None if archive is None else str(archive),
    }


def install(trainer) -> None:
    """Install the strict internal-resume validation and contract hooks."""

    if getattr(trainer, "_fairness_exact_resume_schema", None) == SCHEMA:
        return

    base_validate = trainer.validate_optimizer_protocol
    base_contract = trainer._resume_contract
    base_load = trainer._load_exact_resume_checkpoint
    base_write_jsonl = trainer.write_jsonl
    base_rank0_print = trainer.rank0_print
    active_args = None

    def audited_record(record):
        if (
            active_args is None
            or not isinstance(record, dict)
            or record.get("event") != "summary"
            or active_args.resume_checkpoint is None
        ):
            return record
        corrected = dict(record)
        corrected["realized_lr_audit_steps"] = int(active_args.steps)
        corrected["realized_lr_trace_sha256"] = full_lr_trace_digest(
            active_args
        )
        corrected["realized_lr_trace_reconstructed_after_exact_resume"] = True
        return corrected

    def strict_contract(args, world_size: int) -> dict:
        payload = dict(base_contract(args, world_size))
        payload.update(
            {
                "internal_resume_schema": SCHEMA,
                "fairness_contract": args.fairness_contract,
                "complete_argument_sha256": complete_argument_digest(args),
            }
        )
        return payload

    def strict_validate(args) -> None:
        nonlocal active_args
        active_args = args
        checkpoint = args.resume_checkpoint
        if checkpoint is None:
            base_validate(args)
            return
        actual = Path(checkpoint).resolve()
        expected = expected_checkpoint_path(args)
        if actual != expected:
            raise ValueError(
                "fairness exact resume must use the run-local checkpoint: "
                f"{expected}"
            )
        interval = int(args.resume_checkpoint_interval)
        if interval <= 0 or interval != int(args.eval_interval):
            raise ValueError(
                "fairness exact resume interval must equal the positive "
                "evaluation interval"
            )
        # Preserve all ordinary fairness checks while suppressing only the
        # base trainer's blanket rejection of an external checkpoint.
        args.resume_checkpoint = None
        try:
            base_validate(args)
        finally:
            args.resume_checkpoint = checkpoint

    def robust_load(path, *, rank, world_size, is_distributed, **kwargs):
        resolved = Path(path).resolve()
        if int(rank) == 0:
            recover_interrupted_commit(resolved, int(world_size))
        if is_distributed:
            dist.barrier()
        return base_load(
            resolved,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed,
            **kwargs,
        )

    def strict_write_jsonl(path, record):
        return base_write_jsonl(path, audited_record(record))

    def strict_rank0_print(rank, message):
        if int(rank) == 0 and isinstance(message, str):
            try:
                record = json.loads(message)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict):
                message = json.dumps(audited_record(record), sort_keys=True)
        return base_rank0_print(rank, message)

    trainer.validate_optimizer_protocol = strict_validate
    trainer._resume_contract = strict_contract
    trainer._load_exact_resume_checkpoint = robust_load
    trainer.write_jsonl = strict_write_jsonl
    trainer.rank0_print = strict_rank0_print
    trainer._fairness_exact_resume_schema = SCHEMA


__all__ = (
    "CHECKPOINT_BASENAME",
    "SCHEMA",
    "complete_argument_digest",
    "expected_checkpoint_path",
    "full_lr_trace_digest",
    "install",
    "prepare_trajectory",
    "recover_interrupted_commit",
)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_trajectory(
                args.checkpoint,
                args.trajectory,
                args.world_size,
            ),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    _main()
