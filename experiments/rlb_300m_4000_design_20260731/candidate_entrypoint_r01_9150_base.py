#!/usr/bin/env python3
"""Exact hash-gated R01 trainer wiring shared by R01-based successors."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimizer_design.rlb_r01_9150_archive import (
    R01Optimizer,
    R02AttentionOptimizer,
    verify_r01_9150_archive,
)
from training import transformer_lm_compare as trainer


CANDIDATES = {"r01_9150_parent": R01Optimizer}
_BASE_CONFIGURE_OPTIMIZER = trainer.configure_optimizer
_BASE_CLIP_OR_MEASURE_GRADIENTS = trainer.clip_or_measure_gradients
_ACTIVE_STRUCTURAL = None


class R01CompositeOptimizer(trainer.CompositeOptimizer):
    """Schema-checked structural, attention, and ordinary-AdamW ownership."""

    _ROLES = ("router", "attention", "ordinary_adamw")
    _SCHEMA = "r01_global_cross_layer_rlb_metric_9150_base_v1"

    def __init__(self, optimizers):
        super().__init__(optimizers)
        if len(self.optimizers) != len(self._ROLES):
            raise RuntimeError("R01 9150-base composite child inventory changed")

    def state_dict(self):
        return {
            "schema": self._SCHEMA,
            "roles": self._ROLES,
            "children": [child.state_dict() for child in self.optimizers],
        }

    def collective_state_dict(self):
        """Build the same schema while allowing a child to own rank shards.

        The exact R01 children remain replicated and use their ordinary
        ``state_dict`` path.  A systems-only child may instead expose
        ``consolidated_state_dict``; that method is invoked on every rank by
        the trainer's opt-in collective checkpoint protocol.
        """

        children = []
        for child in self.optimizers:
            consolidate = getattr(child, "consolidated_state_dict", None)
            children.append(
                consolidate() if callable(consolidate) else child.state_dict()
            )
        return {
            "schema": self._SCHEMA,
            "roles": self._ROLES,
            "children": children,
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or set(state_dict) != {"schema", "roles", "children"}
            or state_dict.get("schema") != self._SCHEMA
            or tuple(state_dict.get("roles", ())) != self._ROLES
            or len(state_dict.get("children", ())) != len(self.optimizers)
        ):
            raise RuntimeError("R01 9150-base composite checkpoint schema changed")
        for child, child_state in zip(self.optimizers, state_dict["children"]):
            child.load_state_dict(child_state)


def clip_candidate_gradients(model, grad_clip, capture_norm):
    """Run the unchanged trainer clip and publish its realized scalar."""
    result = _BASE_CLIP_OR_MEASURE_GRADIENTS(model, grad_clip, capture_norm)
    structural = _ACTIVE_STRUCTURAL
    if structural is not None:
        if float(grad_clip) != 1.0 or result[0] is None:
            raise RuntimeError("R01 9150-base requires the fixed clip-1.0 contract")
        structural.record_realized_clipping(result[0], grad_clip)
    return result


def _verify_frozen_cell(args):
    exact = {
        "layers": 18,
        "d_model": 1024,
        "heads": 16,
        "ffn_dim": 3072,
        "seq_len": 256,
        "batch_size": 8,
        "grad_accum": 4,
        "steps": 4000,
        "warmup_steps": 200,
        "lr": 3.0e-4,
        "min_lr": 3.0e-5,
        "weight_decay": 0.10,
        "beta1": 0.90,
        "beta2": 0.95,
        "eps": 1.0e-8,
        "grad_clip": 1.0,
        "muon_momentum": 0.95,
        "muon_ns_steps": 5,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "sam_rho": 0.0,
        "sam_adaptive": False,
    }
    mismatches = {
        key: (getattr(args, key, None), expected)
        for key, expected in exact.items()
        if getattr(args, key, None) != expected
    }
    if mismatches:
        raise RuntimeError(f"R01 9150 exact historical-M1 cell changed: {mismatches}")


def collect_r01_blocks(model, args):
    raw_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    groups = trainer.collect_rlb_optimizer_groups(model, args)
    by_layer = {int(group["layer_index"]): dict(group) for group in groups}
    if len(by_layer) != args.layers or len(raw_model.layers) != args.layers:
        raise RuntimeError("R01 9150-base requires one exact RLB block per layer")
    blocks = []
    for layer_index, block in enumerate(raw_model.layers):
        group = by_layer.get(layer_index)
        if group is None or group["mlp"] is not block.mlp:
            raise RuntimeError("R01 9150-base RLB ownership changed")
        if not isinstance(block.attn.qkv, torch.nn.Linear) or not isinstance(
            block.attn.out, torch.nn.Linear
        ):
            raise RuntimeError("R01 9150-base attention inventory changed")
        group.update({
            "block": block,
            "qkv_weight": block.attn.qkv.weight,
            "attn_out_weight": block.attn.out.weight,
        })
        blocks.append(group)
    return blocks


def configure_candidate_optimizer(model, args):
    global _ACTIVE_STRUCTURAL
    verify_r01_9150_archive()
    router_type = CANDIDATES.get(args.optimizer)
    if router_type is None:
        return _BASE_CONFIGURE_OPTIMIZER(model, args)
    _verify_frozen_cell(args)
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() != 4:
            raise RuntimeError("R01 9150-base requires exactly four ranks")
    elif next(model.parameters()).device.type != "cpu":
        raise RuntimeError("R01 9150-base GPU execution requires four-rank DDP")

    blocks = collect_r01_blocks(model, args)
    structural_ids = {
        id(parameter)
        for block in blocks
        for parameter in (
            block["qkv_weight"],
            block["attn_out_weight"],
            block["in_weight"],
            block["out_weight"],
        )
    }
    if len(structural_ids) != 4 * args.layers:
        raise RuntimeError("R01 9150-base structural inventory changed")

    muon_named = []
    adam_decay = []
    adam_no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in structural_ids:
            continue
        no_decay = trainer.is_no_decay_parameter(name, parameter)
        tied_embedding = trainer.is_tied_embedding_parameter_name(name)
        if parameter.dim() == 2 and not no_decay and not tied_embedding:
            muon_named.append((name, parameter))
        elif no_decay or tied_embedding:
            adam_no_decay.append(parameter)
        else:
            adam_decay.append(parameter)
    if muon_named:
        raise RuntimeError("R01 9150-base ordinary Muon inventory changed")

    router = router_type(
        blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
    )
    attention = R02AttentionOptimizer(
        blocks,
        router,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
        adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    adam_groups = []
    if adam_decay:
        adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    if not adam_groups:
        raise RuntimeError("R01 9150-base found no ordinary AdamW parameters")
    adamw = torch.optim.AdamW(
        adam_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    _ACTIVE_STRUCTURAL = router
    return R01CompositeOptimizer((router, attention, adamw))


def main():
    verify_r01_9150_archive()
    trainer.RATIONAL_SPECIFIC_OPTIMIZERS.update(CANDIDATES)
    trainer.RLB_MATRIX_SYNC_OPTIMIZERS.update(CANDIDATES)
    trainer.RLB_COEFFICIENT_SYNC_OPTIMIZERS.update(CANDIDATES)
    trainer.ACTIVE_OPTIMIZERS = sorted(
        set(trainer.ACTIVE_OPTIMIZERS) | set(CANDIDATES)
    )
    trainer.configure_optimizer = configure_candidate_optimizer
    trainer.clip_or_measure_gradients = clip_candidate_gradients
    trainer.main()


if __name__ == "__main__":
    main()
