#!/usr/bin/env python3
"""Matched trainer wiring for owner-free single-direction group-Muon methods."""

from __future__ import annotations

import torch
import torch.distributed as dist

from experiments.rlb_300m_4000_design_20260731 import (
    candidate_entrypoint_r01_9150_base as historical_base,
)
from training import transformer_lm_compare as trainer


CANDIDATES = {}
_BASE_CONFIGURE_OPTIMIZER = trainer.configure_optimizer
_BASE_CLIP_OR_MEASURE_GRADIENTS = trainer.clip_or_measure_gradients
_ACTIVE_STRUCTURAL = None


class ScalableCompositeOptimizer(trainer.CompositeOptimizer):
    """Schema-checked ownership without a special attention optimizer."""

    _ROLES = ("loss_aware_group_muon", "ordinary_muon", "ordinary_adamw")
    _SCHEMA = "owner_free_single_direction_group_muon_composite_v2"

    def __init__(self, optimizers):
        super().__init__(optimizers)
        if len(self.optimizers) != len(self._ROLES):
            raise RuntimeError("scalable group-Muon child inventory changed")

    def state_dict(self):
        return {
            "schema": self._SCHEMA,
            "roles": self._ROLES,
            "children": [child.state_dict() for child in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or set(state_dict) != {"schema", "roles", "children"}
            or state_dict.get("schema") != self._SCHEMA
            or tuple(state_dict.get("roles", ())) != self._ROLES
            or len(state_dict.get("children", ())) != len(self.optimizers)
        ):
            raise RuntimeError("scalable group-Muon composite checkpoint changed")
        for child, child_state in zip(self.optimizers, state_dict["children"]):
            child.load_state_dict(child_state)


def clip_candidate_gradients(model, grad_clip, capture_norm):
    result = _BASE_CLIP_OR_MEASURE_GRADIENTS(model, grad_clip, capture_norm)
    structural = _ACTIVE_STRUCTURAL
    if structural is not None:
        if float(grad_clip) != 1.0 or result[0] is None:
            raise RuntimeError("scalable group Muon requires clip-1.0")
        structural.record_realized_clipping(result[0], grad_clip)
    return result


def _partition_parameters(model, blocks):
    structural_ids = {
        id(parameter)
        for block in blocks
        for parameter in (block["in_weight"], block["out_weight"])
    }
    if len(structural_ids) != 2 * len(blocks):
        raise RuntimeError("scalable group-Muon matrix ownership overlaps")

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
    return structural_ids, muon_named, adam_decay, adam_no_decay


def configure_candidate_optimizer(model, args):
    global _ACTIVE_STRUCTURAL
    router_type = CANDIDATES.get(args.optimizer)
    if router_type is None:
        return _BASE_CONFIGURE_OPTIMIZER(model, args)
    historical_base._verify_frozen_cell(args)
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() != 4:
            raise RuntimeError("matched endpoint cell requires exactly four DDP ranks")
    elif next(model.parameters()).device.type != "cpu":
        raise RuntimeError("matched GPU cell requires four-rank DDP")

    blocks = historical_base.collect_r01_blocks(model, args)
    _, muon_named, adam_decay, adam_no_decay = _partition_parameters(
        model, blocks
    )
    attention_ids = {
        id(parameter)
        for block in blocks
        for parameter in (block["qkv_weight"], block["attn_out_weight"])
    }
    if {id(parameter) for _, parameter in muon_named} != attention_ids:
        raise RuntimeError(
            "ordinary Muon inventory must be exactly the two attention matrices per layer"
        )

    router = router_type(
        blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
    )
    ordinary_muon = torch.optim.Muon(
        muon_named,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    adam_groups = []
    if adam_decay:
        adam_groups.append({
            "params": adam_decay,
            "weight_decay": args.weight_decay,
        })
    if adam_no_decay:
        adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
    if not adam_groups:
        raise RuntimeError("scalable group Muon found no ordinary AdamW parameters")
    adamw = torch.optim.AdamW(
        adam_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    _ACTIVE_STRUCTURAL = router
    return ScalableCompositeOptimizer((router, ordinary_muon, adamw))


def register_and_run():
    trainer.RATIONAL_SPECIFIC_OPTIMIZERS.update(CANDIDATES)
    trainer.RLB_MATRIX_SYNC_OPTIMIZERS.update(CANDIDATES)
    trainer.RLB_COEFFICIENT_SYNC_OPTIMIZERS.update(CANDIDATES)
    trainer.ACTIVE_OPTIMIZERS = sorted(
        set(trainer.ACTIVE_OPTIMIZERS) | set(CANDIDATES)
    )
    trainer.configure_optimizer = configure_candidate_optimizer
    trainer.clip_or_measure_gradients = clip_candidate_gradients
    trainer.main()


__all__ = (
    "CANDIDATES",
    "ScalableCompositeOptimizer",
    "clip_candidate_gradients",
    "configure_candidate_optimizer",
    "register_and_run",
)

