#!/usr/bin/env python3
"""Locked DCLM entrypoint for batched four-role response homotopy Muon."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rlb_300m_4000_design_20260731 import (  # noqa: E402
    candidate_entrypoint_r01_9150_base as historical_base,
    candidate_entrypoint_scalable_group_muon_base as scalable_base,
)
from optimizer_design.rlb_loss_weighted_four_role_response_homotopy_batched_muon import (  # noqa: E402
    FAMILY_ID,
    BatchedFourRoleResponseHomotopyAttentionOptimizer,
    BatchedFourRoleResponseHomotopyRouter,
)
from training import transformer_lm_compare as trainer  # noqa: E402


OPTIMIZER_ID = FAMILY_ID
CANDIDATES = {OPTIMIZER_ID: BatchedFourRoleResponseHomotopyRouter}
R01CompositeOptimizer = None
_ACTIVE_STRUCTURAL = None
_BASE_CONFIGURE_OPTIMIZER = scalable_base._BASE_CONFIGURE_OPTIMIZER
_BASE_CLIP_OR_MEASURE_GRADIENTS = scalable_base._BASE_CLIP_OR_MEASURE_GRADIENTS


class BatchedFourRoleResponseHomotopyCompositeOptimizer(
    trainer.CompositeOptimizer
):
    _ROLES = (
        "batched_response_homotopy_router",
        "batched_response_homotopy_attention",
        "ordinary_adamw",
    )
    _SCHEMA = "owner_free_four_role_response_homotopy_batched_composite_v2"

    def __init__(self, optimizers):
        super().__init__(optimizers)
        if len(self.optimizers) != len(self._ROLES):
            raise RuntimeError("batched response composite inventory changed")

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
            raise RuntimeError("batched response composite checkpoint changed")
        for child, child_state in zip(self.optimizers, state_dict["children"]):
            child.load_state_dict(child_state)


R01CompositeOptimizer = BatchedFourRoleResponseHomotopyCompositeOptimizer


def clip_candidate_gradients(model, grad_clip, capture_norm):
    result = _BASE_CLIP_OR_MEASURE_GRADIENTS(model, grad_clip, capture_norm)
    if _ACTIVE_STRUCTURAL is not None:
        if float(grad_clip) != 1.0 or result[0] is None:
            raise RuntimeError("batched response homotopy requires clip-1.0")
        _ACTIVE_STRUCTURAL.record_realized_clipping(result[0], grad_clip)
    return result


def configure_candidate_optimizer(model, args):
    global _ACTIVE_STRUCTURAL
    if args.optimizer != OPTIMIZER_ID:
        return _BASE_CONFIGURE_OPTIMIZER(model, args)
    historical_base._verify_frozen_cell(args)
    if dist.is_available() and dist.is_initialized():
        if dist.get_world_size() != 4:
            raise RuntimeError("matched batched response endpoint requires four DDP ranks")
    elif next(model.parameters()).device.type != "cpu":
        raise RuntimeError("matched batched response GPU cell requires four DDP ranks")
    blocks = historical_base.collect_r01_blocks(model, args)
    _, attention_named, adam_decay, adam_no_decay = scalable_base._partition_parameters(
        model, blocks
    )
    attention_ids = {
        id(parameter)
        for block in blocks
        for parameter in (block["qkv_weight"], block["attn_out_weight"])
    }
    if {id(parameter) for _, parameter in attention_named} != attention_ids:
        raise RuntimeError("batched response attention inventory changed")
    router = BatchedFourRoleResponseHomotopyRouter(
        blocks,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        beta2=args.beta2,
        eps=args.eps,
    )
    attention = BatchedFourRoleResponseHomotopyAttentionOptimizer(
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
        raise RuntimeError("batched response candidate found no AdamW parameters")
    adamw = torch.optim.AdamW(
        adam_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    _ACTIVE_STRUCTURAL = router
    return BatchedFourRoleResponseHomotopyCompositeOptimizer(
        (router, attention, adamw)
    )


def main():
    trainer.RATIONAL_SPECIFIC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.RLB_MATRIX_SYNC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.RLB_COEFFICIENT_SYNC_OPTIMIZERS.add(OPTIMIZER_ID)
    trainer.ACTIVE_OPTIMIZERS = sorted(
        set(trainer.ACTIVE_OPTIMIZERS) | {OPTIMIZER_ID}
    )
    trainer.configure_optimizer = configure_candidate_optimizer
    trainer.clip_or_measure_gradients = clip_candidate_gradients
    trainer.main()


if __name__ == "__main__":
    main()
