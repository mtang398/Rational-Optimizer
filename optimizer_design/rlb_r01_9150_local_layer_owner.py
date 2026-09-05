"""Original 9,150-tested R01 with stable four-rank layer ownership."""

from __future__ import annotations

import torch

from .rlb_layer_owner_collectives import (
    FAMILY_ID as COLLECTIVE_FAMILY_ID,
    owner_layer_lists,
)
from .rlb_method1_local_layer_owner import (
    Method1LocalLayerOwnerComposite,
    _LAYERS,
    _require_four_ranks,
)
from .rlb_r01_9150_archive import R01Optimizer, R02AttentionOptimizer


FAMILY_ID = "r01_9150_original_local_layer_owner_bf16_delta_v1"


class R019150LocalLayerOwnerComposite(Method1LocalLayerOwnerComposite):
    """Run the original R01 equations on four or five complete local layers."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(
        self,
        blocks,
        adamw,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
        adjust_lr_fn: str,
    ):
        rank = _require_four_ranks()
        self.rank = rank
        self.all_blocks = list(blocks)
        if len(self.all_blocks) != _LAYERS:
            raise ValueError("original R01 local owner requires 18 blocks")
        inventory = owner_layer_lists()[rank]
        self.owned_layers = tuple(int(layer) for layer in inventory)
        self.local_blocks = [
            dict(self.all_blocks[layer], layer_index=local_index)
            for local_index, layer in enumerate(self.owned_layers)
        ]
        kwargs = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "ns_steps": int(ns_steps),
            "beta2": float(beta2),
            "eps": float(eps),
        }
        self.router = R01Optimizer(self.local_blocks, **kwargs)
        self.attention = R02AttentionOptimizer(
            self.local_blocks,
            self.router,
            adjust_lr_fn=adjust_lr_fn,
            **kwargs,
        )
        self.adamw = adamw
        self.optimizers = [self.router, self.attention, self.adamw]
        self._structural_families = (
            tuple(block["in_weight"] for block in self.all_blocks),
            tuple(block["out_weight"] for block in self.all_blocks),
            tuple(block["qkv_weight"] for block in self.all_blocks),
            tuple(block["attn_out_weight"] for block in self.all_blocks),
        )
        if len({id(value) for family in self._structural_families for value in family}) != (
            4 * _LAYERS
        ):
            raise RuntimeError("original R01 local owner structural inventory changed")
        if any(
            value.dtype != torch.float32
            for family in self._structural_families
            for value in family
        ):
            raise RuntimeError("original R01 local owner requires FP32 parameters")
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)
        stepped_ids = {
            id(parameter)
            for group in self.param_groups
            for parameter in group["params"]
        }
        synchronized_only = [
            parameter
            for family in self._structural_families
            for parameter in family
            if id(parameter) not in stepped_ids
        ]
        if len(synchronized_only) != 4 * (_LAYERS - len(self.owned_layers)):
            raise RuntimeError("original R01 synchronized-only inventory changed")
        self.param_groups.append({
            "params": synchronized_only,
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
            "role": "owner_synchronized_only",
        })
        self._last_wire_elements = 0
        self._last_wire_bytes = 0
        self._last_delta_max_abs = 0.0

    def execution_report(self):
        return {
            "family_id": FAMILY_ID,
            "scientific_parent": "exact_hash_gated_r01_9150_archive",
            "collective_family_id": COLLECTIVE_FAMILY_ID,
            "ownership": "layer_mod_4",
            "owned_layer_counts": tuple(map(len, owner_layer_lists())),
            "inner_statistics": "owner_rank_local",
            "global_cross_layer_model": "approximated_by_four_owner_local_models",
            "direction_wire": "bf16_complete_parameter_delta",
            "fp32_parameters_quantized": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "lr_or_wd_multiplier_changed": False,
            "fresh_quality_required": True,
            "rank_local_checkpoint_requires_collective_consolidation": True,
        }


__all__ = ("FAMILY_ID", "R019150LocalLayerOwnerComposite")
