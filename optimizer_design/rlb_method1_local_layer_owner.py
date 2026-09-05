"""Fast numerical Method1 approximation with stable local layer ownership.

This is the first integrated use of the layer-owner communication boundary.
Each of four ranks runs the unchanged Method1 recursive router and unchanged
local-fused attention equations on its stable ``layer % 4`` subset.  Internal
statistics are deliberately local to that owner, so the global 18-layer and
four-rank score models become four 4/5-layer, one-rank score models.  This is
an explicit numerical approximation, not an exact distributed rewrite.

After the local structural transaction, its complete parameter delta
(including the unchanged decoupled weight decay) is quantized to BF16,
reconstructed in global layer order, and applied identically to the still-FP32
parameters on every DDP replica.  Newton--Schulz's polynomial and five steps,
all local RLB equations, LR, WD, momentum, and cadence remain unchanged.

The approximation exists to test the high-speed/retained-quality frontier.
It cannot inherit Method1 quality and requires a fresh 4,000-step trajectory.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading

import torch
import torch.distributed as dist

from .rlb_layer_owner_collectives import (
    FAMILY_ID as COLLECTIVE_FAMILY_ID,
    gather_quantized_owner_direction_families,
    owner_layer_lists,
)
from .rlb_method12_macro_numerics import (
    Method1LocalFusedAttentionOptimizer,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_recursive_local_layer_owner_bf16_delta_v1"
_WORLD_SIZE = 4
_LAYERS = 18
_LOCAL_REDUCTION_LOCK = threading.RLock()


def _require_four_ranks() -> int:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Method1 local layer ownership requires distributed")
    if int(dist.get_world_size()) != _WORLD_SIZE:
        raise RuntimeError("Method1 local layer ownership requires four ranks")
    return int(dist.get_rank())


@contextmanager
def _owner_local_reductions():
    """Make the inner subset optimizer a mathematical one-rank transaction."""

    original_all_reduce = dist.all_reduce
    original_world_size = dist.get_world_size

    def local_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if group is not None:
            raise RuntimeError("owner-local inner optimizer supplied a process group")
        if async_op:
            raise RuntimeError("owner-local inner optimizer requested async reduction")
        # SUM/MIN/MAX are all identities in a singleton mathematical world.
        return None

    def local_world_size(group=None):
        if group is not None:
            raise RuntimeError("owner-local inner optimizer queried a process group")
        return 1

    with _LOCAL_REDUCTION_LOCK:
        if dist.all_reduce is not original_all_reduce:
            raise RuntimeError("distributed all_reduce was already patched")
        if dist.get_world_size is not original_world_size:
            raise RuntimeError("distributed get_world_size was already patched")
        dist.all_reduce = local_all_reduce
        dist.get_world_size = local_world_size
        try:
            yield
        finally:
            dist.all_reduce = original_all_reduce
            dist.get_world_size = original_world_size


class Method1LocalLayerOwnerComposite:
    """Own local Method1 layers and publish one common quantized FP32 delta."""

    _ROLES = ("local_router", "local_attention", "ordinary_adamw")
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
            raise ValueError("Method1 local layer owner requires 18 blocks")
        inventory = owner_layer_lists()[rank]
        self.owned_layers = tuple(int(layer) for layer in inventory)
        # Historical attention validates that its private inventory is
        # contiguous.  Remap only the optimizer-local index while retaining
        # the exact model modules and parameter objects of the global layers.
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
        self.router = Method1RecursiveInverseRouter(self.local_blocks, **kwargs)
        self.attention = Method1LocalFusedAttentionOptimizer(
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
            raise RuntimeError("Method1 local owner structural inventory changed")
        if any(value.dtype != torch.float32 for family in self._structural_families for value in family):
            raise RuntimeError("Method1 local owner requires FP32 parameters")
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)
        # Every rank holds the complete DDP model, while only one rank advances
        # each structural layer before the common delta is gathered.  Publish
        # the non-owned tensors to the trainer's LR/WD fairness and scheduler
        # inventory without giving them to a second stepping optimizer.
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
            raise RuntimeError("Method1 local owner synchronized-only inventory changed")
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

    def lr_wd_fairness_audit(self):
        return {
            "local_layer_owner_lr_scale": 1.0,
            "local_layer_owner_weight_decay_scale": 1.0,
            "bf16_delta_lr_scale": 1.0,
            "bf16_delta_weight_decay_scale": 1.0,
        }

    def record_realized_clipping(self, preclip_norm, max_norm):
        return self.router.record_realized_clipping(preclip_norm, max_norm)

    def set_telemetry_capture(self, enabled: bool = True):
        self.router.set_telemetry_capture(enabled)
        self.attention.set_telemetry_capture(enabled)

    def telemetry(self):
        result = dict(self.router.telemetry())
        result.update(self.attention.telemetry())
        result.update({
            "rlb_layer_owner_rank": self.rank,
            "rlb_layer_owner_local_layers": len(self.owned_layers),
            "rlb_layer_owner_wire_elements": self._last_wire_elements,
            "rlb_layer_owner_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_delta_max_abs": self._last_delta_max_abs,
        })
        return result

    def zero_grad(self, set_to_none: bool = True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        for family in self._structural_families:
            for parameter in family:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        local_families = tuple(
            tuple(family[layer] for layer in self.owned_layers)
            for family in self._structural_families
        )
        before = tuple(
            tuple(parameter.detach().clone() for parameter in family)
            for family in local_families
        )
        with _owner_local_reductions():
            self.router.step()
            self.attention.step()
        local_deltas = tuple(
            torch.stack([
                parameter.detach() - original
                for parameter, original in zip(family, originals)
            ])
            for family, originals in zip(local_families, before)
        )
        # Owners must consume the same quantized delta as non-owners.  Restore
        # their pre-step parameters before the common update is applied.
        for family, originals in zip(local_families, before):
            for parameter, original in zip(family, originals):
                parameter.copy_(original)
        full_deltas = gather_quantized_owner_direction_families(local_deltas)
        for parameters, delta in zip(self._structural_families, full_deltas):
            for layer, parameter in enumerate(parameters):
                parameter.add_(delta[layer])
        self.adamw.step()
        local_maximum = torch.stack([
            delta.abs().amax() for delta in full_deltas
        ]).amax()
        self._last_delta_max_abs = float(local_maximum.item())
        self._last_wire_elements = sum(delta.numel() for delta in full_deltas)
        self._last_wire_bytes = 2 * self._last_wire_elements

    def state_dict(self):
        # This is deliberately a rank-local shard.  Production checkpointing
        # must call a future collective consolidation boundary; silently
        # labeling rank zero's shard as complete would be incorrect.
        return {
            "schema": self._SCHEMA,
            "rank": self.rank,
            "owned_layers": self.owned_layers,
            "collective_family_id": COLLECTIVE_FAMILY_ID,
            "children": [child.state_dict() for child in self.optimizers],
            "rank_local_incomplete": True,
        }

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("schema") != self._SCHEMA
            or int(state_dict.get("rank", -1)) != self.rank
            or tuple(state_dict.get("owned_layers", ())) != self.owned_layers
            or state_dict.get("rank_local_incomplete") is not True
            or len(state_dict.get("children", ())) != len(self.optimizers)
        ):
            raise RuntimeError("Method1 local owner checkpoint shard changed")
        for child, child_state in zip(self.optimizers, state_dict["children"]):
            child.load_state_dict(child_state)

    def execution_report(self):
        return {
            "family_id": FAMILY_ID,
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


__all__ = ("FAMILY_ID", "Method1LocalLayerOwnerComposite")
