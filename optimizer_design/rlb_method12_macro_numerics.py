"""Macro numerical execution for the qualified fast Method1/Method2 bases.

This module deliberately combines several execution changes into one branch;
none is promoted or timed as a standalone micro-optimization:

* each complete NS5 attention batch is partitioned across the four ranks and
  reconstructed in canonical batch order with one all-gather;
* the ordinary adaptive, family-route, and chord tensor programs are fused;
* the routers use the already audited grouped metric reductions.

The Newton--Schulz polynomial, iteration count, source tensors, refresh
cadence, optimizer state recurrences, LR, and WD are unchanged.  Shaping a
batched kernel differently and fusing elementwise programs can change floating
point association, so this is a numerical execution branch and never inherits
quality.  A faster realization requires a fresh complete 4,000-step run.
"""

from __future__ import annotations

import threading

import torch
import torch.distributed as dist

from . import rlb_r07_frame_878462_lean_attention as _lean_module
from .rlb_method1_grouped_collectives import (
    Method1GroupedCollectiveOptimizer,
)
from .rlb_other_grouped_collectives import Method2GroupedQualifiedRouter
from .rlb_r07_frame_878462_lean_attention import (
    Method1LeanAttentionOptimizer,
)
from .rlb_r07_paired_postpolar_881693_lean import (
    Method2LeanAttentionOptimizer,
)


FAMILY_ID = "method12_qualified_macro_numerics_v1"
LOCAL_FUSION_FAMILY_ID = "method12_qualified_local_attention_fusion_v1"
_WORLD_SIZE = 4
_PATCH_LOCK = threading.RLock()
_EXACT_ZERO_POWER = _lean_module._batched_zero_power


def _fused_adaptive_source(
    gradients: torch.Tensor,
    momenta: torch.Tensor,
    rows: torch.Tensor,
    columns: torch.Tensor,
    beta2: float,
    correction: torch.Tensor,
    adaptive_eps: float,
) -> torch.Tensor:
    squared = gradients.square()
    rows.mul_(beta2).add_(squared.sum(dim=-1), alpha=1.0 - beta2)
    columns.mul_(beta2).add_(squared.sum(dim=-2), alpha=1.0 - beta2)
    row_total = rows.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(rows.dtype).tiny
    )
    variance = rows[:, :, None] * columns[:, None, :] / row_total[:, :, None]
    variance.div_(correction)
    inverse_root = torch.reciprocal(variance.sqrt() + adaptive_eps)
    adaptive = momenta * inverse_root
    momentum_norm = torch.linalg.vector_norm(
        momenta, dim=(-2, -1), keepdim=True
    )
    adaptive_norm = torch.linalg.vector_norm(
        adaptive, dim=(-2, -1), keepdim=True
    )
    return adaptive * (
        momentum_norm / adaptive_norm.clamp_min(torch.finfo(momenta.dtype).tiny)
    )


def _fused_family_route(
    parent: torch.Tensor,
    momentum: torch.Tensor,
    participation: torch.Tensor,
    groups: int,
    width: int,
) -> torch.Tensor:
    layers, hidden, external = parent.shape
    shape = (layers, groups, width, external)
    p = parent.view(shape)
    m = momentum.view(shape)
    dims = (-2, -1)
    tiny = torch.finfo(parent.dtype).tiny
    p_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid_norm = (p_norm > 0.0) & (sign_norm > 0.0)
    coordinate = sign * (p_norm / sign_norm.clamp_min(tiny))
    coefficient = participation[:, :, None, None]
    source = torch.sqrt(coefficient) * p
    source.add_(
        torch.sqrt((1.0 - coefficient).clamp_min(0.0)) * coordinate
    )
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    provisional = source * (p_norm / source_norm.clamp_min(tiny))
    provisional = torch.where(coefficient == 1.0, p, provisional)
    finite = valid_norm & torch.isfinite(provisional).all(dim=dims, keepdim=True)
    return torch.where(finite, provisional, p).reshape_as(parent)


def _fused_chord(
    u6: torch.Tensor,
    u5: torch.Tensor,
    literal_parent: torch.Tensor,
    congruence: torch.Tensor,
    groups: int,
    width: int,
) -> torch.Tensor:
    layers, _, external = u6.shape
    shape = (layers, groups, width, external)
    d6 = u6.reshape(shape)
    d5 = u5.reshape(shape)
    target = literal_parent.reshape(shape)
    dims = (-2, -1)
    tiny = torch.finfo(u6.dtype).tiny
    alpha = congruence[:, None, None, None]
    delta = torch.sqrt((1.0 - alpha.square()).clamp_min(0.0))
    target_norm = torch.linalg.vector_norm(target, dim=dims, keepdim=True)
    d6_norm = torch.linalg.vector_norm(d6, dim=dims, keepdim=True)
    d5_norm = torch.linalg.vector_norm(d5, dim=dims, keepdim=True)
    d6 = d6 * (target_norm / d6_norm.clamp_min(tiny))
    d5 = d5 * (target_norm / d5_norm.clamp_min(tiny))
    source = alpha * d6 + delta * d5
    source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
    mixed = source * (target_norm / source_norm.clamp_min(tiny))
    direction = torch.where(alpha == 1.0, d6, mixed)
    direction = torch.where(alpha == 0.0, d5, direction)
    return direction.reshape_as(u6)


_compiled_adaptive_source = torch.compile(
    _fused_adaptive_source, fullgraph=True, dynamic=False
)
_compiled_family_route = torch.compile(
    _fused_family_route, fullgraph=True, dynamic=False
)
_compiled_chord = torch.compile(_fused_chord, fullgraph=True, dynamic=False)


def _rank_slice(total: int, rank: int) -> tuple[int, int, int]:
    base, extra = divmod(int(total), _WORLD_SIZE)
    count = base + int(rank < extra)
    start = rank * base + min(rank, extra)
    return start, start + count, base + int(extra > 0)


def _sharded_zero_power(source: torch.Tensor, steps: int) -> torch.Tensor:
    """Evaluate the unchanged per-matrix NS5 map on rank-owned batches."""

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or source.device.type != "cuda"
    ):
        return _EXACT_ZERO_POWER(source, steps)
    if int(dist.get_world_size()) != _WORLD_SIZE:
        raise RuntimeError("macro NS5 sharding requires exactly four ranks")
    if source.ndim != 3 or int(source.shape[0]) < _WORLD_SIZE:
        raise RuntimeError("macro NS5 source inventory changed")
    if int(steps) != 5:
        raise RuntimeError("macro NS5 iteration count changed")

    rank = int(dist.get_rank())
    total = int(source.shape[0])
    start, stop, maximum = _rank_slice(total, rank)
    local = _EXACT_ZERO_POWER(source[start:stop], steps)

    # The historical tall-matrix implementation returns a transpose view.
    # Gather its contiguous wire orientation and restore that public layout.
    tall = int(source.shape[-2]) > int(source.shape[-1])
    wire = local.transpose(-2, -1) if tall else local
    wire = wire.contiguous()
    send = torch.zeros(
        (maximum, *wire.shape[1:]), device=wire.device, dtype=wire.dtype
    )
    send[: wire.shape[0]].copy_(wire)
    gathered = torch.empty(
        (_WORLD_SIZE * maximum, *wire.shape[1:]),
        device=wire.device,
        dtype=wire.dtype,
    )
    dist.all_gather_into_tensor(gathered, send)

    pieces = []
    gathered = gathered.view(_WORLD_SIZE, maximum, *wire.shape[1:])
    for owner in range(_WORLD_SIZE):
        owner_start, owner_stop, _ = _rank_slice(total, owner)
        pieces.append(gathered[owner, : owner_stop - owner_start])
    canonical = torch.cat(pieces, dim=0)
    if int(canonical.shape[0]) != total:
        raise RuntimeError("macro NS5 reconstruction inventory changed")
    return canonical.transpose(-2, -1) if tall else canonical


class _MacroAttentionMixin:
    """Fuse ordinary tensor programs and shard their unchanged NS5 calls."""

    _macro_distribute_ns5 = True
    _macro_family_id = FAMILY_ID

    def __init__(self, *args, **kwargs):
        self._macro_sharded_ns5_calls = 0
        super().__init__(*args, **kwargs)
        self.param_groups[0]["rlb_macro_numerics_family_id"] = (
            self._macro_family_id
        )

    def _ordinary_adaptive_source(self, role, gradients, momenta, step):
        if not gradients.is_cuda:
            return super()._ordinary_adaptive_source(
                role, gradients, momenta, step
            )
        anchor_state = self.state[self.role_parameters[role][0]]
        row_key = f"r04_{role}_row_second_moment"
        column_key = f"r04_{role}_column_second_moment"
        rows = anchor_state.get(row_key)
        columns = anchor_state.get(column_key)
        row_shape = gradients.shape[:-1]
        column_shape = (gradients.shape[0], gradients.shape[-1])
        if rows is None:
            rows = torch.zeros(
                row_shape, device=gradients.device, dtype=torch.float32
            )
            columns = torch.zeros(
                column_shape, device=gradients.device, dtype=torch.float32
            )
            anchor_state[row_key] = rows
            anchor_state[column_key] = columns
        if rows.shape != row_shape or columns is None or columns.shape != column_shape:
            raise RuntimeError("macro adaptive moment inventory changed")
        # Keep the changing bias correction as tensor data.  Passing its
        # Python value to torch.compile specializes one graph per step and
        # exhausts Dynamo's cache before both attention roles are observed.
        correction = rows.new_tensor(
            1.0 - float(self.beta2) ** int(step)
        )
        return _compiled_adaptive_source(
            gradients,
            momenta,
            rows,
            columns,
            float(self.beta2),
            correction,
            float(self.adaptive_eps),
        )

    def _ordinary_family_route(
        self, parent, momentum, participation, *, groups, width
    ):
        if not parent.is_cuda:
            return super()._ordinary_family_route(
                parent,
                momentum,
                participation,
                groups=groups,
                width=width,
            )
        if parent.shape != momentum.shape or parent.ndim != 3:
            raise RuntimeError("macro family-route tensor inventory changed")
        layers, hidden, _ = parent.shape
        if hidden != int(groups) * int(width) or participation.shape != (
            layers,
            int(groups),
        ):
            raise RuntimeError("macro family-route group inventory changed")
        return _compiled_family_route(
            parent, momentum, participation, int(groups), int(width)
        )

    def _ordinary_chord(
        self,
        u6,
        u5,
        literal_parent,
        momentum,
        congruence,
        *,
        groups,
        width,
    ):
        if not u6.is_cuda:
            return super()._ordinary_chord(
                u6,
                u5,
                literal_parent,
                momentum,
                congruence,
                groups=groups,
                width=width,
            )
        if not (u6.shape == u5.shape == literal_parent.shape == momentum.shape):
            raise RuntimeError("macro chord tensor inventory changed")
        if u6.ndim != 3 or congruence.shape != (u6.shape[0],):
            raise RuntimeError("macro chord coefficient inventory changed")
        return _compiled_chord(
            u6,
            u5,
            literal_parent,
            congruence,
            int(groups),
            int(width),
        )

    def _ordinary_r02_step(self, closure=None):
        if not bool(self._macro_distribute_ns5):
            return super()._ordinary_r02_step(closure)
        global _EXACT_ZERO_POWER
        with _PATCH_LOCK:
            if _lean_module._batched_zero_power is not _EXACT_ZERO_POWER:
                raise RuntimeError("macro attention zero-power kernel was patched")

            def counted(source, steps):
                self._macro_sharded_ns5_calls += 1
                return _sharded_zero_power(source, steps)

            _lean_module._batched_zero_power = counted
            try:
                return super()._ordinary_r02_step(closure)
            finally:
                _lean_module._batched_zero_power = _EXACT_ZERO_POWER

    def macro_numerics_runtime_report(self):
        return {
            "family_id": self._macro_family_id,
            "sharded_ns5_calls": int(self._macro_sharded_ns5_calls),
            "world_size": _WORLD_SIZE,
            "ns5_iterations": int(self.ns_steps),
            "ns5_polynomial_changed": False,
            "refresh_cadence_changed": False,
            "lr_or_wd_changed": False,
            "fused_adaptive_route_chord": True,
            "distributed_ns5": bool(self._macro_distribute_ns5),
            "floating_point_association_may_change": True,
            "fresh_quality_trajectory_required": True,
        }


class Method1MacroAttentionOptimizer(
    _MacroAttentionMixin, Method1LeanAttentionOptimizer
):
    checkpoint_schema = FAMILY_ID + "_method1_attention"


class Method2MacroAttentionOptimizer(
    _MacroAttentionMixin, Method2LeanAttentionOptimizer
):
    checkpoint_schema = FAMILY_ID + "_method2_attention"


class Method1LocalFusedAttentionOptimizer(
    _MacroAttentionMixin, Method1LeanAttentionOptimizer
):
    """Fused ordinary attention programs without output communication."""

    _macro_distribute_ns5 = False
    _macro_family_id = LOCAL_FUSION_FAMILY_ID
    checkpoint_schema = LOCAL_FUSION_FAMILY_ID + "_method1_attention"


class Method2LocalFusedAttentionOptimizer(
    _MacroAttentionMixin, Method2LeanAttentionOptimizer
):
    """Fused ordinary attention programs without output communication."""

    _macro_distribute_ns5 = False
    _macro_family_id = LOCAL_FUSION_FAMILY_ID
    checkpoint_schema = LOCAL_FUSION_FAMILY_ID + "_method2_attention"


Method1MacroRouter = Method1GroupedCollectiveOptimizer
Method2MacroRouter = Method2GroupedQualifiedRouter


__all__ = (
    "FAMILY_ID",
    "LOCAL_FUSION_FAMILY_ID",
    "Method1LocalFusedAttentionOptimizer",
    "Method1MacroAttentionOptimizer",
    "Method1MacroRouter",
    "Method2LocalFusedAttentionOptimizer",
    "Method2MacroAttentionOptimizer",
    "Method2MacroRouter",
    "_fused_adaptive_source",
    "_fused_chord",
    "_fused_family_route",
    "_rank_slice",
    "_sharded_zero_power",
)
