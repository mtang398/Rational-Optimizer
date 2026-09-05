"""Owner-sharded fused reductions for metric4 stale allocation.

This execution variant leaves the metric4/response/endpoint/attention method
unchanged.  On stale-allocation transitions it computes the five per-group
scalars (incoming/outgoing exact action, incoming/outgoing Nesterov action,
and paired budget) with the previously CUDA-gated Phase-3 Triton reduction.
Each of four DDP ranks owns 81 complete groups and all-gathers only 6,480
bytes.  Floating-point reduction order changes, so this is numerical rather
than bitwise equivalence and needs timing plus quality gates.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from ._r01_9150_full_shard_phase1.core import _all_gather_owned
from ._r01_9150_full_shard_phase3.selector_reduction import (
    fused_selector_layer_reductions,
)
from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_parent_endpoint_metric4_onepass import (
    R01StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowOptimizer,
    R02StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
FUSED_ALLOCATION_FAMILY_ID = "r01_metric4_owner_fused_stale_allocation_v1"
_LAYERS = 18
_GROUPS = 18
_BLOCKS = _LAYERS * _GROUPS
_WORLD_SIZE = 4


class _FusedStaleAllocationMixin:
    checkpoint_schema = "r01_metric4_owner_fused_stale_allocation_v1"

    def __init__(self, pairs, **kwargs):
        self._fused_allocation_backend = None
        self._fused_allocation_packet_bytes = 0
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["r01_fused_allocation_family_id"] = (
            FUSED_ALLOCATION_FAMILY_ID
        )

    @staticmethod
    def _role_blocks(value, *, outgoing, groups, width, external):
        if outgoing:
            value = value.transpose(-2, -1)
        return value.reshape(groups, width, external)

    def _fused_stale_allocation_packet(
        self, incoming_blocks, outgoing_blocks
    ):
        if not (
            dist.is_available()
            and dist.is_initialized()
            and int(dist.get_world_size()) == _WORLD_SIZE
        ):
            return None
        rank = int(dist.get_rank())
        local_packets = []
        backends = set()
        for layer, (incoming, outgoing) in enumerate(
            zip(self.incoming, self.outgoing)
        ):
            incoming_buffer = self.state[incoming].get("momentum_buffer")
            outgoing_buffer = self.state[outgoing].get("momentum_buffer")
            if (
                incoming.grad is None
                or outgoing.grad is None
                or incoming_buffer is None
                or outgoing_buffer is None
            ):
                raise RuntimeError("fused stale-allocation sources are absent")
            incoming_gradient = self._role_blocks(
                incoming.grad.detach(),
                outgoing=False,
                groups=_GROUPS,
                width=int(self.width),
                external=int(self.external_width),
            ).float()
            outgoing_gradient = self._role_blocks(
                outgoing.grad.detach(),
                outgoing=True,
                groups=_GROUPS,
                width=int(self.width),
                external=int(self.external_width),
            ).float()
            incoming_buffer_blocks = self._role_blocks(
                incoming_buffer.detach(),
                outgoing=False,
                groups=_GROUPS,
                width=int(self.width),
                external=int(self.external_width),
            ).float()
            outgoing_buffer_blocks = self._role_blocks(
                outgoing_buffer.detach(),
                outgoing=True,
                groups=_GROUPS,
                width=int(self.width),
                external=int(self.external_width),
            ).float()
            first_group = (rank - layer * _GROUPS) % _WORLD_SIZE
            packet, backend = fused_selector_layer_reductions(
                incoming_blocks[layer],
                outgoing_blocks[layer],
                incoming_gradient,
                outgoing_gradient,
                incoming_buffer_blocks,
                outgoing_buffer_blocks,
                first_group=first_group,
                momentum=float(self.momentum),
            )
            local_packets.append(packet)
            backends.add(backend)
        if len(backends) != 1:
            raise RuntimeError("fused stale-allocation backend changed by layer")
        local_packet = torch.cat(local_packets, dim=0)
        if tuple(local_packet.shape) != (_BLOCKS // _WORLD_SIZE, 5):
            raise RuntimeError("fused stale-allocation owner inventory changed")
        canonical = _all_gather_owned(local_packet, total=_BLOCKS, offset=0)
        self._fused_allocation_backend = next(iter(backends))
        self._fused_allocation_packet_bytes = int(
            canonical.numel() * canonical.element_size()
        )
        return canonical.view(_LAYERS, _GROUPS, 5)

    def _stale_allocation(
        self,
        incoming_endpoint,
        outgoing_endpoint_transpose,
        *,
        force_parent,
    ):
        cached = self._cached_allocation_coefficients
        template = self._cached_allocation_metadata
        if cached is None or template is None:
            raise RuntimeError("fused stale R01 allocation was not initialized")
        layers = len(self.pairs)
        if layers != _LAYERS or self.groups != _GROUPS:
            raise RuntimeError("fused stale-allocation M1 inventory changed")
        shape = (layers, self.groups, self.width, self.external_width)
        incoming_blocks = incoming_endpoint.view(shape)
        outgoing_blocks = outgoing_endpoint_transpose.view(shape)
        packet = self._fused_stale_allocation_packet(
            incoming_blocks, outgoing_blocks
        )
        if packet is None:
            return super()._stale_allocation(
                incoming_endpoint,
                outgoing_endpoint_transpose,
                force_parent=force_parent,
            )
        incoming_exact = packet[:, :, 0]
        outgoing_exact = packet[:, :, 1]
        incoming_momentum_linear = packet[:, :, 2]
        outgoing_momentum_linear = packet[:, :, 3]
        budget_weights = packet[:, :, 4]

        parent_budget = budget_weights.sum()
        raw_budget = (budget_weights * cached.square()).sum()
        tiny = torch.finfo(budget_weights.dtype).tiny
        scale = torch.sqrt(parent_budget / raw_budget.clamp_min(tiny))
        candidate = cached * scale
        candidate_budget = (budget_weights * candidate.square()).sum()
        budget_residual = (
            (candidate_budget - parent_budget).abs()
            / parent_budget.clamp_min(tiny)
        )
        candidate_incoming_exact = (incoming_exact * candidate).sum(dim=-1)
        candidate_outgoing_exact = (outgoing_exact * candidate).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum_linear * candidate
        ).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum_linear * candidate
        ).sum(dim=-1)
        finite = (
            torch.isfinite(candidate).all()
            & torch.isfinite(budget_residual)
            & torch.isfinite(candidate_incoming_exact).all()
            & torch.isfinite(candidate_outgoing_exact).all()
            & torch.isfinite(candidate_incoming_momentum).all()
            & torch.isfinite(candidate_outgoing_momentum).all()
        )
        role_descent_valid = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_momentum > 0.0).all()
            & (candidate_outgoing_momentum > 0.0).all()
        )
        budget_tolerance = 2048.0 * torch.finfo(budget_weights.dtype).eps
        accepted = (
            finite
            & role_descent_valid
            & (budget_residual <= budget_tolerance)
            & (~force_parent.any())
        )
        coefficients = torch.where(
            accepted, candidate, torch.ones_like(candidate)
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        selected_incoming_exact = torch.where(
            accepted,
            candidate_incoming_exact,
            incoming_exact.sum(dim=-1),
        )
        selected_outgoing_exact = torch.where(
            accepted,
            candidate_outgoing_exact,
            outgoing_exact.sum(dim=-1),
        )
        selected_incoming_momentum = torch.where(
            accepted,
            candidate_incoming_momentum,
            incoming_momentum_linear.sum(dim=-1),
        )
        selected_outgoing_momentum = torch.where(
            accepted,
            candidate_outgoing_momentum,
            outgoing_momentum_linear.sum(dim=-1),
        )
        selected_exact = (
            selected_incoming_exact.sum() + selected_outgoing_exact.sum()
        ).reshape(1)
        selected_momentum = (
            selected_incoming_momentum.sum()
            + selected_outgoing_momentum.sum()
        ).reshape(1)
        current_budget_residual = torch.where(
            accepted,
            budget_residual,
            torch.zeros_like(budget_residual),
        ).reshape(1)
        global_count = template["global_count"].clone()
        clip_factor = torch.tensor(
            float(self._r09_clip_factor),
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        zero = torch.zeros(
            (1,), device=coefficients.device, dtype=coefficients.dtype
        )
        metadata = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in template.items()
        }
        metadata.update({
            "accepted": accepted.reshape(1),
            "selected_exact_descent": selected_exact,
            "selected_momentum_descent": selected_momentum,
            "budget_residual": current_budget_residual,
            "improvement": zero,
            "global_count": global_count,
            "clip_factor": clip_factor,
            "selected_coefficient_min": coefficients.amin(),
            "selected_coefficient_median": coefficients.median(),
            "selected_coefficient_max": coefficients.amax(),
            "selected_incoming_exact_descent_min": selected_incoming_exact.amin(),
            "selected_outgoing_exact_descent_min": selected_outgoing_exact.amin(),
            "selected_incoming_momentum_descent_min": (
                selected_incoming_momentum.amin()
            ),
            "selected_outgoing_momentum_descent_min": (
                selected_outgoing_momentum.amin()
            ),
        })
        self._r01_global_metadata = metadata
        repeated = lambda value: value.reshape(1).expand(layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(coefficients.amin()),
            "coefficient_median": repeated(coefficients.median()),
            "coefficient_max": repeated(coefficients.amax()),
            "selected_exact_descent": repeated(selected_exact[0]),
            "selected_momentum_descent": repeated(selected_momentum[0]),
            "budget_residual": repeated(current_budget_residual[0]),
            "improvement": repeated(zero[0]),
            "global_count": global_count,
            "clip_factor": clip_factor,
        }
        choices = torch.where(
            accepted,
            torch.full(
                (layers,), 3, device=coefficients.device, dtype=torch.int64
            ),
            torch.zeros(
                (layers,), device=coefficients.device, dtype=torch.int64
            ),
        )
        parent_score = metadata["parent_score"][0]
        candidate_score = metadata["candidate_score"][0]
        scores = torch.stack((
            parent_score, candidate_score, parent_score, candidate_score
        )).reshape(1, 4).expand(layers, 4)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores,
            "score_margin": torch.zeros(
                (layers,), device=coefficients.device, dtype=coefficients.dtype
            ),
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        }

    def fused_allocation_runtime_report(self):
        return {
            "family_id": FUSED_ALLOCATION_FAMILY_ID,
            "backend": self._fused_allocation_backend,
            "global_group_blocks": _BLOCKS,
            "owned_group_blocks_per_rank": _BLOCKS // _WORLD_SIZE,
            "canonical_scalar_packet_bytes": self._fused_allocation_packet_bytes,
            "reduction_order_changed": True,
            "allocation_equation_changed": False,
            "lr_or_wd_changed": False,
            "fresh_quality_trajectory_required": True,
        }


class R01Metric4FusedStaleAllocationRowOptimizer(
    _FusedStaleAllocationMixin,
    R01StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowOptimizer,
):
    pass


R02Metric4FusedStaleAllocationRowAttentionOptimizer = (
    R02StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FUSED_ALLOCATION_FAMILY_ID",
    "R01Metric4FusedStaleAllocationRowOptimizer",
    "R02Metric4FusedStaleAllocationRowAttentionOptimizer",
)
