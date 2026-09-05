"""Origin-safe Phase-3 owner-local selector reductions for archived R01.

Functional images and cotangents remain rank-local and follow the exact
archived Fisher reduction.  Only DDP-identical selector endpoint, gradient,
and momentum-buffer sources are coordinate-owned.  The CUDA reduction reads
their original strided layer views and emits five FP32 scalars per block; it
never reconstructs an image from a different sample origin.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .._r01_9150_full_shard_phase1.core import (
    R01FullShardPhase1Optimizer,
    R02FullShardPhase1AttentionOptimizer,
    _all_gather_owned,
    _assert_rank_identical,
)
from ..rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer as _ExactR02AttentionOptimizer,
)
from .selector_reduction import fused_selector_layer_reductions


FULL_SHARD_PHASE3_ID = "r01_9150_numerical_selector_reduction_shard_phase3_v1"
SELECTOR_FUSION_ONLY_ID = "r01_9150_selector_fusion_only_v1"
_LAYERS = 18
_GROUPS = 18
_BLOCKS = _LAYERS * _GROUPS
_WORLD_SIZE = 4
_FLOAT_METADATA_KEYS = (
    "rank_threshold",
    "eigenvalue_max",
    "coefficient_min",
    "coefficient_median",
    "coefficient_max",
    "candidate_exact_descent",
    "candidate_momentum_descent",
    "selected_exact_descent",
    "selected_momentum_descent",
    "budget_residual",
    "parent_score",
    "candidate_score",
    "selected_score",
    "improvement",
)


class R01FullShardPhase3Optimizer(R01FullShardPhase1Optimizer):
    """Exact local-origin images plus fused selector stats and rank-zero eig."""

    _phase3_polar_execution = "phase1_exact_18_of_72_owner_shard"
    _phase3_prototype_id = FULL_SHARD_PHASE3_ID

    def __init__(
        self,
        *args,
        phase3_verify_reductions: bool = False,
        phase3_verify_rank0_solve: bool = False,
        **kwargs,
    ):
        self.phase3_verify_reductions = bool(phase3_verify_reductions)
        self.phase3_verify_rank0_solve = bool(phase3_verify_rank0_solve)
        self.phase3_last_reduction_backend = None
        self.phase3_last_reduction_bitwise = None
        self.phase3_last_reduction_max_abs = None
        self.phase3_last_reduction_relative_l2 = None
        self.phase3_last_rank0_solve_bitwise = None
        self.phase3_last_rank0_solve_max_abs = None
        self.phase3_last_scalar_packet_bytes = 0
        self.phase3_last_rank0_broadcast_bytes = 0
        self.phase3_last_report = None
        super().__init__(*args, **kwargs)

    def _reset_phase3_transition(self) -> None:
        self.phase3_last_reduction_backend = None
        self.phase3_last_reduction_bitwise = None
        self.phase3_last_reduction_max_abs = None
        self.phase3_last_reduction_relative_l2 = None
        self.phase3_last_rank0_solve_bitwise = None
        self.phase3_last_rank0_solve_max_abs = None
        self.phase3_last_scalar_packet_bytes = 0
        self.phase3_last_rank0_broadcast_bytes = 0
        self.phase3_last_report = None

    @staticmethod
    def _role_blocks(value, *, outgoing: bool, groups, width, external):
        if outgoing:
            value = value.transpose(-2, -1)
        return value.reshape(groups, width, external)

    def _reference_selector_packet(self, incoming_blocks, outgoing_blocks):
        incoming_gradients = torch.stack([
            parameter.grad for parameter in self.incoming
        ]).float().view_as(incoming_blocks)
        outgoing_gradients = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float().transpose(-2, -1).view_as(outgoing_blocks)
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view_as(incoming_blocks)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view_as(outgoing_blocks)
        return torch.stack((
            (incoming_gradients * incoming_blocks).sum(dim=(-2, -1)),
            (outgoing_gradients * outgoing_blocks).sum(dim=(-2, -1)),
            (incoming_momentum * incoming_blocks).sum(dim=(-2, -1)),
            (outgoing_momentum * outgoing_blocks).sum(dim=(-2, -1)),
            (
                incoming_blocks.square().sum(dim=(-2, -1))
                + outgoing_blocks.square().sum(dim=(-2, -1))
            ),
        ), dim=-1).reshape(_BLOCKS, 5)

    def _owner_selector_packet(self, incoming_blocks, outgoing_blocks):
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
                raise RuntimeError("R01 Phase-3 selector sources are absent")
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
            layer_sources = (
                incoming_blocks[layer],
                outgoing_blocks[layer],
                incoming_gradient,
                outgoing_gradient,
                incoming_buffer_blocks,
                outgoing_buffer_blocks,
            )
            if self.phase3_verify_reductions:
                labels = (
                    "incoming-endpoint",
                    "outgoing-endpoint",
                    "incoming-gradient",
                    "outgoing-gradient",
                    "incoming-buffer",
                    "outgoing-buffer",
                )
                for label, value in zip(labels, layer_sources):
                    _assert_rank_identical(
                        value.contiguous(),
                        f"phase3-layer-{layer}-{label}",
                    )
            first_group = (rank - layer * _GROUPS) % _WORLD_SIZE
            local_packet, backend = fused_selector_layer_reductions(
                *layer_sources,
                first_group=first_group,
                momentum=float(self.momentum),
            )
            local_packets.append(local_packet)
            backends.add(backend)
        if len(backends) != 1:
            raise RuntimeError("R01 Phase-3 reduction backend changed by layer")
        local_packet = torch.cat(local_packets, dim=0)
        if tuple(local_packet.shape) != (_BLOCKS // _WORLD_SIZE, 5):
            raise RuntimeError("R01 Phase-3 owner packet inventory changed")
        backend = next(iter(backends))
        canonical = _all_gather_owned(
            local_packet, total=_BLOCKS, offset=0
        )
        self.phase3_last_reduction_backend = backend
        self.phase3_last_scalar_packet_bytes = int(
            canonical.numel() * canonical.element_size()
        )
        if self.phase3_verify_reductions:
            reference = self._reference_selector_packet(
                incoming_blocks, outgoing_blocks
            )
            difference = (reference - canonical).abs()
            tiny = torch.finfo(reference.dtype).tiny
            self.phase3_last_reduction_bitwise = bool(
                torch.equal(reference, canonical)
            )
            self.phase3_last_reduction_max_abs = float(
                difference.amax().item()
            )
            self.phase3_last_reduction_relative_l2 = float((
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference).clamp_min(tiny)
            ).item())
        return canonical

    def _rank0_span_solve(
        self,
        fisher,
        decay_cross,
        exact_linear,
        momentum_linear,
        flat_budget,
        lr,
    ):
        rank = int(dist.get_rank())
        reference = None
        if self.phase3_verify_rank0_solve:
            reference = self._select_group_span_coefficients(
                fisher,
                decay_cross,
                exact_linear,
                momentum_linear,
                flat_budget,
                lr,
            )
        if rank == 0:
            if reference is None:
                coefficients, metadata = self._select_group_span_coefficients(
                    fisher,
                    decay_cross,
                    exact_linear,
                    momentum_linear,
                    flat_budget,
                    lr,
                )
            else:
                coefficients, metadata = reference
            float_packet = torch.cat((
                coefficients.reshape(-1),
                *(metadata[key].reshape(-1) for key in _FLOAT_METADATA_KEYS),
            ))
            integer_packet = torch.stack((
                metadata["accepted"].to(torch.int64).reshape(()),
                metadata["rank"].to(torch.int64).reshape(()),
            ))
        else:
            float_packet = torch.empty(
                _BLOCKS + len(_FLOAT_METADATA_KEYS),
                device=fisher.device,
                dtype=fisher.dtype,
            )
            integer_packet = torch.empty(
                2, device=fisher.device, dtype=torch.int64
            )
        dist.broadcast(float_packet, src=0)
        dist.broadcast(integer_packet, src=0)
        coefficients = float_packet[:_BLOCKS].reshape(1, _BLOCKS)
        metadata = {
            key: float_packet[_BLOCKS + index : _BLOCKS + index + 1]
            for index, key in enumerate(_FLOAT_METADATA_KEYS)
        }
        metadata["accepted"] = integer_packet[0:1].to(torch.bool)
        metadata["rank"] = integer_packet[1:2]
        self.phase3_last_rank0_broadcast_bytes = (
            int(float_packet.numel() * float_packet.element_size())
            + int(integer_packet.numel() * integer_packet.element_size())
        )
        if reference is not None:
            reference_coefficients, reference_metadata = reference
            differences = [
                (reference_coefficients - coefficients).abs().amax()
            ]
            differences.extend(
                (reference_metadata[key] - metadata[key]).abs().amax()
                for key in _FLOAT_METADATA_KEYS
            )
            maximum = torch.stack(differences).amax()
            discrete_equal = (
                torch.equal(
                    reference_metadata["accepted"], metadata["accepted"]
                )
                and torch.equal(reference_metadata["rank"], metadata["rank"])
            )
            self.phase3_last_rank0_solve_max_abs = float(maximum.item())
            self.phase3_last_rank0_solve_bitwise = bool(
                maximum.item() == 0.0 and discrete_equal
            )
        return coefficients, metadata

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        *,
        force_parent,
    ):
        del (
            incoming_parent,
            outgoing_parent_transpose,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
        )
        self.phase1_last_selector_owned_blocks = _BLOCKS // _WORLD_SIZE
        self.phase1_last_selector_reconstruction_bitwise = False
        cotangents = self._r09_loss_cotangents
        if any(value is None for value in (
            functional_inputs,
            functional_preactivations,
            functional_features,
            cotangents,
        )):
            raise RuntimeError("R01 Phase-3 did not receive global loss rows")
        layers = len(self.pairs)
        if force_parent.shape != (layers,):
            raise RuntimeError("R01 Phase-3 parent-limit inventory changed")

        factors = self._functional_jvp_factors(functional_preactivations)
        images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            factors=factors,
        )
        weight_decay = float(self.param_groups[0]["weight_decay"])
        group_decay = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * weight_decay,
            torch.stack(self.outgoing).float().transpose(-2, -1) * weight_decay,
            factors=factors,
        )
        fisher, decay_cross, global_count = self._reduce_global_loss_metric(
            images, cotangents, group_decay
        )

        incoming_blocks = incoming_endpoint.view(
            layers, self.groups, self.width, self.external_width
        )
        outgoing_blocks = outgoing_endpoint_transpose.view_as(incoming_blocks)
        packet = self._owner_selector_packet(incoming_blocks, outgoing_blocks)
        incoming_exact = packet[:, 0].view(layers, self.groups)
        outgoing_exact = packet[:, 1].view(layers, self.groups)
        incoming_momentum_linear = packet[:, 2].view(layers, self.groups)
        outgoing_momentum_linear = packet[:, 3].view(layers, self.groups)
        budget_weights = packet[:, 4].view(layers, self.groups)
        exact_linear = (incoming_exact + outgoing_exact).reshape(1, -1)
        momentum_linear = (
            incoming_momentum_linear + outgoing_momentum_linear
        ).reshape(1, -1)
        flat_budget = budget_weights.reshape(1, -1)

        flat_coefficients, metadata = self._rank0_span_solve(
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            flat_budget,
            lr,
        )
        candidate_coefficients = flat_coefficients.view(layers, self.groups)
        candidate_incoming_exact = (
            incoming_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_exact = (
            outgoing_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        role_descent_valid = (
            (candidate_incoming_exact > 0.0).all()
            & (candidate_outgoing_exact > 0.0).all()
            & (candidate_incoming_momentum > 0.0).all()
            & (candidate_outgoing_momentum > 0.0).all()
        )
        accepted = (
            metadata["accepted"][0]
            & role_descent_valid
            & (~force_parent.any())
        )
        coefficients = torch.where(
            accepted,
            candidate_coefficients,
            torch.ones_like(candidate_coefficients),
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        parent_exact = exact_linear.sum(dim=-1)
        parent_momentum = momentum_linear.sum(dim=-1)
        metadata["accepted"] = accepted.reshape(1)
        metadata["selected_exact_descent"] = torch.where(
            accepted, metadata["candidate_exact_descent"], parent_exact
        )
        metadata["selected_momentum_descent"] = torch.where(
            accepted, metadata["candidate_momentum_descent"], parent_momentum
        )
        metadata["selected_score"] = torch.where(
            accepted, metadata["candidate_score"], metadata["parent_score"]
        )
        metadata["improvement"] = (
            metadata["parent_score"] - metadata["selected_score"]
        )
        metadata["global_count"] = global_count
        metadata["clip_factor"] = torch.tensor(
            float(self._r09_clip_factor),
            device=global_count.device,
            dtype=global_count.dtype,
        )
        metadata["cross_layer_coupling_ratio"] = (
            self._cross_layer_coupling_ratio(fisher, layers, self.groups)
        )
        metadata["selected_coefficient_min"] = coefficients.amin()
        metadata["selected_coefficient_median"] = coefficients.median()
        metadata["selected_coefficient_max"] = coefficients.amax()
        metadata["selected_incoming_exact_descent_min"] = torch.where(
            accepted,
            candidate_incoming_exact.amin(),
            incoming_exact.sum(dim=-1).amin(),
        )
        metadata["selected_outgoing_exact_descent_min"] = torch.where(
            accepted,
            candidate_outgoing_exact.amin(),
            outgoing_exact.sum(dim=-1).amin(),
        )
        metadata["selected_incoming_momentum_descent_min"] = torch.where(
            accepted,
            candidate_incoming_momentum.amin(),
            incoming_momentum_linear.sum(dim=-1).amin(),
        )
        metadata["selected_outgoing_momentum_descent_min"] = torch.where(
            accepted,
            candidate_outgoing_momentum.amin(),
            outgoing_momentum_linear.sum(dim=-1).amin(),
        )
        self._r01_global_metadata = metadata

        repeated = lambda value: value.reshape(1).expand(layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(metadata["coefficient_min"][0]),
            "coefficient_median": repeated(metadata["coefficient_median"][0]),
            "coefficient_max": repeated(metadata["coefficient_max"][0]),
            "selected_exact_descent": repeated(
                metadata["selected_exact_descent"][0]
            ),
            "selected_momentum_descent": repeated(
                metadata["selected_momentum_descent"][0]
            ),
            "budget_residual": repeated(metadata["budget_residual"][0]),
            "improvement": repeated(metadata["improvement"][0]),
            "global_count": global_count,
            "clip_factor": metadata["clip_factor"],
        }

        choices = torch.where(
            accepted,
            torch.full(
                (layers,),
                3,
                device=coefficients.device,
                dtype=torch.int64,
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
        score_margin = (parent_score - candidate_score).abs().expand(layers)
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores,
            "score_margin": score_margin,
            "energies": torch.zeros_like(scores),
            "global_count": global_count,
        }

    def phase3_equivalence_report(self):
        if self.phase3_last_report is None:
            raise RuntimeError("R01 Phase-3 has not completed a transition")
        return dict(self.phase3_last_report)

    def _finish_phase3_transition(self, result):
        if self.phase3_last_reduction_backend is None:
            raise RuntimeError("R01 Phase-3 selector reductions did not run")
        elements = int(self.width) * int(self.external_width)
        original_flops_per_rank = 12 * _BLOCKS * elements
        owner_flops_per_rank = original_flops_per_rank // _WORLD_SIZE
        original_compulsory_input_bytes = 10 * _BLOCKS * elements * 4
        owner_fused_input_bytes = (
            6 * (_BLOCKS // _WORLD_SIZE) * elements * 4
        )
        original_eager_tensor_traffic = 22 * _BLOCKS * elements * 4
        self.phase3_last_report = {
            "schema": "r01_9150_phase3_numerical_equivalence_v1",
            "prototype_id": self._phase3_prototype_id,
            "polar_execution": self._phase3_polar_execution,
            "reduction_backend": self.phase3_last_reduction_backend,
            "selector_sources_rank_identical_verified": bool(
                self.phase3_verify_reductions
            ),
            "owner_source_materialization": (
                "none for M1 FP32 sources; direct strided layer reads"
            ),
            "owned_blocks": _BLOCKS // _WORLD_SIZE,
            "global_blocks": _BLOCKS,
            "scalar_packet_bytes": self.phase3_last_scalar_packet_bytes,
            "reduction_verified": bool(self.phase3_verify_reductions),
            "reduction_bitwise": self.phase3_last_reduction_bitwise,
            "reduction_max_abs": self.phase3_last_reduction_max_abs,
            "reduction_relative_l2": self.phase3_last_reduction_relative_l2,
            "rank0_eigh_enabled": True,
            "rank0_solve_verified": bool(self.phase3_verify_rank0_solve),
            "rank0_solve_bitwise": self.phase3_last_rank0_solve_bitwise,
            "rank0_solve_max_abs": self.phase3_last_rank0_solve_max_abs,
            "rank0_broadcast_bytes": self.phase3_last_rank0_broadcast_bytes,
            "fisher_construction": "archived_replicated_allreduce",
            "functional_origin_semantics": (
                "rank-local images and cotangents; never mixed across origins"
            ),
            "rank0_fisher_construction": (
                "not enabled: exact distributed sum association retained"
            ),
            "theoretical_original_reduction_flops_per_rank": (
                original_flops_per_rank
            ),
            "theoretical_owner_reduction_flops_per_rank": owner_flops_per_rank,
            "theoretical_flops_removed_per_rank": (
                original_flops_per_rank - owner_flops_per_rank
            ),
            "theoretical_original_compulsory_input_bytes_per_rank": (
                original_compulsory_input_bytes
            ),
            "theoretical_owner_fused_input_bytes_per_rank": (
                owner_fused_input_bytes
            ),
            "theoretical_original_eager_tensor_traffic_bytes_per_rank": (
                original_eager_tensor_traffic
            ),
        }
        return result

    @torch.no_grad()
    def step(self, closure=None):
        self._reset_phase3_transition()
        result = super().step(closure)
        return self._finish_phase3_transition(result)


class R01SelectorFusionOnlyOptimizer(R01FullShardPhase3Optimizer):
    """Selector fusion/rank-zero eig over literal archived replicated polar."""

    _phase3_polar_execution = "exact_archived_replicated_no_phase1_shard"
    _phase3_prototype_id = SELECTOR_FUSION_ONLY_ID

    def _group_tangent_images(self, *args, **kwargs):
        # Bypass Phase1's transaction guard: this variant intentionally runs
        # the archived full local image path and changes only selector stats.
        return _ExactR01Optimizer._group_tangent_images(self, *args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        self._reset_phase3_transition()
        # Explicitly enter the untouched archived transaction so no Phase1
        # polar monkey-patch, owner computation, or polar all-gather can run.
        result = _ExactR01Optimizer.step(self, closure)
        return self._finish_phase3_transition(result)


class R02FullShardPhase3AttentionOptimizer(
    R02FullShardPhase1AttentionOptimizer
):
    """Unchanged exact attention optimizer paired with Phase 3."""


class R02SelectorFusionOnlyAttentionOptimizer(_ExactR02AttentionOptimizer):
    """Literal archived attention optimizer paired with selector fusion."""


__all__ = (
    "FULL_SHARD_PHASE3_ID",
    "SELECTOR_FUSION_ONLY_ID",
    "R01FullShardPhase3Optimizer",
    "R01SelectorFusionOnlyOptimizer",
    "R02FullShardPhase3AttentionOptimizer",
    "R02SelectorFusionOnlyAttentionOptimizer",
)
