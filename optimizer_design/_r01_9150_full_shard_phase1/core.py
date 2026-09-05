"""Exact four-rank transaction geometry for the archived 9,150-step R01.

This opt-in Phase-1 bridge leaves every archived equation and persistent state
unchanged.  It slices the three router polar batches by the fixed global task
map ``task % 4`` and losslessly reconstructs each canonical BF16 result.  At
the post-polar selector boundary it likewise owns the fixed
``layer * 18 + group`` blocks and reconstructs the two canonical endpoint
tensors before entering the archived global selector.

The selector reconstruction is intentionally only an execution boundary in
Phase 1: the coupled 324-dimensional loss-metric solve still runs through the
archived implementation on every rank.  No reduction fusion, changed
association order, CUDA graph, or backward overlap is present here.
"""

from __future__ import annotations

import threading

import torch
import torch.distributed as dist

from ..rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer as _ExactR02AttentionOptimizer,
    verify_r01_9150_archive,
)


# Fail closed before reaching into the private historical module.
ARCHIVE_CERTIFICATE = verify_r01_9150_archive()

from .._archive_r01_9150 import rlb_r05_core as _router_module  # noqa: E402


FULL_SHARD_PHASE1_ID = "r01_9150_exact_full_transaction_shard_phase1_v1"
_EXACT_ROUTER_ZERO_POWER = _router_module._batched_zero_power
_PATCH_LOCK = threading.RLock()
_WORLD_SIZE = 4
_LAYERS = 18
_GROUPS = 18
_POLAR_BATCHES = (18, 18, 36)
_POLAR_OFFSETS = (0, 18, 36)
_POLAR_TASKS = 72
_SELECTOR_BLOCKS = _LAYERS * _GROUPS


def _require_four_ranks() -> int:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "R01 full-shard Phase 1 requires initialized distributed"
        )
    if int(dist.get_world_size()) != _WORLD_SIZE:
        raise RuntimeError(
            "R01 full-shard Phase 1 requires exactly four ranks"
        )
    return int(dist.get_rank())


def _owned_indices(total: int, offset: int, rank: int, device) -> torch.Tensor:
    values = [
        index
        for index in range(int(total))
        if (int(offset) + index) % _WORLD_SIZE == int(rank)
    ]
    return torch.tensor(values, dtype=torch.long, device=device)


def _all_gather_owned(
    local: torch.Tensor,
    *,
    total: int,
    offset: int,
) -> torch.Tensor:
    """Gather disjoint owner slices without any floating-point arithmetic."""

    rank = int(dist.get_rank())
    local_indices = _owned_indices(total, offset, rank, local.device)
    if local.ndim < 1 or local.shape[0] != local_indices.numel():
        raise RuntimeError("R01 Phase-1 local shard inventory changed")
    owner_indices = [
        _owned_indices(total, offset, owner, local.device)
        for owner in range(_WORLD_SIZE)
    ]
    maximum = max(int(indices.numel()) for indices in owner_indices)
    send = torch.empty(
        (maximum, *local.shape[1:]), device=local.device, dtype=local.dtype
    )
    send[: local.shape[0]].copy_(local)
    if local.shape[0] < maximum:
        send[local.shape[0] :].zero_()
    gathered = [torch.empty_like(send) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, send)
    canonical = torch.empty(
        (int(total), *local.shape[1:]),
        device=local.device,
        dtype=local.dtype,
    )
    for indices, owner_payload in zip(owner_indices, gathered):
        count = int(indices.numel())
        canonical.index_copy_(0, indices, owner_payload[:count])
    return canonical


def _assert_rank_identical(value: torch.Tensor, label: str) -> None:
    gathered = [torch.empty_like(value) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered, value)
    if any(not torch.equal(value, other) for other in gathered):
        raise RuntimeError(f"R01 Phase-1 cross-rank source changed: {label}")


class R01FullShardPhase1Optimizer(_ExactR01Optimizer):
    """Exact R01 with 72 polar tasks and 324 selector blocks rank-owned."""

    def __init__(
        self,
        *args,
        phase1_verify_sources: bool = False,
        phase1_verify_sliced_polar: bool = False,
        phase1_audit_block_kernel: bool = False,
        **kwargs,
    ):
        self.phase1_verify_sources = bool(phase1_verify_sources)
        self.phase1_verify_sliced_polar = bool(phase1_verify_sliced_polar)
        self.phase1_audit_block_kernel = bool(phase1_audit_block_kernel)
        self._phase1_active = False
        self._phase1_rank = None
        self._phase1_polar_call = 0
        self.phase1_last_polar_local_counts = ()
        self.phase1_last_polar_bitwise = ()
        self.phase1_last_selector_owned_blocks = 0
        self.phase1_last_selector_reconstruction_bitwise = False
        self.phase1_last_postpolar_image_bitwise = ()
        self.phase1_last_postpolar_image_max_abs = ()
        super().__init__(*args, **kwargs)
        if len(self.pairs) != _LAYERS or int(self.groups) != _GROUPS:
            raise ValueError(
                "R01 full-shard Phase 1 is fixed to 18 layers and 18 groups"
            )

    def _reset_phase1_transition(self, rank: int) -> None:
        self._phase1_active = True
        self._phase1_rank = int(rank)
        self._phase1_polar_call = 0
        self.phase1_last_polar_local_counts = ()
        self.phase1_last_polar_bitwise = ()
        self.phase1_last_selector_owned_blocks = 0
        self.phase1_last_selector_reconstruction_bitwise = False
        self.phase1_last_postpolar_image_bitwise = ()
        self.phase1_last_postpolar_image_max_abs = ()

    def _phase1_zero_power(
        self, source: torch.Tensor, steps: int
    ) -> torch.Tensor:
        if not self._phase1_active or self._phase1_rank is None:
            raise RuntimeError("R01 Phase-1 polar wrapper escaped its transaction")
        call = int(self._phase1_polar_call)
        if call >= len(_POLAR_BATCHES):
            raise RuntimeError("R01 Phase-1 polar call inventory increased")
        expected_batch = _POLAR_BATCHES[call]
        offset = _POLAR_OFFSETS[call]
        if source.ndim != 3 or int(source.shape[0]) != expected_batch:
            raise RuntimeError(
                "R01 Phase-1 polar batch inventory changed at "
                f"call {call}: {tuple(source.shape)}"
            )
        if int(steps) != int(self.ns_steps):
            raise RuntimeError("R01 Phase-1 Newton--Schulz step count changed")

        snapshot = source.clone() if self.phase1_verify_sources else None
        if self.phase1_verify_sources:
            _assert_rank_identical(source, f"polar-{call}")
        indices = _owned_indices(
            expected_batch, offset, self._phase1_rank, source.device
        )
        local_source = source.index_select(0, indices)
        local_result = _EXACT_ROUTER_ZERO_POWER(local_source, steps)

        # Preserve the archived tall-matrix return layout.  Its underlying
        # wire tensor is [batch, cols, rows] and the public result is a
        # transpose view; gathering in that orientation retains the layout.
        tall = source.shape[-2] > source.shape[-1]
        local_wire = (
            local_result.transpose(-2, -1) if tall else local_result
        ).contiguous()
        wire = _all_gather_owned(
            local_wire, total=expected_batch, offset=offset
        )
        result = wire.transpose(-2, -1) if tall else wire

        bitwise = True
        if self.phase1_verify_sliced_polar:
            reference = _EXACT_ROUTER_ZERO_POWER(source, steps)
            bitwise = bool(torch.equal(reference, result))
            if not bitwise:
                raise RuntimeError(
                    f"R01 Phase-1 sliced polar is not bitwise at call {call}"
                )
        if snapshot is not None and not torch.equal(source, snapshot):
            raise RuntimeError(
                f"R01 Phase-1 polar source was mutated at call {call}"
            )
        self.phase1_last_polar_local_counts += (int(indices.numel()),)
        self.phase1_last_polar_bitwise += (bitwise,)
        self._phase1_polar_call += 1
        return result

    def _reconstruct_selector_endpoint(
        self, value: torch.Tensor, label: str
    ) -> torch.Tensor:
        expected = (
            _LAYERS,
            int(self.groups) * int(self.width),
            int(self.external_width),
        )
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"R01 Phase-1 selector endpoint changed: {label} {tuple(value.shape)}"
            )
        snapshot = value.clone() if self.phase1_verify_sources else None
        if self.phase1_verify_sources:
            _assert_rank_identical(value, f"selector-{label}")
        blocks = value.reshape(
            _SELECTOR_BLOCKS, int(self.width), int(self.external_width)
        )
        indices = _owned_indices(
            _SELECTOR_BLOCKS, 0, int(self._phase1_rank), value.device
        )
        local = blocks.index_select(0, indices)
        canonical = _all_gather_owned(
            local, total=_SELECTOR_BLOCKS, offset=0
        ).reshape_as(value)
        if snapshot is not None and not torch.equal(value, snapshot):
            raise RuntimeError(f"R01 Phase-1 selector source mutated: {label}")
        if not torch.equal(value, canonical):
            raise RuntimeError(
                f"R01 Phase-1 selector reconstruction is not bitwise: {label}"
            )
        return canonical

    def _group_tangent_images(
        self,
        inputs,
        preactivations,
        features,
        incoming_direction,
        outgoing_direction_transpose,
        *,
        factors,
    ):
        """Audit, but never use, the numerical-only 81-block image kernel."""

        if not self._phase1_active or self._phase1_rank is None:
            raise RuntimeError("R01 Phase-1 tangent image escaped its transaction")
        if not self.phase1_audit_block_kernel:
            return super()._group_tangent_images(
                inputs,
                preactivations,
                features,
                incoming_direction,
                outgoing_direction_transpose,
                factors=factors,
            )
        samples = int(inputs.shape[1])
        expected_matrix = (
            _LAYERS,
            int(self.hidden),
            int(self.external_width),
        )
        if (
            tuple(inputs.shape)
            != (_LAYERS, samples, int(self.external_width))
            or tuple(preactivations.shape)
            != (_LAYERS, samples, int(self.hidden))
            or tuple(features.shape)
            != (_LAYERS, samples, int(self.hidden))
            or tuple(incoming_direction.shape) != expected_matrix
            or tuple(outgoing_direction_transpose.shape) != expected_matrix
        ):
            raise RuntimeError("R01 Phase-1 tangent-image inventory changed")
        if self.phase1_verify_sources:
            _assert_rank_identical(incoming_direction, "image-incoming")
            _assert_rank_identical(
                outgoing_direction_transpose, "image-outgoing"
            )

        indices = _owned_indices(
            _SELECTOR_BLOCKS,
            0,
            int(self._phase1_rank),
            inputs.device,
        )
        layer_indices = torch.div(
            indices, _GROUPS, rounding_mode="floor"
        )
        group_indices = torch.remainder(indices, _GROUPS)
        incoming_blocks = incoming_direction.view(
            _LAYERS, _GROUPS, int(self.width), int(self.external_width)
        )
        outgoing_blocks = outgoing_direction_transpose.view_as(
            incoming_blocks
        )
        local_inputs = inputs.index_select(0, layer_indices)
        local_incoming = incoming_blocks[layer_indices, group_indices]
        perturbation = torch.bmm(
            local_inputs, local_incoming.transpose(-2, -1)
        )

        u, derivative, radial = factors
        local_u = u[layer_indices, :, group_indices, :]
        local_derivative = derivative[layer_indices, :, group_indices, :]
        local_radial = radial[layer_indices, :, group_indices, :]
        projected = (local_u * perturbation).mean(dim=-1, keepdim=True)
        response = (
            local_derivative * perturbation + local_radial * projected
        )

        outgoing_weights = torch.stack(self.outgoing).float().view(
            _LAYERS,
            int(self.external_width),
            _GROUPS,
            int(self.width),
        ).permute(0, 2, 3, 1)
        local_weights = outgoing_weights[layer_indices, group_indices]
        incoming_image = torch.bmm(response, local_weights)
        feature_blocks = features.view(
            _LAYERS, samples, _GROUPS, int(self.width)
        )
        local_features = feature_blocks[
            layer_indices, :, group_indices, :
        ]
        local_outgoing = outgoing_blocks[layer_indices, group_indices]
        local_images = incoming_image + torch.bmm(
            local_features, local_outgoing
        )
        flat_images = _all_gather_owned(
            local_images, total=_SELECTOR_BLOCKS, offset=0
        )
        images = flat_images.view(
            _LAYERS, _GROUPS, samples, int(self.external_width)
        ).permute(0, 2, 1, 3).contiguous()

        reference = super()._group_tangent_images(
            inputs,
            preactivations,
            features,
            incoming_direction,
            outgoing_direction_transpose,
            factors=factors,
        )
        bitwise = bool(torch.equal(reference, images))
        maximum = float((reference - images).abs().amax().item())
        self.phase1_last_postpolar_image_bitwise += (bitwise,)
        self.phase1_last_postpolar_image_max_abs += (maximum,)
        # The shape-changing sliced GEMMs are a numerical-equivalence path.
        # Phase 1 rejects that boundary and feeds only the archived result to
        # the optimizer transaction.
        return reference

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
        if not self._phase1_active or self._phase1_rank is None:
            raise RuntimeError("R01 Phase-1 selector escaped its transaction")
        incoming_canonical = self._reconstruct_selector_endpoint(
            incoming_endpoint, "incoming"
        )
        outgoing_canonical = self._reconstruct_selector_endpoint(
            outgoing_endpoint_transpose, "outgoing"
        )
        self.phase1_last_selector_owned_blocks = int(
            _owned_indices(
                _SELECTOR_BLOCKS,
                0,
                int(self._phase1_rank),
                incoming_endpoint.device,
            ).numel()
        )
        self.phase1_last_selector_reconstruction_bitwise = True
        return super()._select_functional_corner(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_parent,
            incoming_canonical,
            outgoing_parent_transpose,
            outgoing_canonical,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
            lr,
            force_parent=force_parent,
        )

    @torch.no_grad()
    def step(self, closure=None):
        rank = _require_four_ranks()
        with _PATCH_LOCK:
            if _router_module._batched_zero_power is not _EXACT_ROUTER_ZERO_POWER:
                raise RuntimeError("R01 Phase-1 polar kernel was already patched")
            self._reset_phase1_transition(rank)
            _router_module._batched_zero_power = self._phase1_zero_power
            try:
                result = super().step(closure)
                if self._phase1_polar_call != len(_POLAR_BATCHES):
                    raise RuntimeError("R01 Phase-1 polar call inventory decreased")
                if sum(self.phase1_last_polar_local_counts) != (
                    _POLAR_TASKS // _WORLD_SIZE
                ):
                    raise RuntimeError("R01 Phase-1 polar ownership is imbalanced")
                if self.phase1_last_selector_owned_blocks != (
                    _SELECTOR_BLOCKS // _WORLD_SIZE
                ):
                    raise RuntimeError("R01 Phase-1 selector ownership is imbalanced")
                if self.phase1_audit_block_kernel and len(
                    self.phase1_last_postpolar_image_bitwise
                ) != 2:
                    raise RuntimeError(
                        "R01 Phase-1 tangent-image call inventory changed"
                    )
                return result
            finally:
                _router_module._batched_zero_power = _EXACT_ROUTER_ZERO_POWER
                self._phase1_active = False
                self._phase1_rank = None


class R02FullShardPhase1AttentionOptimizer(_ExactR02AttentionOptimizer):
    """Unchanged exact attention optimizer paired with the Phase-1 router."""


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FULL_SHARD_PHASE1_ID",
    "R01FullShardPhase1Optimizer",
    "R02FullShardPhase1AttentionOptimizer",
)
