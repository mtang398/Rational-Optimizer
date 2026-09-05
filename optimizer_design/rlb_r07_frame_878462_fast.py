"""Bitwise-preserving execution cache for the exact job-878462 Method1.

The immutable archive remains the scientific source of truth.  This module
subclasses that archive and only retains values which the archived program
otherwise recomputes, unchanged, later in the *same* optimizer step:

* normalized P5/Q4 JVP factors for the same captured preactivation tensor;
* reconstructed Nesterov stacks after their momentum buffers have formed;
* the FP32 stack of the current outgoing RLB matrices used by every tangent
  image contraction; and
* deterministic evenly-spaced row-index tensors used by capture hooks.

No matrix product is reassociated, no collective is packed, no Newton--Schulz
call is fused, and no LR, WD, state recurrence, direction, or budget equation
is changed.  The step-local tensor caches are deliberately cleared even when
the archived step raises, and they are not part of the checkpoint schema.
"""

from __future__ import annotations

import torch

from .rlb_r07_frame_878462_replay import (
    R07Frame878462AttentionOptimizer,
    R07Frame878462Core,
    verify_r07_frame_878462_archive,
)


EXPECTED_IMPLEMENTATION_CACHE_STATS = {
    "functional_factor_hits": 3,
    "functional_factor_misses": 1,
    "nesterov_stack_hits": 4,
    "nesterov_stack_misses": 2,
    "outgoing_stack_hits": 7,
    "outgoing_stack_misses": 1,
}


class R07Frame878462FastCore(R07Frame878462Core):
    """Exact Method1 with step-local caches of pure duplicate results."""

    def __init__(self, pairs, **kwargs):
        verify_r07_frame_878462_archive()
        self._r07_878462_step_cache = None
        self._r07_878462_row_index_cache = {}
        self._r07_878462_last_cache_stats = {}
        super().__init__(pairs, **kwargs)

    @staticmethod
    def _tensor_key(value: torch.Tensor):
        return (
            id(value),
            int(value.data_ptr()),
            int(value._version),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
            value.device,
        )

    def _cached_even_indices(self, count, row_count, device):
        count = int(count)
        row_count = int(row_count)
        key = (count, row_count, device.type, device.index)
        indices = self._r07_878462_row_index_cache.get(key)
        if indices is None:
            if count == 1:
                indices = torch.zeros(1, device=device, dtype=torch.int64)
            else:
                numerators = torch.arange(
                    count, device=device, dtype=torch.int64
                ) * (row_count - 1)
                indices = torch.div(
                    numerators, count - 1, rounding_mode="floor"
                )
            self._r07_878462_row_index_cache[key] = indices
        return indices

    def _functional_row_indices(self, row_count, device):
        if row_count < self.probe_capture_count:
            raise RuntimeError(
                "R05 functional capture is smaller than its probe"
            )
        return self._cached_even_indices(
            self.probe_capture_count, row_count, device
        )

    def _sample_input_rows(self, value):
        flat = value.detach().reshape(-1, self.external_width)
        if flat.shape[0] < self.input_capture_count:
            raise RuntimeError("training tensor is smaller than the R08 input sample")
        indices = self._cached_even_indices(
            self.input_capture_count, flat.shape[0], flat.device
        )
        return flat.index_select(0, indices).clone()

    def _functional_jvp_factors(self, preactivations):
        cache = self._r07_878462_step_cache
        if cache is None:
            return super()._functional_jvp_factors(preactivations)
        coefficient_key = tuple(
            (
                self._tensor_key(pair["numerator"]),
                self._tensor_key(pair["denominator"]),
            )
            for pair in self.pairs
        )
        key = (self._tensor_key(preactivations), coefficient_key)
        cached = cache["functional_factors"].get(key)
        if cached is None:
            cached = super()._functional_jvp_factors(preactivations)
            cache["functional_factors"][key] = cached
            cache["stats"]["functional_factor_misses"] += 1
        else:
            cache["stats"]["functional_factor_hits"] += 1
        return cached

    def _current_nesterov_stack(self, parameters, *, transpose=False):
        cache = self._r07_878462_step_cache
        if cache is None:
            return super()._current_nesterov_stack(
                parameters, transpose=transpose
            )
        parameter_key = []
        for parameter in parameters:
            gradient = parameter.grad
            buffer = self.state[parameter].get("momentum_buffer")
            if gradient is None or buffer is None:
                # Preserve the archive's error path and wording.
                return super()._current_nesterov_stack(
                    parameters, transpose=transpose
                )
            parameter_key.append((
                self._tensor_key(parameter),
                self._tensor_key(gradient),
                self._tensor_key(buffer),
            ))
        key = (tuple(parameter_key), bool(transpose))
        cached = cache["nesterov_stacks"].get(key)
        if cached is None:
            cached = super()._current_nesterov_stack(
                parameters, transpose=transpose
            )
            cache["nesterov_stacks"][key] = cached
            cache["stats"]["nesterov_stack_misses"] += 1
        else:
            cache["stats"]["nesterov_stack_hits"] += 1
        return cached

    def _current_outgoing_stack(self):
        cache = self._r07_878462_step_cache
        if cache is None:
            return torch.stack(self.outgoing).float()
        key = tuple(self._tensor_key(parameter) for parameter in self.outgoing)
        cached = cache["outgoing_stacks"].get(key)
        if cached is None:
            cached = torch.stack(self.outgoing).float()
            cache["outgoing_stacks"][key] = cached
            cache["stats"]["outgoing_stack_misses"] += 1
        else:
            cache["stats"]["outgoing_stack_hits"] += 1
        return cached

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
        """Archived contraction with only its duplicate weight stack cached."""
        layers = len(self.pairs)
        samples = inputs.shape[1]
        expected_matrix = (layers, self.hidden, self.external_width)
        if (
            inputs.shape != (layers, samples, self.external_width)
            or preactivations.shape != (layers, samples, self.hidden)
            or features.shape != (layers, samples, self.hidden)
            or incoming_direction.shape != expected_matrix
            or outgoing_direction_transpose.shape != expected_matrix
        ):
            raise RuntimeError("R09 functional group-span inventory changed")
        perturbation = torch.bmm(
            inputs, incoming_direction.transpose(-2, -1)
        )
        response = self._functional_jvp(
            preactivations, perturbation, factors=factors
        ).view(layers, samples, self.groups, self.width)
        outgoing_weights = self._current_outgoing_stack().view(
            layers, self.external_width, self.groups, self.width
        ).permute(0, 2, 3, 1)
        incoming_image = torch.einsum(
            "lngw,lgwd->lngd", response, outgoing_weights
        )
        feature_blocks = features.view(
            layers, samples, self.groups, self.width
        )
        direction_blocks = outgoing_direction_transpose.view(
            layers, self.groups, self.width, self.external_width
        )
        outgoing_image = torch.einsum(
            "lngw,lgwd->lngd", feature_blocks, direction_blocks
        )
        return incoming_image + outgoing_image

    @torch.no_grad()
    def step(self, closure=None):
        if self._r07_878462_step_cache is not None:
            raise RuntimeError("Method1 fast cache encountered a nested step")
        stats = {
            "functional_factor_hits": 0,
            "functional_factor_misses": 0,
            "nesterov_stack_hits": 0,
            "nesterov_stack_misses": 0,
            "outgoing_stack_hits": 0,
            "outgoing_stack_misses": 0,
        }
        self._r07_878462_step_cache = {
            "functional_factors": {},
            "nesterov_stacks": {},
            "outgoing_stacks": {},
            "stats": stats,
        }
        try:
            return super().step(closure)
        finally:
            self._r07_878462_last_cache_stats = dict(stats)
            self._r07_878462_step_cache = None

    def implementation_cache_stats(self):
        """Return counters from the last transition (never checkpointed)."""
        return dict(self._r07_878462_last_cache_stats)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r07_878462_step_cache = None
        self._r07_878462_last_cache_stats = {}
        return result


class R07Frame878462FastOptimizer(R07Frame878462FastCore):
    """Public exact-Method1 router with duplicate-result caching."""


__all__ = (
    "EXPECTED_IMPLEMENTATION_CACHE_STATS",
    "R07Frame878462AttentionOptimizer",
    "R07Frame878462FastCore",
    "R07Frame878462FastOptimizer",
)
