"""Step-local duplicate-result cache layered over exact Method2.

The operations here are the independently toggleable router-cache subset of
the parity-proven exact-Method1 speed wrapper.  They retain only results which
the scientific program otherwise recomputes, unchanged, later in the same
optimizer transition.  The paired-postpolar equation remains implemented
solely by the recovered historical Method2 superclass.
"""

from __future__ import annotations

import torch

from optimizer_design.rlb_r07_frame_878462_fast import (
    R07Frame878462FastCore as _SharedCacheReference,
)
from optimizer_design.rlb_r07_paired_postpolar_881693_replay import (
    R07PairedPostpolar881693Optimizer,
)


EXPECTED_ROUTER_CACHE_STATS = {
    "functional_factor_hits": 3,
    "functional_factor_misses": 1,
    "nesterov_stack_hits": 4,
    "nesterov_stack_misses": 2,
    "outgoing_stack_hits": 7,
    "outgoing_stack_misses": 1,
}


class _R07PairedPostpolarDuplicateCacheMixin:
    """Cache only pure duplicate results during one recovered-Method2 step."""

    # These helpers have no ``super()`` closure and are reused directly from
    # the already parity-proven Method1 cache wrapper.  The five methods below
    # which do call ``super()`` are reproduced so they continue through the
    # recovered Method2 MRO, including its paired-postpolar component.
    _tensor_key = staticmethod(_SharedCacheReference._tensor_key)
    _cached_even_indices = _SharedCacheReference._cached_even_indices
    _functional_row_indices = _SharedCacheReference._functional_row_indices
    _sample_input_rows = _SharedCacheReference._sample_input_rows
    _current_outgoing_stack = _SharedCacheReference._current_outgoing_stack
    _group_tangent_images = _SharedCacheReference._group_tangent_images

    def __init__(self, pairs, **kwargs):
        self._r07_881693_step_cache = None
        self._r07_878462_step_cache = None
        self._r07_878462_row_index_cache = {}
        self._r07_878462_last_cache_stats = {}
        super().__init__(pairs, **kwargs)

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

    @torch.no_grad()
    def step(self, closure=None):
        if self._r07_881693_step_cache is not None:
            raise RuntimeError("Method2 fast cache encountered a nested step")
        stats = {
            "functional_factor_hits": 0,
            "functional_factor_misses": 0,
            "nesterov_stack_hits": 0,
            "nesterov_stack_misses": 0,
            "outgoing_stack_hits": 0,
            "outgoing_stack_misses": 0,
        }
        cache = {
            "functional_factors": {},
            "nesterov_stacks": {},
            "outgoing_stacks": {},
            "stats": stats,
        }
        self._r07_881693_step_cache = cache
        self._r07_878462_step_cache = cache
        try:
            return super().step(closure)
        finally:
            self._r07_878462_last_cache_stats = dict(stats)
            self._r07_878462_step_cache = None
            self._r07_881693_step_cache = None

    def implementation_cache_stats(self):
        """Return counters from the last transition (never checkpointed)."""
        return dict(self._r07_878462_last_cache_stats)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r07_881693_step_cache = None
        self._r07_878462_step_cache = None
        self._r07_878462_last_cache_stats = {}
        return result


class R07PairedPostpolar881693RouterCacheOptimizer(
    _R07PairedPostpolarDuplicateCacheMixin,
    R07PairedPostpolar881693Optimizer,
):
    """Exact recovered Method2 with the router cache as its only repair."""


__all__ = (
    "EXPECTED_ROUTER_CACHE_STATS",
    "R07PairedPostpolar881693RouterCacheOptimizer",
)
