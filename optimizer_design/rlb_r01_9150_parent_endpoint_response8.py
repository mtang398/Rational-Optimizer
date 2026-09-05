"""Periodic learned-response routing for the fast U6-parent R01 variant.

This numerical execution variant refreshes the complete R05/R07/R06/R02
response-statistic transaction once every eight optimizer transitions and
reuses the last routed coefficients between refreshes.  It leaves the current
gradients, Nesterov state, coordinate maps, polar directions, functional
allocation, realized budgets, LR, and WD active on every transition.
"""

from __future__ import annotations

import torch

from ._archive_r01_9150.rlb_response_capture_core import RLBResponseCaptureCore
from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_parent_endpoint_fast import (
    R01StaleMetricAllocation8BF16InverseParentEndpointFastRowOptimizer,
    R02StaleMetricAllocation8BF16InverseParentEndpointFastRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
RESPONSE_REFRESH_INTERVAL = 8
RESPONSE8_FAMILY_ID = (
    "r01_stale8_bf16_inverse_u6_parent_response_route8_v1"
)

_ROUTE_FIELDS = (
    "_r04_last_alignments",
    "_r07_attention_participation",
    "_r07_last_participation",
    "_r06_role_participation",
    "_r06_pair_alignments",
    "_r06_attention_alignments",
    "_r06_last_output_participation",
    "_r06_last_pair_alignments",
    "_r02_group_participation",
    "_r02_congruences",
    "_r02_r06_attention_intrinsic",
    "_r02_r05_attention_intrinsic",
    "_r02_attention_congruence",
)


class _PeriodicResponseRouteMixin:
    checkpoint_schema = "r01_stale8_bf16_inverse_u6_parent_response_route8_v1"

    def __init__(self, pairs, **kwargs):
        self._response_refresh_step = 0
        self._capture_full_response_this_step = True
        self._cached_response_route = None
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_response_route_refresh_interval"] = RESPONSE_REFRESH_INTERVAL
        group["r01_response_route_family_id"] = RESPONSE8_FAMILY_ID

    def _layer_response_metric(self, layer_index):
        if self._capture_full_response_this_step:
            return super()._layer_response_metric(layer_index)
        # Drain exactly the registered probe inventory, but do not evaluate
        # the P5/Q4 response/Jacobian statistics on a reuse transition.
        probe = RLBResponseCaptureCore._consume_probe(self, layer_index)
        self._router_probes[layer_index] = None
        return torch.ones((), device=probe.device, dtype=torch.float32)

    @staticmethod
    def _copy_route_tensor(value):
        if not torch.is_tensor(value):
            raise RuntimeError("cached response route contained a non-tensor")
        return value.detach().clone()

    def _capture_response_route(self):
        route = {}
        for name in _ROUTE_FIELDS:
            value = getattr(self, name, None)
            if value is None:
                raise RuntimeError(f"response refresh did not publish {name}")
            route[name] = self._copy_route_tensor(value)
        self._cached_response_route = route
        self.state[self.incoming[0]]["r01_cached_response_route"] = route

    def _restore_response_route(self):
        route = self._cached_response_route
        if route is None or set(route) != set(_ROUTE_FIELDS):
            raise RuntimeError("periodic response route was not initialized")
        for name in _ROUTE_FIELDS:
            setattr(self, name, self._copy_route_tensor(route[name]))

    def _consume_router_alignments(self):
        if self._capture_full_response_this_step:
            result = super()._consume_router_alignments()
            self._capture_response_route()
            return result

        # The per-layer method deliberately formed none of these statistics.
        # Clear their inventories and restore the complete last routed packet
        # that every descendant consumer would otherwise have produced.
        self._router_local_statistics = [None for _ in self.pairs]
        self._router_exact_initializer = [False for _ in self.pairs]
        self._r07_local_participation = [None for _ in self.pairs]
        self._r06_local_output_participation = [None for _ in self.pairs]
        self._r02_local_group_participation = [None for _ in self.pairs]
        self._restore_response_route()
        return self._r06_pair_alignments

    @torch.no_grad()
    def step(self, closure=None):
        refresh = bool(self._capture_full_response_this_step)
        publish = bool(self._capture_telemetry_next_step)
        result = super().step(closure)
        self._response_refresh_step += 1
        self.state[self.incoming[0]]["r01_response_refresh_step"] = (
            self._response_refresh_step
        )
        self._capture_full_response_this_step = (
            self._response_refresh_step % RESPONSE_REFRESH_INTERVAL == 0
        )
        if self._cached_response_route is None:
            raise RuntimeError("periodic response route lost its initial refresh")
        if publish:
            self._last_telemetry.update({
                "rlb_r01_response_route_refresh_interval": (
                    RESPONSE_REFRESH_INTERVAL
                ),
                "rlb_r01_response_route_refreshed": int(refresh),
                "rlb_r01_response_route_age": (
                    0
                    if refresh
                    else self._response_refresh_step % RESPONSE_REFRESH_INTERVAL
                ),
            })
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        local_state = self.state[self.incoming[0]]
        step = local_state.get("r01_response_refresh_step")
        route = local_state.get("r01_cached_response_route")
        if not isinstance(step, int) or step < 0:
            raise RuntimeError("periodic response checkpoint step changed")
        if not isinstance(route, dict) or set(route) != set(_ROUTE_FIELDS):
            raise RuntimeError("periodic response checkpoint route changed")
        self._response_refresh_step = step
        self._cached_response_route = route
        self._capture_full_response_this_step = (
            step % RESPONSE_REFRESH_INTERVAL == 0
        )
        return result

    def periodic_response_route_report(self):
        return {
            "family_id": RESPONSE8_FAMILY_ID,
            "refresh_interval": RESPONSE_REFRESH_INTERVAL,
            "refresh_transition_is_complete_response_route": True,
            "current_matrix_geometry_every_step": True,
            "current_functional_allocation_every_step": True,
            "current_budget_and_descent_gates_every_step": True,
            "lr_or_wd_changed": False,
            "fresh_quality_trajectory_required": True,
        }


class R01StaleMetricAllocation8BF16InverseParentEndpointResponse8RowOptimizer(
    _PeriodicResponseRouteMixin,
    R01StaleMetricAllocation8BF16InverseParentEndpointFastRowOptimizer,
):
    pass


R02StaleMetricAllocation8BF16InverseParentEndpointResponse8RowAttentionOptimizer = (
    R02StaleMetricAllocation8BF16InverseParentEndpointFastRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "RESPONSE8_FAMILY_ID",
    "RESPONSE_REFRESH_INTERVAL",
    "R01StaleMetricAllocation8BF16InverseParentEndpointResponse8RowOptimizer",
    "R02StaleMetricAllocation8BF16InverseParentEndpointResponse8RowAttentionOptimizer",
)
