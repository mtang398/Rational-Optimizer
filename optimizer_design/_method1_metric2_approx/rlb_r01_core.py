"""Metric-2 approximate R01 ancestor for the exact job-878462 Method1.

This module intentionally changes only the execution/numerical choices that
are already under an independent R01 quality trajectory: row-polar base
directions, periodic metric/allocation refresh, BF16 inverse-coordinate
application, the U6 parent endpoint, one-pass adjoints, owner-fused stale
allocation, and periodic learned-response routing.  Descendant R03 and R07
frame source files are loaded byte-for-byte from the immutable archive.
"""

from __future__ import annotations

import threading

import torch

from .._archive_r07_frame_878462 import rlb_r05_core as _router_module
from .._archive_r07_frame_878462.rlb_r01_core import R01Core as _ExactR01Core
from .._archive_r07_frame_878462.rlb_r05_core import R05Core as _ExactR05Core
from ..rlb_r01_9150_cheap_polar import _cheap_polar
from ..rlb_r01_9150_inverse_coordinate import _BF16InverseCoordinateMixin
from ..rlb_r01_9150_metric4_fused_allocation import _FusedStaleAllocationMixin
from ..rlb_r01_9150_parent_endpoint import _ParentEndpointMixin
from ..rlb_r01_9150_parent_endpoint_metric4_onepass import _MetricAllocation4Mixin
from ..rlb_r01_9150_parent_endpoint_response8 import _PeriodicResponseRouteMixin
from ..rlb_r01_9150_parent_endpoint_response8_onepass import _OnePassAdjointMixin
from ..rlb_r01_9150_stale_allocation import _PeriodicGlobalAllocationMixin
from ..rlb_r01_9150_stale_metric import _PeriodicFullMetricMixin


METRIC_ALLOCATION_INTERVAL = 2
RESPONSE_INTERVAL = 8
RESPONSE_PHASE = 0
R01_APPROXIMATION_ID = "method1_r01_metric2_response0_rowpolar_v1"
_PATCH_LOCK = threading.RLock()
_ROUTER_ZERO_POWER = _router_module._batched_zero_power


class _LocalRowPolarMixin:
    """Patch only the three archived base-R01 polar calls for one step."""

    @torch.no_grad()
    def step(self, closure=None):
        with _PATCH_LOCK:
            if _router_module._batched_zero_power is not _ROUTER_ZERO_POWER:
                raise RuntimeError("job-878462 base polar kernel was already patched")
            self._method1_metric2_polar_calls = 0

            def row_polar(source, requested):
                self._method1_metric2_polar_calls += 1
                return _cheap_polar(source, requested, "row")

            _router_module._batched_zero_power = row_polar
            try:
                result = super().step(closure)
                if self._method1_metric2_polar_calls != 3:
                    raise RuntimeError("job-878462 base polar inventory changed")
                return result
            finally:
                _router_module._batched_zero_power = _ROUTER_ZERO_POWER


class _LocalDeadU5BlendElisionMixin:
    """Elide the unused U5 blend while retaining Method1's exact R05 bytes."""

    def _blend_equalized(self, ordinary, adaptive_equal, alignment):
        role = len(self._r02_blend_records)
        if role not in (0, 1) or self._r02_congruences is None:
            raise RuntimeError("Method1 R02 shared branch order changed")
        result = _ExactR05Core._blend_equalized(
            ordinary, adaptive_equal, alignment
        )
        self._r02_blend_records.append((ordinary, None))
        return result


class _ResponseStaggerControlMixin:
    def __init__(self, pairs, **kwargs):
        self._response_stagger_step = 0
        self._response_stagger_last_refresh = 0
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_response_stagger_interval"] = RESPONSE_INTERVAL
        group["r01_response_stagger_phase"] = RESPONSE_PHASE

    @staticmethod
    def _response_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("Method1 metric2 response transition changed")
        return next_transition == 1 or next_transition % RESPONSE_INTERVAL == 0

    def _install_response_plan(self, next_transition):
        response = self._response_refresh(next_transition)
        metric = self._metric_allocation_refresh(next_transition)
        if next_transition > 1 and response and metric:
            raise RuntimeError("Method1 response and metric refreshes overlap")
        self._capture_full_response_this_step = response

    @torch.no_grad()
    def step(self, closure=None):
        response = bool(self._capture_full_response_this_step)
        publish = bool(self._capture_telemetry_next_step)
        result = super().step(closure)
        self._response_stagger_step += 1
        if self._response_stagger_step != self._metric4_refresh_step:
            raise RuntimeError("Method1 response and metric steps diverged")
        if response:
            self._response_stagger_last_refresh = self._response_stagger_step
        local_state = self.state[self.incoming[0]]
        local_state["r01_response_stagger_step"] = self._response_stagger_step
        local_state["r01_response_stagger_last_refresh"] = (
            self._response_stagger_last_refresh
        )
        self._install_response_plan(self._response_stagger_step + 1)
        if publish:
            self._last_telemetry.update({
                "rlb_r01_response_stagger_interval": RESPONSE_INTERVAL,
                "rlb_r01_response_stagger_phase": RESPONSE_PHASE,
                "rlb_r01_response_stagger_refreshed": int(response),
                "rlb_r01_response_stagger_age": (
                    self._response_stagger_step
                    - self._response_stagger_last_refresh
                ),
            })
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        local_state = self.state[self.incoming[0]]
        step = local_state.get("r01_response_stagger_step")
        last = local_state.get("r01_response_stagger_last_refresh")
        if (
            not isinstance(step, int)
            or not isinstance(last, int)
            or step < 0
            or last < 0
            or last > step
            or step != self._metric4_refresh_step
        ):
            raise RuntimeError("Method1 response checkpoint changed")
        self._response_stagger_step = step
        self._response_stagger_last_refresh = last
        self._install_response_plan(step + 1)
        return result


class _Metric2ControlMixin:
    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["method1_r01_approximation_id"] = R01_APPROXIMATION_ID
        group["r01_metric_refresh_interval"] = METRIC_ALLOCATION_INTERVAL
        group["r01_allocation_refresh_interval"] = METRIC_ALLOCATION_INTERVAL

    @staticmethod
    def _metric_allocation_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("Method1 metric2 transition changed")
        return next_transition % METRIC_ALLOCATION_INTERVAL == 1

    def _stale_allocation(
        self,
        incoming_endpoint,
        outgoing_endpoint_transpose,
        *,
        force_parent,
    ):
        # The fused owner kernel is intentionally specialized to the exact
        # 18x18 M1 inventory.  Small CPU fixtures exercise the same stale
        # equations through the generic implementation.
        if len(self.pairs) != 18 or int(self.groups) != 18:
            return _PeriodicGlobalAllocationMixin._stale_allocation(
                self,
                incoming_endpoint,
                outgoing_endpoint_transpose,
                force_parent=force_parent,
            )
        return super()._stale_allocation(
            incoming_endpoint,
            outgoing_endpoint_transpose,
            force_parent=force_parent,
        )

    @classmethod
    def _clone_cache_value(cls, value):
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: cls._clone_cache_value(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._clone_cache_value(item) for item in value)
        if isinstance(value, list):
            return [cls._clone_cache_value(item) for item in value]
        return value

    def state_dict(self):
        if set(self._cached_metric_factors) not in (set(), {1, 2}):
            raise RuntimeError("Method1 metric-factor cache is incomplete")
        inverse_values = {
            call: value[1]
            for call, value in self._cached_bf16_metric_inverses.items()
        }
        if set(inverse_values) not in (set(), {1, 2}):
            raise RuntimeError("Method1 inverse cache is incomplete")
        snapshot = {
            "metric_factors": self._clone_cache_value(
                self._cached_metric_factors
            ),
            "metric_inverses": self._clone_cache_value(inverse_values),
            "allocation_coefficients": self._clone_cache_value(
                self._cached_allocation_coefficients
            ),
            "allocation_metadata": self._clone_cache_value(
                self._cached_allocation_metadata
            ),
        }
        self.state[self.incoming[0]]["method1_metric2_cache"] = snapshot
        return super().state_dict()

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        snapshot = self.state[self.incoming[0]].get("method1_metric2_cache")
        if not isinstance(snapshot, dict) or set(snapshot) != {
            "metric_factors",
            "metric_inverses",
            "allocation_coefficients",
            "allocation_metadata",
        }:
            raise RuntimeError("Method1 metric2 checkpoint cache changed")
        metrics = snapshot["metric_factors"]
        inverses = snapshot["metric_inverses"]
        if set(metrics) != {1, 2} or set(inverses) != {1, 2}:
            raise RuntimeError("Method1 metric2 checkpoint cache is incomplete")
        self._cached_metric_factors = metrics
        self._cached_bf16_metric_inverses = {
            call: (metrics[call][0], inverses[call]) for call in (1, 2)
        }
        self._cached_allocation_coefficients = snapshot[
            "allocation_coefficients"
        ]
        self._cached_allocation_metadata = snapshot["allocation_metadata"]
        if (
            self._cached_allocation_coefficients is None
            or not isinstance(self._cached_allocation_metadata, dict)
        ):
            raise RuntimeError("Method1 allocation checkpoint cache changed")
        return result

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        result = super().step(closure)
        if publish:
            self._last_telemetry.update({
                "rlb_method1_r01_metric_interval": METRIC_ALLOCATION_INTERVAL,
                "rlb_method1_r01_response_interval": RESPONSE_INTERVAL,
                "rlb_method1_r01_response_phase": RESPONSE_PHASE,
                "rlb_method1_r01_base_polar": "row_normalized",
            })
        return result


class R01Core(
    _Metric2ControlMixin,
    _ResponseStaggerControlMixin,
    _FusedStaleAllocationMixin,
    _MetricAllocation4Mixin,
    _OnePassAdjointMixin,
    _PeriodicResponseRouteMixin,
    _LocalDeadU5BlendElisionMixin,
    _ParentEndpointMixin,
    _BF16InverseCoordinateMixin,
    _PeriodicGlobalAllocationMixin,
    _PeriodicFullMetricMixin,
    _LocalRowPolarMixin,
    _ExactR01Core,
):
    """Method1-compatible approximate R01 ancestor."""

    checkpoint_schema = R01_APPROXIMATION_ID


__all__ = ("R01Core",)
