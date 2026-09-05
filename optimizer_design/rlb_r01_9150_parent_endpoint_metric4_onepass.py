"""Higher-fidelity periodic geometry for the aggressive fast R01 path.

Metric factors and the 324-coordinate functional allocation refresh on
transitions 1, 5, 9, ... instead of 1, 9, 17, ... .  Learned-response routing
retains its existing eight-transition refresh interval.  All refresh and
reuse equations are unchanged; only the metric/allocation cache age changes.
This is a distinct numerical trajectory and requires a fresh quality run.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_parent_endpoint_response8_onepass import (
    R01StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowOptimizer,
    R02StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
METRIC_ALLOCATION_REFRESH_INTERVAL = 4
RESPONSE_REFRESH_INTERVAL = 8
METRIC4_FAMILY_ID = (
    "r01_u6_parent_metric4_allocation4_response8_onepass_v1"
)


class _MetricAllocation4Mixin:
    checkpoint_schema = "r01_u6_parent_metric4_allocation4_response8_onepass_v1"

    def __init__(self, pairs, **kwargs):
        self._metric4_refresh_step = 0
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_metric_refresh_interval"] = (
            METRIC_ALLOCATION_REFRESH_INTERVAL
        )
        group["r01_allocation_refresh_interval"] = (
            METRIC_ALLOCATION_REFRESH_INTERVAL
        )
        group["r01_response_route_refresh_interval"] = RESPONSE_REFRESH_INTERVAL
        group["r01_metric4_family_id"] = METRIC4_FAMILY_ID

    @staticmethod
    def _metric_allocation_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("metric4 refresh transition changed")
        return next_transition % METRIC_ALLOCATION_REFRESH_INTERVAL == 1

    def _install_metric4_plan(self, next_transition):
        refresh = self._metric_allocation_refresh(next_transition)
        self._capture_full_metric_this_step = refresh
        self._capture_full_allocation_this_step = refresh

    @torch.no_grad()
    def step(self, closure=None):
        metric = bool(self._capture_full_metric_this_step)
        allocation = bool(self._capture_full_allocation_this_step)
        publish = bool(self._capture_telemetry_next_step)
        if metric != allocation:
            raise RuntimeError("metric4 metric/allocation phases diverged")
        result = super().step(closure)
        self._metric4_refresh_step += 1
        self.state[self.incoming[0]]["r01_metric4_refresh_step"] = (
            self._metric4_refresh_step
        )
        self._install_metric4_plan(self._metric4_refresh_step + 1)
        if publish:
            self._last_telemetry.update({
                "rlb_r01_metric4_refresh_interval": (
                    METRIC_ALLOCATION_REFRESH_INTERVAL
                ),
                "rlb_r01_metric4_refreshed": int(metric),
            })
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        step = self.state[self.incoming[0]].get("r01_metric4_refresh_step")
        if not isinstance(step, int) or step < 0:
            raise RuntimeError("metric4 checkpoint step changed")
        self._metric4_refresh_step = step
        self._install_metric4_plan(step + 1)
        return result

    def metric4_runtime_report(self):
        return {
            "family_id": METRIC4_FAMILY_ID,
            "metric_refresh_interval": METRIC_ALLOCATION_REFRESH_INTERVAL,
            "allocation_refresh_interval": METRIC_ALLOCATION_REFRESH_INTERVAL,
            "response_refresh_interval": RESPONSE_REFRESH_INTERVAL,
            "refresh_and_reuse_equations_changed": False,
            "cache_ages_changed": True,
            "lr_or_wd_changed": False,
            "fresh_quality_trajectory_required": True,
        }


class R01StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowOptimizer(
    _MetricAllocation4Mixin,
    R01StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowOptimizer,
):
    pass


R02StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer = (
    R02StaleMetricAllocation8BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "METRIC4_FAMILY_ID",
    "METRIC_ALLOCATION_REFRESH_INTERVAL",
    "RESPONSE_REFRESH_INTERVAL",
    "R01StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowOptimizer",
    "R02StaleMetricAllocation4BF16InverseParentEndpointResponse8OnePassRowAttentionOptimizer",
)
