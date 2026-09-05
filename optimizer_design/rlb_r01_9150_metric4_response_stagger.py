"""Stagger response refresh away from fused metric4 refresh transitions.

Transition one still initializes every cache.  Metric factors and the global
allocation then refresh on transitions 1,5,9,... while the learned-response
route refreshes on 7,15,23,... .  Frequencies and refresh/reuse equations are
unchanged; only response-cache ages change, so this is a distinct numerical
trajectory with a fresh quality requirement.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_metric4_fused_allocation import (
    R01Metric4FusedStaleAllocationRowOptimizer,
    R02Metric4FusedStaleAllocationRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
RESPONSE_INTERVAL = 8
RESPONSE_PHASE = 7
STAGGERED_RESPONSE_FAMILY_ID = "r01_metric4_fused_allocation_response_phase7_v1"


class R01Metric4FusedAllocationResponseStaggerOptimizer(
    R01Metric4FusedStaleAllocationRowOptimizer
):
    checkpoint_schema = STAGGERED_RESPONSE_FAMILY_ID

    def __init__(self, pairs, **kwargs):
        self._response_stagger_step = 0
        self._response_stagger_last_refresh = 0
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_response_stagger_family_id"] = STAGGERED_RESPONSE_FAMILY_ID
        group["r01_response_stagger_interval"] = RESPONSE_INTERVAL
        group["r01_response_stagger_phase"] = RESPONSE_PHASE

    @staticmethod
    def _response_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("staggered response transition changed")
        return next_transition == 1 or next_transition % RESPONSE_INTERVAL == RESPONSE_PHASE

    def _install_response_plan(self, next_transition):
        response = self._response_refresh(next_transition)
        metric_allocation = self._metric_allocation_refresh(next_transition)
        if next_transition > 1 and response and metric_allocation:
            raise RuntimeError("staggered response overlaps metric4 refresh")
        self._capture_full_response_this_step = response

    @torch.no_grad()
    def step(self, closure=None):
        response = bool(self._capture_full_response_this_step)
        publish = bool(self._capture_telemetry_next_step)
        result = super().step(closure)
        self._response_stagger_step += 1
        if self._response_stagger_step != self._metric4_refresh_step:
            raise RuntimeError("staggered response and metric4 steps diverged")
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
        last_refresh = local_state.get("r01_response_stagger_last_refresh")
        if (
            not isinstance(step, int)
            or not isinstance(last_refresh, int)
            or step < 0
            or last_refresh < 0
            or last_refresh > step
            or step != self._metric4_refresh_step
        ):
            raise RuntimeError("staggered response checkpoint changed")
        self._response_stagger_step = step
        self._response_stagger_last_refresh = last_refresh
        self._install_response_plan(step + 1)
        return result

    def staggered_response_runtime_report(self):
        return {
            "family_id": STAGGERED_RESPONSE_FAMILY_ID,
            "metric_allocation_refresh_interval": 4,
            "response_refresh_interval": RESPONSE_INTERVAL,
            "response_refresh_phase": RESPONSE_PHASE,
            "initial_transition_initializes_all_caches": True,
            "refresh_and_reuse_equations_changed": False,
            "cache_ages_changed": True,
            "fresh_quality_trajectory_required": True,
            "lr_or_wd_changed": False,
        }


R02Metric4FusedAllocationResponseStaggerAttentionOptimizer = (
    R02Metric4FusedStaleAllocationRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "RESPONSE_INTERVAL",
    "RESPONSE_PHASE",
    "STAGGERED_RESPONSE_FAMILY_ID",
    "R01Metric4FusedAllocationResponseStaggerOptimizer",
    "R02Metric4FusedAllocationResponseStaggerAttentionOptimizer",
)
