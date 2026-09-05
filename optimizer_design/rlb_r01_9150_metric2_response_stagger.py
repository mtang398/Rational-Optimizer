"""Conservative metric-2 R01 runtime approximation with exact attention.

This successor keeps the existing U6-parent, BF16 inverse-coordinate,
periodic-response, and owner-fused allocation equations.  It refreshes metric
factors and the 324-coordinate allocation on transitions 1,3,5,... .  After
the mandatory transition-1 initialization, the response route refreshes on
8,16,24,... so the two expensive refreshes do not coincide.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_metric4_response_stagger import (
    R01Metric4FusedAllocationResponseStaggerOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
METRIC_ALLOCATION_INTERVAL = 2
RESPONSE_INTERVAL = 8
RESPONSE_PHASE = 0
FAMILY_ID = "r01_metric2_fused_allocation_response_phase0_exact_attention_v1"


class R01Metric2FusedAllocationResponseStaggerOptimizer(
    R01Metric4FusedAllocationResponseStaggerOptimizer
):
    checkpoint_schema = FAMILY_ID

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["r01_metric2_family_id"] = FAMILY_ID
        group["r01_metric_refresh_interval"] = METRIC_ALLOCATION_INTERVAL
        group["r01_allocation_refresh_interval"] = METRIC_ALLOCATION_INTERVAL
        group["r01_response_stagger_interval"] = RESPONSE_INTERVAL
        group["r01_response_stagger_phase"] = RESPONSE_PHASE

    @staticmethod
    def _metric_allocation_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("metric2 refresh transition changed")
        return next_transition % METRIC_ALLOCATION_INTERVAL == 1

    @staticmethod
    def _response_refresh(next_transition):
        if not isinstance(next_transition, int) or next_transition < 1:
            raise RuntimeError("metric2 response transition changed")
        return next_transition == 1 or next_transition % RESPONSE_INTERVAL == 0

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        result = super().step(closure)
        if publish:
            self._last_telemetry.update({
                "rlb_r01_metric4_refresh_interval": (
                    METRIC_ALLOCATION_INTERVAL
                ),
                "rlb_r01_response_stagger_interval": RESPONSE_INTERVAL,
                "rlb_r01_response_stagger_phase": RESPONSE_PHASE,
            })
        return result

    def staggered_response_runtime_report(self):
        return {
            "family_id": FAMILY_ID,
            "metric_allocation_refresh_interval": (
                METRIC_ALLOCATION_INTERVAL
            ),
            "response_refresh_interval": RESPONSE_INTERVAL,
            "response_refresh_phase": RESPONSE_PHASE,
            "initial_transition_initializes_all_caches": True,
            "full_archived_attention_required": True,
            "refresh_and_reuse_equations_changed": False,
            "cache_ages_changed": True,
            "fresh_quality_trajectory_required": True,
            "lr_or_wd_changed": False,
        }


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FAMILY_ID",
    "METRIC_ALLOCATION_INTERVAL",
    "RESPONSE_INTERVAL",
    "RESPONSE_PHASE",
    "R01Metric2FusedAllocationResponseStaggerOptimizer",
)
