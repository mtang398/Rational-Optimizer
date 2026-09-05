"""Exact-NS5 R01 with a two-step covariance-factor refresh cadence.

This approximation changes one operation only: the complete outgoing-feature
and residual-input covariance factors are evaluated on odd transitions and
reused on the following even transition.  The response router, functional
loss metric, 324-coordinate allocation, exact five-step Newton--Schulz maps,
attention transaction, momentum, LR, and weight decay remain current on every
step.

It is a numerical method variant and therefore requires a fresh 4,000-step
quality trajectory before retention.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import (
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer,
    verify_r01_9150_archive,
)
from .rlb_r01_9150_stale_metric import _PeriodicFullMetricMixin


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
METRIC_REFRESH_INTERVAL = 2
FAMILY_ID = "r01_9150_metric2_exact_ns5_v1"


class R01Metric2ExactOptimizer(_PeriodicFullMetricMixin, _ExactR01Optimizer):
    """R01 whose two complete covariance factors refresh every other step."""

    checkpoint_schema = FAMILY_ID

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["r01_metric_refresh_interval"] = (
            METRIC_REFRESH_INTERVAL
        )
        self.param_groups[0]["r01_metric2_exact_family_id"] = FAMILY_ID

    @torch.no_grad()
    def step(self, closure=None):
        refresh = bool(self._capture_full_metric_this_step)
        # Bypass only the interval-8 policy method.  Its capture, consume, and
        # factor-cache methods remain active through this class's MRO.
        result = super(_PeriodicFullMetricMixin, self).step(closure)
        self._metric_refresh_step += 1
        self._capture_full_metric_this_step = (
            self._metric_refresh_step % METRIC_REFRESH_INTERVAL == 0
        )
        if refresh and set(self._cached_metric_factors) != {1, 2}:
            raise RuntimeError("metric2 exact refresh did not cache both factors")
        self.state[self.incoming[0]]["r01_metric2_exact_step"] = (
            self._metric_refresh_step
        )
        return result

    def periodic_metric_runtime_report(self):
        return {
            "family_id": FAMILY_ID,
            "metric_refresh_interval": METRIC_REFRESH_INTERVAL,
            "refresh_step_uses_exact_covariance_factors": True,
            "exact_ns5_every_step": True,
            "exact_attention_every_step": True,
            "response_router_every_step": True,
            "functional_fisher_every_step": True,
            "global_allocation_every_step": True,
            "current_momentum_lr_wd_every_step": True,
        }


R02Metric2ExactAttentionOptimizer = R02AttentionOptimizer


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FAMILY_ID",
    "METRIC_REFRESH_INTERVAL",
    "R01Metric2ExactOptimizer",
    "R02Metric2ExactAttentionOptimizer",
)
