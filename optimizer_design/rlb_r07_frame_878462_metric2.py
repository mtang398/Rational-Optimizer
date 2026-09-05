"""Hash-gated Method1 with the quality-gated metric-2 R01 execution path."""

from __future__ import annotations

import torch

from .rlb_r07_frame_878462_replay import verify_r07_frame_878462_archive


ARCHIVE_CERTIFICATE = verify_r07_frame_878462_archive()

from ._method1_metric2_approx.rlb_r01_core import (  # noqa: E402
    R01_APPROXIMATION_ID,
)
from ._method1_metric2_approx.rlb_r01_core import R01Core as _ApproximateR01Core  # noqa: E402
from ._method1_metric2_approx.rlb_r07 import (  # noqa: E402
    R07AttentionOptimizer as _ExactAttentionOptimizer,
)
from ._method1_metric2_approx.rlb_r07 import R07Optimizer as _OverlayOptimizer  # noqa: E402
from ._method1_metric2_approx.rlb_r07_frame_core import (  # noqa: E402
    R07FrameCore as _OverlayCore,
)


FAMILY_ID = "method1_878462_with_metric2_r01_approximation_v1"


class Method1Metric2Core(_OverlayCore):
    checkpoint_schema = FAMILY_ID


class Method1Metric2Optimizer(Method1Metric2Core):
    pass


class _PeriodicOuterMethod1Mixin:
    """Execute the exact R03+frame outer transaction periodically.

    Reuse transitions execute the complete metric-2 R01 ancestor directly;
    they do not reuse an old matrix update.  Current gradients, momentum,
    budgets, LR, WD, and exact attention therefore remain active every step.
    """

    outer_refresh_interval: int

    def __init__(self, pairs, **kwargs):
        interval = int(self.outer_refresh_interval)
        if interval < 2:
            raise ValueError("periodic Method1 outer interval must be >=2")
        self._method1_outer_step = 0
        self._method1_outer_active = True
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["method1_outer_refresh_interval"] = interval

    def _select_functional_corner(self, *args, **kwargs):
        if self._method1_outer_active:
            return super()._select_functional_corner(*args, **kwargs)
        return _ApproximateR01Core._select_functional_corner(
            self, *args, **kwargs
        )

    @torch.no_grad()
    def step(self, closure=None):
        active = bool(self._method1_outer_active)
        publish = bool(self._capture_telemetry_next_step)
        if active:
            result = super().step(closure)
        else:
            self._r05_next_metadata = None
            self._r08_inverse_sqrt = None
            self._r08_role_direction = None
            self._r08_response_metadata = None
            self._r03_persistent_metadata = None
            self._r07_frame_metadata = None
            result = _ApproximateR01Core.step(self, closure)
        self._method1_outer_step += 1
        self.state[self.incoming[0]]["method1_outer_step"] = (
            self._method1_outer_step
        )
        next_transition = self._method1_outer_step + 1
        self._method1_outer_active = (
            next_transition % int(self.outer_refresh_interval) == 1
        )
        if publish:
            self._last_telemetry.update({
                "rlb_method1_outer_refresh_interval": int(
                    self.outer_refresh_interval
                ),
                "rlb_method1_outer_refreshed": int(active),
                "rlb_method1_outer_refresh_step": self._method1_outer_step,
            })
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        step = self.state[self.incoming[0]].get("method1_outer_step")
        if not isinstance(step, int) or step < 0:
            raise RuntimeError("periodic Method1 outer checkpoint changed")
        self._method1_outer_step = step
        self._method1_outer_active = (
            (step + 1) % int(self.outer_refresh_interval) == 1
        )
        return result


class Method1Metric2Outer2Optimizer(
    _PeriodicOuterMethod1Mixin, Method1Metric2Core
):
    outer_refresh_interval = 2
    checkpoint_schema = FAMILY_ID + "_outer2"


class Method1Metric2Outer4Optimizer(
    _PeriodicOuterMethod1Mixin, Method1Metric2Core
):
    outer_refresh_interval = 4
    checkpoint_schema = FAMILY_ID + "_outer4"


Method1Metric2AttentionOptimizer = _ExactAttentionOptimizer


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FAMILY_ID",
    "Method1Metric2AttentionOptimizer",
    "Method1Metric2Core",
    "Method1Metric2Optimizer",
    "Method1Metric2Outer2Optimizer",
    "Method1Metric2Outer4Optimizer",
    "R01_APPROXIMATION_ID",
)
