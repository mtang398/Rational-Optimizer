"""Shared periodic-outer execution policy for metric-2 approximations."""

from __future__ import annotations

import torch

from ._method1_metric2_approx.rlb_r01_core import R01Core as _ApproximateR01Core


class PeriodicOuterMetric2Mixin:
    """Refresh a complete outer transaction periodically.

    Reuse transitions execute the current metric-2 R01 ancestor.  They never
    replay an old matrix update, so current gradients, momentum, budgets,
    LR, WD, and the separately owned attention optimizer remain active.
    """

    outer_refresh_interval: int
    periodic_outer_label: str
    periodic_outer_metadata_names: tuple[str, ...] = ()

    def __init__(self, pairs, **kwargs):
        interval = int(self.outer_refresh_interval)
        if interval < 2:
            raise ValueError("periodic outer interval must be >=2")
        label = str(self.periodic_outer_label)
        if not label or not label.replace("_", "").isalnum():
            raise ValueError("periodic outer label is invalid")
        self._periodic_outer_step = 0
        self._periodic_outer_active = True
        super().__init__(pairs, **kwargs)
        self.param_groups[0][f"{label}_outer_refresh_interval"] = interval

    def _select_functional_corner(self, *args, **kwargs):
        if self._periodic_outer_active:
            return super()._select_functional_corner(*args, **kwargs)
        return _ApproximateR01Core._select_functional_corner(
            self, *args, **kwargs
        )

    @torch.no_grad()
    def step(self, closure=None):
        active = bool(self._periodic_outer_active)
        publish = bool(self._capture_telemetry_next_step)
        if active:
            result = super().step(closure)
        else:
            for name in self.periodic_outer_metadata_names:
                setattr(self, name, None)
            result = _ApproximateR01Core.step(self, closure)
        self._periodic_outer_step += 1
        state_key = f"{self.periodic_outer_label}_outer_step"
        self.state[self.incoming[0]][state_key] = self._periodic_outer_step
        next_transition = self._periodic_outer_step + 1
        self._periodic_outer_active = (
            next_transition % int(self.outer_refresh_interval) == 1
        )
        if publish:
            prefix = f"rlb_{self.periodic_outer_label}_outer"
            self._last_telemetry.update({
                f"{prefix}_refresh_interval": int(self.outer_refresh_interval),
                f"{prefix}_refreshed": int(active),
                f"{prefix}_refresh_step": self._periodic_outer_step,
            })
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        state_key = f"{self.periodic_outer_label}_outer_step"
        step = self.state[self.incoming[0]].get(state_key)
        if not isinstance(step, int) or step < 0:
            raise RuntimeError("periodic outer checkpoint changed")
        self._periodic_outer_step = step
        self._periodic_outer_active = (
            (step + 1) % int(self.outer_refresh_interval) == 1
        )
        return result


__all__ = ("PeriodicOuterMetric2Mixin",)
