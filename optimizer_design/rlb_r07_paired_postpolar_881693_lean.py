"""Method2 paired post-polar router with telemetry-only summary statistics.

The historical Method2 source computes normalized-energy coefficients of
variation, extrema, and a second post-scale energy tensor on every outer
transition.  Those values do not enter the paired post-polar update and are
read only when telemetry is published.  This subclass keeps the complete
historical path on telemetry transitions and omits only those unused summary
statistics on ordinary transitions.

The joint frame polar, paired second-moment recurrence, row scale, descent
certificates, direction, NS5 calls, LR, WD, and checkpoint state are
unchanged.
"""

from __future__ import annotations

import torch

from ._method2_metric2_approx.rlb_r07_frame_core import (
    R07PairedAdaptiveFrameCore as _PairedAdaptiveCore,
)
from .rlb_r07_paired_postpolar_881693_metric2 import (
    Method2Metric2AttentionOptimizer,
    Method2Metric2Outer4Optimizer,
)
from .rlb_r07_frame_878462_lean_attention import _LeanR02AttentionMixin
from ._r07_paired_postpolar_881693_fast.router_cache import (
    _R07PairedPostpolarDuplicateCacheMixin,
)


FAMILY_ID = "method2_881693_telemetry_summary_elision_v1"


class Method2LeanOuter4Optimizer(
    _R07PairedPostpolarDuplicateCacheMixin,
    Method2Metric2Outer4Optimizer,
):
    """Exact ordinary Method2 update without unused telemetry summaries."""

    def _frame_source(
        self,
        incoming_momentum,
        outgoing_momentum,
        *,
        ns_steps,
        cross_role,
        polarize,
        measure=True,
    ):
        if measure or not self.use_paired_postpolar_adaptive:
            return super()._frame_source(
                incoming_momentum,
                outgoing_momentum,
                ns_steps=ns_steps,
                cross_role=cross_role,
                polarize=polarize,
                measure=measure,
            )

        # Bypass only the adaptive wrapper and execute its complete frame
        # parent before applying the same paired post-polar update below.
        incoming, outgoing, metadata = super(
            _PairedAdaptiveCore, self
        )._frame_source(
            incoming_momentum,
            outgoing_momentum,
            ns_steps=ns_steps,
            cross_role=cross_role,
            polarize=polarize,
            measure=measure,
        )
        if incoming.shape != outgoing.shape or incoming.ndim != 4:
            raise RuntimeError("R07 paired post-polar inventory changed")
        pair_energy = 0.5 * (
            incoming.square().mean(dim=-1)
            + outgoing.square().mean(dim=-1)
        )
        expected = (len(self.pairs), self.groups, self.width)
        if pair_energy.shape != expected:
            raise RuntimeError("R07 paired post-polar state shape changed")

        anchor = self.state[self.incoming[0]]
        moment = anchor.get("r07_pair_postpolar_second_moment")
        step = anchor.get("r07_pair_postpolar_step", 0)
        if moment is None:
            moment = torch.zeros_like(pair_energy)
        if (
            moment.shape != expected
            or moment.dtype != pair_energy.dtype
            or moment.device != pair_energy.device
            or type(step) is not int
            or step < 0
        ):
            raise RuntimeError("R07 paired post-polar checkpoint state changed")
        next_step = step + 1
        next_moment = (
            moment * float(self._r05_beta2)
            + pair_energy * (1.0 - float(self._r05_beta2))
        )
        correction = 1.0 - float(self._r05_beta2) ** next_step
        persistent_energy = next_moment / correction
        row_scale = 1.0 / (
            torch.sqrt(persistent_energy) + float(self._r05_eps)
        )
        torch._assert_async(
            torch.isfinite(pair_energy).all()
            & torch.isfinite(next_moment).all()
            & torch.isfinite(row_scale).all()
            & (pair_energy > 0.0).all()
            & (persistent_energy > 0.0).all()
            & (row_scale > 0.0).all()
        )
        incoming = incoming * row_scale[..., None]
        outgoing = outgoing * row_scale[..., None]
        incoming_descent = (incoming_momentum * incoming).sum(dim=(-2, -1))
        outgoing_descent = (outgoing_momentum * outgoing).sum(dim=(-2, -1))
        total_descent = incoming_descent + outgoing_descent
        adaptive_valid = (
            torch.isfinite(incoming).all(dim=(-2, -1))
            & torch.isfinite(outgoing).all(dim=(-2, -1))
            & torch.isfinite(total_descent)
            & (total_descent > 0.0)
        )
        torch._assert_async(adaptive_valid.all())

        anchor["r07_pair_postpolar_second_moment"] = next_moment
        anchor["r07_pair_postpolar_step"] = next_step
        metadata = dict(metadata)
        metadata.update({
            "valid": metadata["valid"] & adaptive_valid,
            "incoming_source_descent_min": incoming_descent.amin(),
            "outgoing_source_descent_min": outgoing_descent.amin(),
            "total_source_descent_min": total_descent.amin(),
        })
        # The wrapper checks presence after the step; remaining fields are
        # consumed only on the exact telemetry path above.
        self._r07_pair_adaptive_metadata = {
            "enabled": True,
            "step": next_step,
        }
        return incoming, outgoing, metadata


class Method2LeanAttentionOptimizer(
    _LeanR02AttentionMixin, Method2Metric2AttentionOptimizer
):
    """Exact Method2 attention update without ordinary telemetry products."""


def lean_method2_report():
    return {
        "family_id": FAMILY_ID,
        "paired_postpolar_update_changed": False,
        "step_local_duplicate_cache_enabled": True,
        "attention_update_changed": False,
        "frame_polar_changed": False,
        "ns5_changed": False,
        "lr_or_wd_changed": False,
        "telemetry_transitions_use_historical_path": True,
        "ordinary_transition_gate": "bitwise_parameters_and_state",
    }


__all__ = (
    "FAMILY_ID",
    "Method2LeanAttentionOptimizer",
    "Method2LeanOuter4Optimizer",
    "lean_method2_report",
)
