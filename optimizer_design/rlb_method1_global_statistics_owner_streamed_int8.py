"""Compiled outer execution layered onto the all-rank-statistics M1 owner.

This implementation combines two already isolated numerical mechanisms:

* the quality-tested all-rank observation/global-R01 owner path; and
* the equation-preserving compiled/streamed Method1 outer transaction.

It does not change the R01/R03/R05/R07/R08 equations, the owner partition,
the block-256 INT8 publication, cadence, LR, WD, or any of the five
Newton--Schulz iterations.  Compilation can reassociate floating-point
operations, so this path still requires its own complete quality trajectory
if matched timing improves over the quality-tested owner parent.
"""

from __future__ import annotations

from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_global_statistics_owner_int8 import (
    _CONSTRUCTION_LOCK,
    _GlobalStatisticsOwnerMixin,
    Method1GlobalStatisticsOwnerInt8Composite,
)
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_method1_streamed_outer import Method1StreamedOuterRecursiveRouter
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_statistics_owner_streamed_outer_int8_v1"


class _StreamedGlobalStatisticsOwnerRouter(
    _GlobalStatisticsOwnerMixin,
    Method1StreamedOuterRecursiveRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalStatisticsOwnerStreamedInt8Composite(
    Method1GlobalStatisticsOwnerInt8Composite
):
    """Use compiled outer tensor programs inside the global-statistics owner."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        router_kwargs = {
            key: kwargs[key]
            for key in (
                "lr",
                "weight_decay",
                "momentum",
                "ns_steps",
                "beta2",
                "eps",
            )
        }
        # Reproduce the parent composite's construction boundary, changing
        # only the router class installed for the owner-local transaction.
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = (
                _StreamedGlobalStatisticsOwnerRouter
            )
            try:
                Method1LocalLayerOwnerInt8DirectComposite.__init__(
                    self, blocks, adamw, **kwargs
                )
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(self.router, _StreamedGlobalStatisticsOwnerRouter):
            raise RuntimeError("streamed global-statistics owner was not installed")

        # The capture broker observes all 18 layers but never steps them.  It
        # must remain the literal parent broker so compilation changes only
        # the owner-local outer arithmetic, not captured observations.
        self.capture_broker = Method1RecursiveInverseRouter(
            self.all_blocks, **router_kwargs
        )
        self._owner_original_probe_count = int(self.router.probe_count)
        self._owner_original_input_capture_count = int(
            self.router.input_capture_count
        )
        self._last_global_functional_rows = 0
        self._last_global_response_rows = 0
        self._last_global_input_rows = 0
        self._last_global_feature_samples = 0
        self._sync_capture_plan()

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "compiled_streamed_outer": True,
            "outer_scientific_equations_changed": False,
            "floating_point_association_may_change": True,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "Method1GlobalStatisticsOwnerStreamedInt8Composite",
    "_StreamedGlobalStatisticsOwnerRouter",
)
