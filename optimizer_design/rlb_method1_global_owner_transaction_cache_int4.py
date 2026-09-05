"""Packed-INT4 publication for the current fast Method 1 executor.

The qualified Method 1 path is dominated by owner-update communication after
functional transaction caching.  This composition retains that exact router,
compiled functional program, compiled span64 solve, global statistics,
attention, cadence, LR/WD, and NS5.  It changes only the already quantized
owner-update publication from block-256 INT8 to packed block-256 INT4 with
FP32 scales.  The numerical wire change requires a fresh complete quality run
if matched timing improves.
"""

from __future__ import annotations

import torch

from .rlb_method1_global_owner_transaction_cache_int8 import (
    Method1GlobalOwnerTransactionCachedFunctionalInt8Composite,
    _PADDED_TRANSPORT_TENSORS,
)
from .rlb_r01_9150_local_owner_int4_direct import (
    Method1LocalLayerOwnerInt4DirectComposite,
)


FAMILY_ID = "method1_global_owner_transaction_cache_block256_int4_v1"


class Method1GlobalOwnerTransactionCachedFunctionalInt4Composite(
    Method1GlobalOwnerTransactionCachedFunctionalInt8Composite,
    Method1LocalLayerOwnerInt4DirectComposite,
):
    """Current Method 1 equations with the proven packed-INT4 transport."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        # In this MRO, the current INT8 constructor reaches the INT4 transport
        # constructor before the common owner base.  Both transport buffers
        # are therefore initialized around the same exact router; discard the
        # superseded padded INT8 tensors after construction.
        super().__init__(blocks, adamw, **kwargs)
        for name in _PADDED_TRANSPORT_TENSORS:
            delattr(self, name)
        if not hasattr(self, "_int4_send_packet"):
            raise RuntimeError("Method 1 INT4 transport was not initialized")

    @torch.no_grad()
    def step(self):
        self._prepare_functional_rows()
        self._prepare_response_rows()
        self._prepare_metric_rows()
        try:
            return Method1LocalLayerOwnerInt4DirectComposite.step(self)
        finally:
            self.router.probe_count = self._owner_original_probe_count
            self.router.input_capture_count = self._owner_original_input_capture_count
            self._sync_capture_plan()

    def telemetry(self):
        result = dict(Method1LocalLayerOwnerInt4DirectComposite.telemetry(self))
        result.update({
            "rlb_owner_global_functional_rows": self._last_global_functional_rows,
            "rlb_owner_global_response_rows": self._last_global_response_rows,
            "rlb_owner_global_input_rows": self._last_global_input_rows,
            "rlb_owner_global_feature_samples": self._last_global_feature_samples,
            "rlb_owner_global_r01_used": int(self.router._owner_global_r01_used),
            "rlb_owner_global_r01_dimension": self.router._owner_global_r01_dimension,
            "rlb_owner_global_r01_sample_count": self.router._owner_global_r01_sample_count,
        })
        return result

    def execution_report(self):
        result = dict(
            Method1GlobalOwnerTransactionCachedFunctionalInt8Composite.execution_report(
                self
            )
        )
        result.update({
            "family_id": FAMILY_ID,
            "scientific_parent_family_id": (
                "method1_global_owner_transaction_cached_functional_int8_v1"
            ),
            "direction_wire": "block256_symmetric_packed_int4_plus_fp32_scales",
            "quantization_code_range": (-7, 7),
            "values_per_wire_byte": 2,
            "ragged_owner_counts": self._int4_owner_counts,
            "padded_owner_slots_removed": True,
            "packed_value_scale_publication": True,
            "functional_equations_changed": False,
            "method1_equations_changed": False,
            "floating_point_update_changed_vs_int8_parent": True,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "Method1GlobalOwnerTransactionCachedFunctionalInt4Composite",
)
