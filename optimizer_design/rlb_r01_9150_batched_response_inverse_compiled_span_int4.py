"""Compile the fixed 64-round span solve on the current fast R01 path.

The owner-batched compiled-response inverse branch still executes three
literal 64-round secular bisections eagerly.  This composition captures only
those fixed rounds as one CUDA program.  Response statistics, inverse reuse,
coordinates, allocation, attention, packed-INT4 publication, ordinary AdamW,
and all five Newton--Schulz iterations remain unchanged.
"""

from __future__ import annotations

from . import rlb_r01_9150_local_layer_owner as _owner_module
from .rlb_method1_global_statistics_owner_compiled_span_int8 import (
    BISECTION_ROUNDS,
    _CompiledSpan64Mixin,
)
from .rlb_r01_9150_archive import R01Optimizer
from .rlb_r01_9150_batched_compiled_response_inverse_int4 import (
    _R01BatchedCompiledResponseInverseRouter,
)
from .rlb_r01_9150_compiled_response_inverse_int4 import _CONSTRUCTION_LOCK
from .rlb_r01_9150_local_owner_int4_direct import (
    R019150LocalLayerOwnerInt4DirectComposite,
)


FAMILY_ID = (
    "r01_9150_batched_compiled_response_inverse_compiled_span64_"
    "block256_int4_v1"
)


class _R01BatchedResponseInverseCompiledSpanRouter(
    _CompiledSpan64Mixin,
    _R01BatchedCompiledResponseInverseRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class R019150BatchedResponseInverseCompiledSpanInt4Composite(
    R019150LocalLayerOwnerInt4DirectComposite
):
    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _owner_module.R01Optimizer
            if original is not R01Optimizer:
                raise RuntimeError("R01 owner router was already patched")
            _owner_module.R01Optimizer = (
                _R01BatchedResponseInverseCompiledSpanRouter
            )
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.R01Optimizer = original
        if not isinstance(
            self.router, _R01BatchedResponseInverseCompiledSpanRouter
        ):
            raise RuntimeError("compiled-span batched R01 was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "scientific_parent_family_id": (
                "r01_9150_batched_compiled_response_inverse_"
                "block256_int4_v1"
            ),
            "group_span_bisection_rounds": BISECTION_ROUNDS,
            "group_span_executor": "fixed_shape_compiled_cuda_program",
            "r01_equations_changed": False,
            "floating_point_realization_changed": True,
            "response_statistics_changed": False,
            "coordinate_executor_changed": False,
            "owner_publication_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "R019150BatchedResponseInverseCompiledSpanInt4Composite",
    "_R01BatchedResponseInverseCompiledSpanRouter",
)
