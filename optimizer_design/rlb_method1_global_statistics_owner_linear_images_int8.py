"""Linearity-reused tangent images for the streamed global-owner Method1.

For every installed-RLB Jacobian ``J`` and paired atlas directions ``P, M``,

    J[(P + M) / 2] = (J[P] + J[M]) / 2,
    J[(P - M) / 2] = (J[P] - J[M]) / 2.

The parent implementation independently evaluates the plus and minus images
inside both the R05 paired-radial atlas and the R07 frame atlas, and evaluates
the identical weight-decay image again at every nested level.  This module
carries the already-computed selected parent image and R01 decay image through
the nested transaction, evaluates only each new orthogonal image, and reuses
the common decay image.  The mathematical
R01/R03/R05/R07/R08 equations, owner partition, cadence, INT8 publication,
LR, WD, and all five Newton--Schulz iterations are unchanged.  Floating-point
association changes, so a timing win still requires a fresh quality run.
"""

from __future__ import annotations

import torch

from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_global_statistics_owner_int8 import (
    _CONSTRUCTION_LOCK,
    Method1GlobalStatisticsOwnerInt8Composite,
)
from .rlb_method1_global_statistics_owner_streamed_int8 import (
    _StreamedGlobalStatisticsOwnerRouter,
)
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_statistics_owner_linear_image_reuse_int8_v1"


class _LinearImageReuseMixin:
    """Carry one exact selected Jacobian image through a nested atlas call."""

    def __init__(self, pairs, **kwargs):
        self._linear_reuse_selected_group_images = None
        self._linear_reuse_group_decay_images = None
        super().__init__(pairs, **kwargs)
        self._r07_linear_image_reuse_enabled = True

    def _selected_group_images(self):
        return self._linear_reuse_selected_group_images

    def _remember_selected_group_images(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise RuntimeError("linear image reuse inventory changed")
        if (
            int(images.shape[0]) != len(self.pairs)
            or int(images.shape[2]) != int(self.groups)
            or int(images.shape[3]) != int(self.external_width)
        ):
            raise RuntimeError("linear image reuse shape changed")
        self._linear_reuse_selected_group_images = images

    def _remember_group_decay_images(self, images):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise RuntimeError("linear decay-image reuse inventory changed")
        if (
            int(images.shape[0]) != len(self.pairs)
            or int(images.shape[2]) != int(self.groups)
            or int(images.shape[3]) != int(self.external_width)
        ):
            raise RuntimeError("linear decay-image reuse shape changed")
        self._linear_reuse_group_decay_images = images

    def _group_functional_decay_images(self, *args, **kwargs):
        cached = self._linear_reuse_group_decay_images
        if cached is not None:
            return cached
        return super()._group_functional_decay_images(*args, **kwargs)

    def _summed_functional_decay_image(self, *args, **kwargs):
        cached = self._linear_reuse_group_decay_images
        if cached is not None:
            return cached.sum(dim=2)
        return super()._summed_functional_decay_image(*args, **kwargs)

    def _select_functional_corner(self, *args, **kwargs):
        if (
            self._linear_reuse_selected_group_images is not None
            or self._linear_reuse_group_decay_images is not None
        ):
            raise RuntimeError("linear image reuse leaked across transactions")
        try:
            return super()._select_functional_corner(*args, **kwargs)
        finally:
            # The image is a step-local algebraic common subexpression, not
            # optimizer state and not a cross-step numerical approximation.
            self._linear_reuse_selected_group_images = None
            self._linear_reuse_group_decay_images = None


class _LinearImageReuseGlobalStatisticsOwnerRouter(
    _LinearImageReuseMixin,
    _StreamedGlobalStatisticsOwnerRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalStatisticsOwnerLinearImagesInt8Composite(
    Method1GlobalStatisticsOwnerInt8Composite
):
    """Install streamed Method1 with exact Jacobian-image common subexpressions."""

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
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = (
                _LinearImageReuseGlobalStatisticsOwnerRouter
            )
            try:
                Method1LocalLayerOwnerInt8DirectComposite.__init__(
                    self, blocks, adamw, **kwargs
                )
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(
            self.router, _LinearImageReuseGlobalStatisticsOwnerRouter
        ):
            raise RuntimeError("linear-image global owner was not installed")

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
            "linear_jacobian_image_reuse": True,
            "plus_minus_image_evaluations_per_atlas": "two_to_one",
            "outer_scientific_equations_changed": False,
            "floating_point_association_may_change": True,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "Method1GlobalStatisticsOwnerLinearImagesInt8Composite",
    "_LinearImageReuseGlobalStatisticsOwnerRouter",
    "_LinearImageReuseMixin",
)
