"""Fuse Method 1's exact functional factors and group tangent images.

After compiling the literal 64-round secular solve, the remaining profiled
outer-refresh tail is dominated by repeated normalized-P5/Q4 factor and joint
group-tangent image construction.  This module captures those two fixed-shape
tensor programs without changing their formulas, input tensors, directions,
cadence, owner/global-statistics execution, INT8 wire, LR, WD, or NS5.
Compilation may reassociate floating-point operations, so a faster result
requires a fresh complete quality trajectory.
"""

from __future__ import annotations

import torch

from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_global_statistics_owner_compiled_span_int8 import (
    Method1GlobalStatisticsOwnerCompiledSpanInt8Composite,
    _CompiledSpanGlobalStatisticsOwnerRouter,
)
from .rlb_method1_global_statistics_owner_int8 import _CONSTRUCTION_LOCK
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_owner_compiled_span64_functional_int8_v1"


def _functional_factor_program(
    preactivations,
    numerator,
    denominator,
    rlb_eps: float,
    groups: int,
    width: int,
):
    layers, samples, _hidden = preactivations.shape
    z = preactivations.view(layers, samples, groups, width)
    rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + rlb_eps)
    u = z / rms
    u2 = u.square()
    u3 = u2 * u
    u4 = u2.square()
    u5 = u4 * u
    abs_u = u.abs()
    polynomial = (
        numerator[..., 0]
        + numerator[..., 1] * u
        + numerator[..., 2] * u2
        + numerator[..., 3] * u3
        + numerator[..., 4] * u4
        + numerator[..., 5] * u5
    )
    polynomial_derivative = (
        numerator[..., 1]
        + 2.0 * numerator[..., 2] * u
        + 3.0 * numerator[..., 3] * u2
        + 4.0 * numerator[..., 4] * u3
        + 5.0 * numerator[..., 5] * u4
    )
    divisor = (
        1.0
        + denominator[..., 0] * abs_u
        + denominator[..., 1] * u2
        + denominator[..., 2] * abs_u * u2
        + denominator[..., 3] * u4
    )
    divisor_derivative = (
        denominator[..., 0] * torch.sign(u)
        + 2.0 * denominator[..., 1] * u
        + 3.0 * denominator[..., 2] * u * abs_u
        + 4.0 * denominator[..., 3] * u3
    )
    function = polynomial / divisor
    derivative = (
        polynomial_derivative * divisor
        - polynomial * divisor_derivative
    ) / divisor.square()
    radial = function - u * derivative
    return u, derivative, radial


def _group_tangent_image_program(
    inputs,
    features,
    incoming_direction,
    outgoing_direction_transpose,
    outgoing_weights,
    u,
    derivative,
    radial,
    groups: int,
    width: int,
):
    layers, samples, _external = inputs.shape
    perturbation = torch.bmm(
        inputs, incoming_direction.transpose(-2, -1)
    )
    value = perturbation.view_as(u)
    projected = (u * value).mean(dim=-1, keepdim=True)
    response = (
        derivative * value + radial * projected
    ).reshape_as(perturbation).view(layers, samples, groups, width)
    incoming_image = torch.einsum(
        "lngw,lgwd->lngd", response, outgoing_weights
    )
    feature_blocks = features.view(layers, samples, groups, width)
    direction_blocks = outgoing_direction_transpose.view(
        layers, groups, width, outgoing_direction_transpose.shape[-1]
    )
    outgoing_image = torch.einsum(
        "lngw,lgwd->lngd", feature_blocks, direction_blocks
    )
    return incoming_image + outgoing_image


_compiled_functional_factors = torch.compile(
    _functional_factor_program, fullgraph=True, dynamic=False
)
_compiled_group_tangent_images = torch.compile(
    _group_tangent_image_program, fullgraph=True, dynamic=False
)


class _CompiledFunctionalImageMixin:
    def _functional_jvp_factors(self, preactivations):
        layers, _samples, hidden = preactivations.shape
        if layers != len(self.pairs) or hidden != self.hidden:
            raise RuntimeError("compiled functional JVP inventory changed")
        numerator = torch.stack(
            [pair["numerator"] for pair in self.pairs]
        ).float()[:, None, :, None, :]
        denominator = torch.stack(
            [pair["denominator"] for pair in self.pairs]
        ).float().abs()[:, None, :, None, :]
        program = (
            _compiled_functional_factors
            if preactivations.is_cuda
            else _functional_factor_program
        )
        return program(
            preactivations,
            numerator,
            denominator,
            float(self.rlb_eps),
            int(self.groups),
            int(self.width),
        )

    def _group_tangent_images(
        self,
        inputs,
        preactivations,
        features,
        incoming_direction,
        outgoing_direction_transpose,
        *,
        factors,
    ):
        layers = len(self.pairs)
        samples = inputs.shape[1]
        expected_matrix = (layers, self.hidden, self.external_width)
        if (
            inputs.shape != (layers, samples, self.external_width)
            or preactivations.shape != (layers, samples, self.hidden)
            or features.shape != (layers, samples, self.hidden)
            or incoming_direction.shape != expected_matrix
            or outgoing_direction_transpose.shape != expected_matrix
        ):
            raise RuntimeError("compiled functional group-span inventory changed")
        outgoing_weights = torch.stack(self.outgoing).float().view(
            layers, self.external_width, self.groups, self.width
        ).permute(0, 2, 3, 1)
        program = (
            _compiled_group_tangent_images
            if inputs.is_cuda
            else _group_tangent_image_program
        )
        return program(
            inputs,
            features,
            incoming_direction,
            outgoing_direction_transpose,
            outgoing_weights,
            *factors,
            int(self.groups),
            int(self.width),
        )


class _CompiledFunctionalGlobalOwnerRouter(
    _CompiledFunctionalImageMixin,
    _CompiledSpanGlobalStatisticsOwnerRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalOwnerCompiledFunctionalInt8Composite(
    Method1GlobalStatisticsOwnerCompiledSpanInt8Composite
):
    """Compiled-span Method 1 with exact functional-image launch fusion."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        router_kwargs = {
            key: kwargs[key]
            for key in ("lr", "weight_decay", "momentum", "ns_steps", "beta2", "eps")
        }
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = (
                _CompiledFunctionalGlobalOwnerRouter
            )
            try:
                Method1LocalLayerOwnerInt8DirectComposite.__init__(
                    self, blocks, adamw, **kwargs
                )
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(self.router, _CompiledFunctionalGlobalOwnerRouter):
            raise RuntimeError("compiled-functional Method1 owner was not installed")

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
            "functional_factor_equations_changed": False,
            "group_tangent_image_equations_changed": False,
            "functional_executor": "fixed_shape_compiled_programs",
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "floating_point_association_may_change": True,
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "Method1GlobalOwnerCompiledFunctionalInt8Composite",
    "_CompiledFunctionalGlobalOwnerRouter",
    "_functional_factor_program",
    "_group_tangent_image_program",
)
