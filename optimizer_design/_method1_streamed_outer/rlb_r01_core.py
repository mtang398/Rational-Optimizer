"""Qualified metric-2 R01 ancestor with exact stack convenience methods."""

import torch

from .._method1_metric2_approx.rlb_r01_core import (  # noqa: F401
    R01_APPROXIMATION_ID,
)
from .._method1_metric2_approx.rlb_r01_core import R01Core as _QualifiedR01Core


class R01Core(_QualifiedR01Core):
    """Same qualified R01 equations; only centralize repeated FP32 stacks."""

    def _stack_incoming_parameters(self):
        return torch.stack(self.incoming).float()

    def _stack_outgoing_parameters(self, *, transpose=False):
        value = torch.stack(self.outgoing).float()
        return value.transpose(-2, -1) if transpose else value

    def _stack_incoming_gradients(self):
        return torch.stack([parameter.grad for parameter in self.incoming]).float()

    def _stack_outgoing_gradients(self, *, transpose=False):
        value = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float()
        return value.transpose(-2, -1) if transpose else value

    def _select_functional_corner(self, *args, **kwargs):
        if not getattr(self, "_r07_linear_image_reuse_enabled", False):
            return super()._select_functional_corner(*args, **kwargs)

        # Observe the immutable qualified R01 solve without rewriting it.
        # Its first tangent-image call is J[P] for the unscaled group bank,
        # and its one span solve returns the literal group coefficients.  The
        # selected image is therefore J[P] * c groupwise.  Carrying that exact
        # algebraic common subexpression into R05 removes one of R05's two
        # direct plus/minus Jacobian evaluations.
        image_calls = []
        coefficient_calls = []
        had_image_override = "_group_tangent_images" in self.__dict__
        old_image_override = self.__dict__.get("_group_tangent_images")
        image_function = self._group_tangent_images
        had_solve_override = "_select_group_span_coefficients" in self.__dict__
        old_solve_override = self.__dict__.get("_select_group_span_coefficients")
        solve_function = self._select_group_span_coefficients

        def capture_images(*call_args, **call_kwargs):
            value = image_function(*call_args, **call_kwargs)
            image_calls.append(value)
            return value

        def capture_coefficients(*call_args, **call_kwargs):
            value = solve_function(*call_args, **call_kwargs)
            coefficient_calls.append(value[0])
            return value

        self._group_tangent_images = capture_images
        self._select_group_span_coefficients = capture_coefficients
        try:
            result = super()._select_functional_corner(*args, **kwargs)
            if len(image_calls) != 2 or len(coefficient_calls) != 1:
                raise RuntimeError("qualified R01 image/solve inventory changed")
            packet = result[2]
            layers = len(self.pairs)
            coefficients = coefficient_calls[0].view(layers, self.groups)
            accepted = packet["choices"][0].eq(3)
            selected = torch.where(
                accepted, coefficients, torch.ones_like(coefficients)
            )
            self._remember_selected_group_images(
                image_calls[0] * selected[:, None, :, None]
            )
            self._remember_group_decay_images(image_calls[1])
            return result
        finally:
            if had_image_override:
                self._group_tangent_images = old_image_override
            else:
                del self._group_tangent_images
            if had_solve_override:
                self._select_group_span_coefficients = old_solve_override
            else:
                del self._select_group_span_coefficients


__all__ = ("R01Core", "R01_APPROXIMATION_ID")
