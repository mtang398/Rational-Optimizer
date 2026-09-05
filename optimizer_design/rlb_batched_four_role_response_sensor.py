"""Layer-batched exact sensors for four-role Global-RLB response routing."""

from __future__ import annotations

import torch

from .rlb_response_alignment_row import _jacobian_kernel_inner


def stacked_version_a_factors(
    preactivation: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    groups: int,
    width: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the exact P5/Q4 factors for all layers in one tensor program."""

    if preactivation.ndim != 3 or preactivation.shape[-1] != int(groups) * int(width):
        raise RuntimeError("batched response preactivation inventory changed")
    layers = preactivation.shape[0]
    if numerator.shape != (layers, int(groups), 6) or denominator.shape != (
        layers, int(groups), 4
    ):
        raise RuntimeError("batched response coefficient inventory changed")
    value = preactivation.float().view(layers, -1, int(groups), int(width))
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    unit = value / rms
    unit2 = unit.square()
    unit3 = unit2 * unit
    unit4 = unit2.square()
    unit5 = unit4 * unit
    absolute = unit.abs()
    num = numerator.detach().float()[:, None, :, None, :]
    den = denominator.detach().float().abs()[:, None, :, None, :]
    polynomial = (
        num[..., 0] + num[..., 1] * unit + num[..., 2] * unit2
        + num[..., 3] * unit3 + num[..., 4] * unit4 + num[..., 5] * unit5
    )
    polynomial_derivative = (
        num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * unit2
        + 4.0 * num[..., 4] * unit3 + 5.0 * num[..., 5] * unit4
    )
    quotient = (
        1.0 + den[..., 0] * absolute + den[..., 1] * unit2
        + den[..., 2] * absolute * unit2 + den[..., 3] * unit4
    )
    quotient_derivative = (
        den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit
        + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * unit3
    )
    function = polynomial / quotient
    derivative = (
        polynomial_derivative * quotient - polynomial * quotient_derivative
    ) / quotient.square()
    radial = function - unit * derivative
    return unit, derivative, radial


def stacked_evaluate_response(
    unit: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate arbitrary live/frozen coefficients on stacked normalized rows."""

    if unit.ndim != 4:
        raise RuntimeError("batched response unit inventory changed")
    layers, _positions, groups, _width = unit.shape
    if numerator.shape != (layers, groups, 6) or denominator.shape != (
        layers, groups, 4
    ):
        raise RuntimeError("batched response evaluation inventory changed")
    square = unit.square()
    cube = square * unit
    fourth = square.square()
    fifth = fourth * unit
    absolute = unit.abs()
    num = numerator.float()[:, None, :, None, :]
    den = denominator.float().abs()[:, None, :, None, :]
    polynomial = (
        num[..., 0] + num[..., 1] * unit + num[..., 2] * square
        + num[..., 3] * cube + num[..., 4] * fourth + num[..., 5] * fifth
    )
    polynomial_derivative = (
        num[..., 1] + 2.0 * num[..., 2] * unit + 3.0 * num[..., 3] * square
        + 4.0 * num[..., 4] * cube + 5.0 * num[..., 5] * fourth
    )
    divisor = (
        1.0 + den[..., 0] * absolute + den[..., 1] * square
        + den[..., 2] * absolute * square + den[..., 3] * fourth
    )
    divisor_derivative = (
        den[..., 0] * torch.sign(unit) + 2.0 * den[..., 1] * unit
        + 3.0 * den[..., 2] * unit * absolute + 4.0 * den[..., 3] * cube
    )
    function = polynomial / divisor
    derivative = (
        polynomial_derivative * divisor - polynomial * divisor_derivative
    ) / divisor.square()
    return function, derivative


def stacked_response_adjoint(
    cotangents: torch.Tensor,
    outgoing_weight: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    groups: int,
    width: int,
) -> torch.Tensor:
    """Pull every layer's residual cotangent through its outgoing RLB map."""

    unit, derivative, radial = factors
    expected = (
        cotangents.shape[0], cotangents.shape[1], int(groups), int(width)
    )
    if cotangents.ndim != 3 or any(value.shape != expected for value in factors):
        raise RuntimeError("batched response-adjoint factor inventory changed")
    if outgoing_weight.shape != (
        cotangents.shape[0], cotangents.shape[2], int(groups) * int(width)
    ):
        raise RuntimeError("batched response-adjoint matrix inventory changed")
    pulled = torch.bmm(cotangents.float(), outgoing_weight.float()).view(expected)
    result = derivative * pulled + unit * (
        radial * pulled
    ).mean(dim=-1, keepdim=True)
    torch._assert_async(torch.isfinite(result).all())
    return result


def stacked_intrinsic_and_response_statistics(
    preactivation: torch.Tensor,
    cotangents: torch.Tensor,
    response_adjoint: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    frozen_numerator: torch.Tensor,
    frozen_denominator: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact 4LG intrinsic and 6LG live/frozen response sums."""

    unit, derivative, radial = factors
    if not (
        unit.shape == derivative.shape == radial.shape == response_adjoint.shape
        and unit.ndim == 4
        and preactivation.shape[:2] == unit.shape[:2]
        and preactivation.shape[-1] == unit.shape[-2] * unit.shape[-1]
        and cotangents.shape[:2] == unit.shape[:2]
    ):
        raise RuntimeError("batched response-statistic inventory changed")
    width = float(unit.shape[-1])
    function = radial + unit * derivative
    radial_jacobian = radial / width
    trace = derivative.square().sum(dim=-1)
    trace = trace + 2.0 * (derivative * unit * radial_jacobian).sum(dim=-1)
    trace = trace + unit.square().sum(dim=-1) * radial_jacobian.square().sum(dim=-1)
    trace_square = _jacobian_kernel_inner(
        unit, function, derivative, function, derivative
    )
    tiny = torch.finfo(trace.dtype).tiny
    incoming = (
        trace.square() / (width * trace_square.clamp_min(tiny))
    ).clamp(0.0, 1.0)
    energy = function.square().sum(dim=-1)
    fourth = function.pow(4).sum(dim=-1)
    outgoing = (
        energy.square() / (width * fourth.clamp_min(tiny))
    ).clamp(0.0, 1.0)
    incoming_weight = response_adjoint.square().mean(dim=-1)
    outgoing_weight = cotangents.float().square().mean(dim=-1)[..., None]
    participation = torch.stack((
        (incoming * incoming_weight).sum(dim=1),
        incoming_weight.sum(dim=1),
        (outgoing * outgoing_weight).sum(dim=1),
        outgoing_weight.expand_as(outgoing).sum(dim=1),
    ), dim=-1)

    frozen_function, frozen_derivative = stacked_evaluate_response(
        unit, frozen_numerator, frozen_denominator
    )
    incoming_cross = (
        _jacobian_kernel_inner(
            unit, function, derivative, frozen_function, frozen_derivative
        ) * incoming_weight
    ).sum(dim=1)
    incoming_live = (
        _jacobian_kernel_inner(unit, function, derivative, function, derivative)
        * incoming_weight
    ).sum(dim=1)
    incoming_frozen = (
        _jacobian_kernel_inner(
            unit, frozen_function, frozen_derivative,
            frozen_function, frozen_derivative,
        ) * incoming_weight
    ).sum(dim=1)
    value = preactivation.float().view_as(unit)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    live_h = rms * function
    frozen_h = rms * frozen_function
    outgoing_cross = (
        (live_h * frozen_h).sum(dim=-1).square() * outgoing_weight
    ).sum(dim=1)
    outgoing_live = (
        live_h.square().sum(dim=-1).square() * outgoing_weight
    ).sum(dim=1)
    outgoing_frozen = (
        frozen_h.square().sum(dim=-1).square() * outgoing_weight
    ).sum(dim=1)
    response = torch.stack((
        torch.stack((incoming_cross, incoming_live, incoming_frozen), dim=-1),
        torch.stack((outgoing_cross, outgoing_live, outgoing_frozen), dim=-1),
    ), dim=-2)
    torch._assert_async(torch.isfinite(participation).all())
    torch._assert_async(torch.isfinite(response).all())
    return participation, response


__all__ = (
    "stacked_evaluate_response",
    "stacked_intrinsic_and_response_statistics",
    "stacked_response_adjoint",
    "stacked_version_a_factors",
)
