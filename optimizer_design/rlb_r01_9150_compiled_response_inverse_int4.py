"""Compile R01 response statistics on top of inverse reuse.

The qualified inverse-reuse branch spends its largest named router component
evaluating the same live/frozen P5/Q4 response statistics for each owned
layer.  This module captures only that fixed-shape elementwise/reduction
program.  It retains the literal statistics, uses the already-computed live
Jacobian self inner product for participation, and leaves metric factors,
coordinates, allocation, attention, packed-INT4 publication, and NS5
unchanged.  CUDA reduction scheduling can change floating-point realization,
so any timing win requires a new complete 9,150-step quality run.
"""

from __future__ import annotations

import torch

from . import rlb_r01_9150_local_layer_owner as _owner_module
from ._archive_r07_frame_878462.rlb_response_capture_core import (
    RLBResponseCaptureCore,
)
from .rlb_r01_9150_archive import R01Optimizer
from .rlb_r01_9150_inverse_reuse_int4 import (
    _CONSTRUCTION_LOCK,
    _R01InverseReuseRouter,
)
from .rlb_r01_9150_local_owner_int4_direct import (
    R019150LocalLayerOwnerInt4DirectComposite,
)


FAMILY_ID = "r01_9150_compiled_response_inverse_block256_int4_v1"


def _evaluate_response_program(u, numerator, denominator):
    t2 = u.square()
    t3 = t2 * u
    t4 = t2.square()
    t5 = t4 * u
    abs_t = u.abs()
    powers = torch.stack((torch.ones_like(u), u, t2, t3, t4, t5), dim=-1)
    derivative_powers = torch.stack((
        torch.zeros_like(u),
        torch.ones_like(u),
        2.0 * u,
        3.0 * t2,
        4.0 * t3,
        5.0 * t4,
    ), dim=-1)
    denominator_powers = torch.stack((abs_t, t2, abs_t * t2, t4), dim=-1)
    denominator_derivative_powers = torch.stack((
        torch.sign(u), 2.0 * u, 3.0 * u * abs_t, 4.0 * t3
    ), dim=-1)
    a = numerator.float().view(1, numerator.shape[0], 1, 6)
    b = denominator.float().abs().view(1, denominator.shape[0], 1, 4)
    polynomial = (powers * a).sum(dim=-1)
    polynomial_derivative = (derivative_powers * a).sum(dim=-1)
    divisor = 1.0 + (denominator_powers * b).sum(dim=-1)
    divisor_derivative = (denominator_derivative_powers * b).sum(dim=-1)
    function = polynomial / divisor
    derivative = (
        polynomial_derivative * divisor - polynomial * divisor_derivative
    ) / divisor.square()
    return function, derivative


def _jacobian_inner_program(u, function_a, derivative_a, function_b, derivative_b):
    width = float(u.shape[-1])
    radial_a = (function_a - u * derivative_a) / width
    radial_b = (function_b - u * derivative_b) / width
    x_a = derivative_a * u
    x_b = derivative_b * u
    p_a = derivative_a.square()
    p_b = derivative_b.square()
    s = u.square().sum(dim=-1)
    diagonal = (p_a * p_b).sum(dim=-1)
    diagonal = diagonal + 2.0 * (p_a * x_b * radial_b).sum(dim=-1)
    diagonal = diagonal + s * (p_a * radial_b.square()).sum(dim=-1)
    diagonal = diagonal + 2.0 * (p_b * x_a * radial_a).sum(dim=-1)
    diagonal = diagonal + s * (p_b * radial_a.square()).sum(dim=-1)
    basis_a = torch.stack((x_a, radial_a), dim=-1)
    basis_b = torch.stack((x_b, radial_b), dim=-1)
    gram = basis_a.transpose(-2, -1) @ basis_b
    coupling = torch.zeros_like(gram)
    coupling[..., 0, 1] = 1.0
    coupling[..., 1, 0] = 1.0
    coupling[..., 1, 1] = s
    rank = ((coupling @ gram @ coupling) * gram).sum(dim=(-2, -1))
    return diagonal + rank


def _response_statistics_program(
    z, live_numerator, live_denominator, frozen_numerator, frozen_denominator, eps
):
    rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + eps)
    u = z / rms
    live_f, live_d = _evaluate_response_program(
        u, live_numerator, live_denominator
    )
    frozen_f, frozen_d = _evaluate_response_program(
        u, frozen_numerator, frozen_denominator
    )
    incoming_cross = _jacobian_inner_program(
        u, live_f, live_d, frozen_f, frozen_d
    )
    incoming_live = _jacobian_inner_program(
        u, live_f, live_d, live_f, live_d
    )
    incoming_frozen = _jacobian_inner_program(
        u, frozen_f, frozen_d, frozen_f, frozen_d
    )
    live_h = rms * live_f
    frozen_h = rms * frozen_f
    outgoing_cross = (live_h * frozen_h).sum(dim=-1).square()
    outgoing_live = live_h.square().sum(dim=-1).square()
    outgoing_frozen = frozen_h.square().sum(dim=-1).square()
    local_statistics = torch.stack((
        torch.stack((
            incoming_cross.sum(), incoming_live.sum(), incoming_frozen.sum()
        )),
        torch.stack((
            outgoing_cross.sum(), outgoing_live.sum(), outgoing_frozen.sum()
        )),
    ))

    width = float(u.shape[-1])
    radial = (live_f - u * live_d) / width
    trace = live_d.square().sum(dim=-1)
    trace = trace + 2.0 * (live_d * u * radial).sum(dim=-1)
    trace = trace + u.square().sum(dim=-1) * radial.square().sum(dim=-1)
    tiny = torch.finfo(trace.dtype).tiny
    incoming = trace.square() / (width * incoming_live.clamp_min(tiny))
    incoming_valid = (
        torch.isfinite(incoming) & (trace > 0.0) & (incoming_live > 0.0)
    )
    incoming = incoming.clamp_(0.0, 1.0)

    energy = live_f.square().sum(dim=-1)
    fourth = live_f.pow(4).sum(dim=-1)
    outgoing = energy.square() / (width * fourth.clamp_min(tiny))
    outgoing_valid = (
        torch.isfinite(outgoing) & (energy > 0.0) & (fourth > 0.0)
    )
    outgoing = outgoing.clamp_(0.0, 1.0)
    return local_statistics, incoming, outgoing, incoming_valid, outgoing_valid


_compiled_response_statistics = torch.compile(
    _response_statistics_program, fullgraph=True, dynamic=False
)


class _R01CompiledResponseInverseRouter(_R01InverseReuseRouter):
    checkpoint_schema = FAMILY_ID + "_router"

    def _layer_response_metric(self, layer_index):
        probe = RLBResponseCaptureCore._consume_probe(self, layer_index)
        z = probe.float().view(self.probe_count, self.groups, self.width)
        pair = self.pairs[layer_index]
        program = (
            _compiled_response_statistics
            if z.is_cuda
            else _response_statistics_program
        )
        statistics, incoming, outgoing, incoming_valid, outgoing_valid = program(
            z,
            pair["numerator"],
            pair["denominator"],
            self._frozen_numerators[layer_index],
            self._frozen_denominators[layer_index],
            float(self.rlb_eps),
        )
        torch._assert_async(incoming_valid.all())
        torch._assert_async(outgoing_valid.all())
        self._router_local_statistics[layer_index] = statistics
        self._router_exact_initializer[layer_index] = bool(
            torch.equal(
                pair["numerator"].detach().float(),
                self._frozen_numerators[layer_index],
            )
            and torch.equal(
                pair["denominator"].detach().float(),
                self._frozen_denominators[layer_index],
            )
        )
        self._router_probes[layer_index] = None
        sample_count = torch.tensor(
            float(incoming.numel()), device=z.device, dtype=z.dtype
        )
        self._r07_local_participation[layer_index] = torch.stack((
            incoming.sum(), sample_count
        ))
        self._r06_local_output_participation[layer_index] = torch.stack((
            outgoing.sum(), sample_count
        ))
        count = torch.full(
            (self.groups,),
            float(self.probe_count),
            device=z.device,
            dtype=z.dtype,
        )
        self._r02_local_group_participation[layer_index] = torch.stack((
            incoming.sum(dim=0), outgoing.sum(dim=0), count
        ))
        return torch.ones((), device=z.device, dtype=z.dtype)


class R019150CompiledResponseInverseInt4Composite(
    R019150LocalLayerOwnerInt4DirectComposite
):
    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _owner_module.R01Optimizer
            if original is not R01Optimizer:
                raise RuntimeError("R01 owner router was already patched")
            _owner_module.R01Optimizer = _R01CompiledResponseInverseRouter
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.R01Optimizer = original
        if not isinstance(self.router, _R01CompiledResponseInverseRouter):
            raise RuntimeError("compiled-response inverse R01 was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "scientific_parent_family_id": (
                "r01_9150_inverse_reuse_block256_int4_v1"
            ),
            "response_statistics_executor": "one_fixed_shape_compiled_program",
            "live_jacobian_self_inner_reused_for_participation": True,
            "response_statistics_changed": False,
            "coordinate_executor": "one_fp32_inverse_per_factor_then_matmul",
            "r01_equations_changed": False,
            "floating_point_realization_changed": True,
            "owner_publication_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "R019150CompiledResponseInverseInt4Composite",
    "_R01CompiledResponseInverseRouter",
    "_response_statistics_program",
)
