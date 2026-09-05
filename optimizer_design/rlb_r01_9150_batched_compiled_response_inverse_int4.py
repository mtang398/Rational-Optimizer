"""Batch owned-layer response statistics on compiled-response inverse R01.

The parent launches the same compiled response program separately for every
owned layer.  This realization stacks the four-or-five owner-local layers and
executes their mathematically independent reductions in one compiled program.
It does not mix layer reductions or change response, coordinate, allocation,
attention, packed-INT4 publication, AdamW, or Newton--Schulz equations.
"""

from __future__ import annotations

import torch

from . import rlb_r01_9150_local_layer_owner as _owner_module
from ._archive_r07_frame_878462.rlb_response_capture_core import (
    RLBResponseCaptureCore,
)
from .rlb_r01_9150_archive import R01Optimizer
from .rlb_r01_9150_compiled_response_inverse_int4 import (
    _CONSTRUCTION_LOCK,
    _R01CompiledResponseInverseRouter,
    _jacobian_inner_program,
)
from .rlb_r01_9150_local_owner_int4_direct import (
    R019150LocalLayerOwnerInt4DirectComposite,
)


FAMILY_ID = "r01_9150_batched_compiled_response_inverse_block256_int4_v1"


def _batched_evaluate_response_program(u, numerator, denominator):
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
    a = numerator.float()[:, None, :, None, :]
    b = denominator.float().abs()[:, None, :, None, :]
    polynomial = (powers * a).sum(dim=-1)
    polynomial_derivative = (derivative_powers * a).sum(dim=-1)
    divisor = 1.0 + (denominator_powers * b).sum(dim=-1)
    divisor_derivative = (denominator_derivative_powers * b).sum(dim=-1)
    function = polynomial / divisor
    derivative = (
        polynomial_derivative * divisor - polynomial * divisor_derivative
    ) / divisor.square()
    return function, derivative


def _batched_response_statistics_program(
    z, live_numerator, live_denominator, frozen_numerator, frozen_denominator, eps
):
    rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + eps)
    u = z / rms
    live_f, live_d = _batched_evaluate_response_program(
        u, live_numerator, live_denominator
    )
    frozen_f, frozen_d = _batched_evaluate_response_program(
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
            incoming_cross.sum(dim=(-2, -1)),
            incoming_live.sum(dim=(-2, -1)),
            incoming_frozen.sum(dim=(-2, -1)),
        ), dim=-1),
        torch.stack((
            outgoing_cross.sum(dim=(-2, -1)),
            outgoing_live.sum(dim=(-2, -1)),
            outgoing_frozen.sum(dim=(-2, -1)),
        ), dim=-1),
    ), dim=1)

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


_compiled_batched_response_statistics = torch.compile(
    _batched_response_statistics_program, fullgraph=True, dynamic=False
)


class _R01BatchedCompiledResponseInverseRouter(
    _R01CompiledResponseInverseRouter
):
    checkpoint_schema = FAMILY_ID + "_router"

    def _prepare_response_batch(self) -> None:
        probes = [
            RLBResponseCaptureCore._consume_probe(self, index)
            for index in range(len(self.pairs))
        ]
        z = torch.stack([
            probe.float().view(self.probe_count, self.groups, self.width)
            for probe in probes
        ])
        live_numerator = torch.stack([
            pair["numerator"] for pair in self.pairs
        ])
        live_denominator = torch.stack([
            pair["denominator"] for pair in self.pairs
        ])
        frozen_numerator = torch.stack(self._frozen_numerators)
        frozen_denominator = torch.stack(self._frozen_denominators)
        program = (
            _compiled_batched_response_statistics
            if z.is_cuda
            else _batched_response_statistics_program
        )
        statistics, incoming, outgoing, incoming_valid, outgoing_valid = program(
            z,
            live_numerator,
            live_denominator,
            frozen_numerator,
            frozen_denominator,
            float(self.rlb_eps),
        )
        torch._assert_async(incoming_valid.all())
        torch._assert_async(outgoing_valid.all())
        sample_count = torch.tensor(
            float(incoming.shape[1] * incoming.shape[2]),
            device=z.device,
            dtype=z.dtype,
        )
        count = torch.full(
            (self.groups,),
            float(self.probe_count),
            device=z.device,
            dtype=z.dtype,
        )
        for index, pair in enumerate(self.pairs):
            self._router_local_statistics[index] = statistics[index]
            self._router_exact_initializer[index] = bool(
                torch.equal(
                    pair["numerator"].detach().float(),
                    self._frozen_numerators[index],
                )
                and torch.equal(
                    pair["denominator"].detach().float(),
                    self._frozen_denominators[index],
                )
            )
            self._router_probes[index] = None
            self._r07_local_participation[index] = torch.stack((
                incoming[index].sum(), sample_count
            ))
            self._r06_local_output_participation[index] = torch.stack((
                outgoing[index].sum(), sample_count
            ))
            self._r02_local_group_participation[index] = torch.stack((
                incoming[index].sum(dim=0),
                outgoing[index].sum(dim=0),
                count,
            ))

    def _layer_response_metric(self, layer_index):
        if layer_index == 0:
            self._prepare_response_batch()
        elif self._router_local_statistics[layer_index] is None:
            raise RuntimeError("batched response transaction was not prepared")
        pair = self.pairs[layer_index]
        return torch.ones(
            (), device=pair["numerator"].device, dtype=torch.float32
        )


class R019150BatchedCompiledResponseInverseInt4Composite(
    R019150LocalLayerOwnerInt4DirectComposite
):
    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        with _CONSTRUCTION_LOCK:
            original = _owner_module.R01Optimizer
            if original is not R01Optimizer:
                raise RuntimeError("R01 owner router was already patched")
            _owner_module.R01Optimizer = _R01BatchedCompiledResponseInverseRouter
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.R01Optimizer = original
        if not isinstance(
            self.router, _R01BatchedCompiledResponseInverseRouter
        ):
            raise RuntimeError("batched compiled-response R01 was not installed")

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "scientific_parent_family_id": (
                "r01_9150_compiled_response_inverse_block256_int4_v1"
            ),
            "response_statistics_executor": (
                "one_owner_local_layer_batched_compiled_program"
            ),
            "response_layer_reductions_mixed": False,
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
    "R019150BatchedCompiledResponseInverseInt4Composite",
    "_R01BatchedCompiledResponseInverseRouter",
    "_batched_response_statistics_program",
)
