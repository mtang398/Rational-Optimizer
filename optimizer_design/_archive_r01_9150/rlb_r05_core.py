"""Learned-response router over the recursively retained R08 B+C geometry.

R05 measures how far each learned P5/Q4 response has rotated away from its
initializer on the *current* normalized preactivations.  Incoming matrices use
the exact normalized-RLB Jacobian response and outgoing matrices use the RLB
feature response.  Their uncentred block-kernel alignments are parameter-free
numbers ``c`` in ``[0, 1]``.

For either matrix role, let ``C`` be the retained R08 B+C coordinate and let
``M`` be the ordinary Muon Nesterov tensor.  R05 forms

    U = C* NS5(C M),
    A = C* D NS5(D C M),

where ``D`` is the positive coordinatewise inverse square root of the matched
``beta2=0.95`` second moment.  The second identity is deliberately enclosed by
both ``C`` and its exact adjoint: ``<M,A> = <D C M,NS5(D C M)> > 0``.  ``A`` is
matched to the Frobenius budget of ``U``.  Since CKA is a squared response
alignment, its canonical amplitudes are ``sqrt(c)`` and ``sqrt(1-c)``; their
blend is matched once more to the budget of ``U``.  This makes the alternate
branch first-order in a small nonconformal response change instead of
incorrectly suppressing it to second order.  The inherited Muon shape
calibration is then applied exactly once.  Thus response morphology selects
geometry, never an LR or WD multiplier.

At the frozen initializer ``c`` is assigned exactly one and the selected
direction is the literal recursively retained R08 B+C direction.  The second
moment is nevertheless updated on every step so the alternate geometry has no
late-start discontinuity.

The current same-slot generation adds a parameter-free paired-role decision.
For each role it constructs a response-angle-, branch-angle-, and descent-
bounded endpoint on the parent's Frobenius sphere.  It then evaluates the four
parent/endpoint role pairs with the exact sampled RLB functional tangent
``D_out h + W_out J D_in x``.  One all-rank direction-only quadratic chooses
the pair; the scheduled LR, inherited Muon shape calibration, and decoupled WD
are each applied once.  Thirty-six structurally inherited aligned samples per
rank supply the decision, with no adjustable sample count or selector scale.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_group_muon_core import _batched_zero_power
from .rlb_r08_core import R08RevisionCore


class R05Core(R08RevisionCore):
    """Current-vs-initial RLB response router over exact B+C coordinates."""

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
        use_response_router: bool = True,
        use_functional_selector: bool = True,
    ):
        if float(beta2) != 0.95:
            raise ValueError("R05 requires the matched beta2=0.95")
        if float(eps) != 1.0e-8:
            raise ValueError("R05 requires the matched eps=1e-8")
        self.beta2 = float(beta2)
        self.adaptive_eps = float(eps)
        self.use_response_router = bool(use_response_router)
        self.use_functional_selector = bool(use_functional_selector)
        # These transient records are initialized before ``super`` because the
        # shared capture constructors install hooks through virtual dispatch.
        # Each training microbatch contributes exactly ``groups`` aligned
        # (x,z,h) rows; this is the already installed R05 probe inventory, not
        # a new sampling hyperparameter.
        pairs = list(pairs)
        self._functional_pending_inputs = [None for _ in pairs]
        self._functional_records = [[] for _ in pairs]
        self._functional_packets = [None for _ in pairs]
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
        )
        self._frozen_numerators = [
            pair["numerator"].detach().float().clone() for pair in self.pairs
        ]
        self._frozen_denominators = [
            pair["denominator"].detach().float().clone() for pair in self.pairs
        ]
        self.param_groups[0]["r05_beta2"] = self.beta2
        self.param_groups[0]["r05_eps"] = self.adaptive_eps
        self.param_groups[0]["r05_use_response_router"] = self.use_response_router
        self.param_groups[0]["r05_use_functional_selector"] = (
            self.use_functional_selector
        )
        self.state[self.incoming[0]]["r05_frozen_numerators"] = torch.stack(
            self._frozen_numerators
        )
        self.state[self.incoming[0]]["r05_frozen_denominators"] = torch.stack(
            self._frozen_denominators
        )
        self._router_probes = [None for _ in self.pairs]
        self._router_local_statistics = [None for _ in self.pairs]
        self._router_exact_initializer = [False for _ in self.pairs]
        self._incoming_coordinate_second_moment = None
        self._outgoing_coordinate_second_moment = None
        self._r05_step = 0
        self._last_corner_choices = None

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "recursive_b_plus_c_parent_lr_scale": 1.0,
            "response_alignment_router_lr_scale": 1.0,
            "second_moment_coordinate_lr_scale": 1.0,
            "alternate_frobenius_budget_scale": 1.0,
            "blend_frobenius_budget_scale": 1.0,
            "endpoint_geometry_lr_scale": 1.0,
            "functional_selector_lr_scale": 1.0,
            "aligned_functional_tangent_lr_scale": 1.0,
        })
        return report

    def _functional_row_indices(self, row_count, device):
        if row_count < self.probe_capture_count:
            raise RuntimeError("R05 functional capture is smaller than its probe")
        numerators = torch.arange(
            self.probe_capture_count, device=device, dtype=torch.int64
        ) * (row_count - 1)
        return torch.div(
            numerators, self.probe_capture_count - 1, rounding_mode="floor"
        )

    def _make_input_hook(self, index):
        parent_capture = super()._make_input_hook(index)

        @torch.no_grad()
        def capture(module, inputs):
            parent_capture(module, inputs)
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("R05 functional input hook received invalid input")
            if self._functional_pending_inputs[index] is not None:
                raise RuntimeError("R05 functional input was not paired with a response")
            flat = inputs[0].detach().reshape(-1, self.external_width)
            indices = self._functional_row_indices(flat.shape[0], flat.device)
            self._functional_pending_inputs[index] = (
                int(flat.shape[0]),
                indices,
                flat.index_select(0, indices),
            )

        return capture

    def _make_feature_hook(self, index):
        parent_capture = super()._make_feature_hook(index)

        @torch.no_grad()
        def capture(module, inputs, output):
            parent_capture(module, inputs, output)
            if not module.training:
                return
            pending = self._functional_pending_inputs[index]
            if pending is None:
                raise RuntimeError("R05 functional response has no aligned input")
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("R05 functional response hook received invalid input")
            if not torch.is_tensor(output):
                raise RuntimeError("R05 functional response hook received invalid output")
            row_count, indices, sampled_input = pending
            flat_z = inputs[0].detach().reshape(-1, self.hidden)
            flat_h = output.detach().reshape(-1, self.hidden)
            if flat_z.shape[0] != row_count or flat_h.shape[0] != row_count:
                raise RuntimeError("R05 functional x/z/h token inventories differ")
            self._functional_records[index].append(
                (
                    sampled_input,
                    flat_z.index_select(0, indices),
                    flat_h.index_select(0, indices),
                )
            )
            self._functional_pending_inputs[index] = None

        return capture

    def _consume_functional_samples(self):
        packets = []
        for index, records in enumerate(self._functional_records):
            pending = self._functional_pending_inputs[index]
            self._functional_pending_inputs[index] = None
            self._functional_records[index] = []
            if pending is not None:
                raise RuntimeError("R05 functional input remained unmatched")
            if len(records) != self.expected_microbatches:
                raise RuntimeError(
                    f"R05 layer {index} did not observe four aligned microbatches"
                )
            inputs = torch.cat([record[0] for record in records], dim=0)
            preactivations = torch.cat([record[1] for record in records], dim=0)
            features = torch.cat([record[2] for record in records], dim=0)
            local_rows = self.probe_capture_count * self.expected_microbatches
            if inputs.shape != (local_rows, self.external_width):
                raise RuntimeError("R05 aligned input inventory changed")
            if preactivations.shape != (local_rows, self.hidden):
                raise RuntimeError("R05 aligned preactivation inventory changed")
            if features.shape != (local_rows, self.hidden):
                raise RuntimeError("R05 aligned feature inventory changed")
            numerators = torch.arange(
                self.probe_count,
                device=inputs.device,
                dtype=torch.int64,
            ) * (local_rows - 1)
            selected = torch.div(
                numerators, self.probe_count - 1, rounding_mode="floor"
            )
            packet = (
                inputs.index_select(0, selected).float(),
                preactivations.index_select(0, selected).float(),
                features.index_select(0, selected).float(),
            )
            self._functional_packets[index] = packet
            packets.append(packet)
        stacked = tuple(torch.stack(items) for items in zip(*packets))
        self._functional_packets = [None for _ in self.pairs]
        return stacked

    def _consume_probe(self, layer_index):
        probe = super()._consume_probe(layer_index)
        self._router_probes[layer_index] = probe
        return probe

    @staticmethod
    def _evaluate_response(u, numerator, denominator):
        """Evaluate the P5/Q4 function and derivative on normalized inputs."""
        t2 = u.square()
        t3 = t2 * u
        t4 = t2.square()
        t5 = t4 * u
        abs_t = u.abs()
        powers = torch.stack((torch.ones_like(u), u, t2, t3, t4, t5), dim=-1)
        derivative_powers = torch.stack(
            (
                torch.zeros_like(u),
                torch.ones_like(u),
                2.0 * u,
                3.0 * t2,
                4.0 * t3,
                5.0 * t4,
            ),
            dim=-1,
        )
        denominator_powers = torch.stack((abs_t, t2, abs_t * t2, t4), dim=-1)
        denominator_derivative_powers = torch.stack(
            (torch.sign(u), 2.0 * u, 3.0 * u * abs_t, 4.0 * t3), dim=-1
        )
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

    @staticmethod
    def _jacobian_kernel_inner(u, function_a, derivative_a, function_b, derivative_b):
        """Return ``<Ja Ja^T, Jb Jb^T>`` without materialising a Jacobian.

        For width ``m``, the normalized-RLB Jacobian is

            J = diag(d) + ((f-u*d)/m) u^T.

        Therefore ``J J^T`` is diagonal plus rank two, so its Frobenius inner
        product costs O(m) per sampled group rather than O(m^2) storage.
        """
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

    @classmethod
    def _response_statistics(
        cls,
        z,
        eps,
        numerator,
        denominator,
        frozen_numerator,
        frozen_denominator,
    ):
        """Return additive ``[role, cross, live_self, frozen_self]`` statistics."""
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + float(eps))
        u = z / rms
        live_f, live_d = cls._evaluate_response(u, numerator, denominator)
        frozen_f, frozen_d = cls._evaluate_response(
            u, frozen_numerator, frozen_denominator
        )

        incoming_cross = cls._jacobian_kernel_inner(
            u, live_f, live_d, frozen_f, frozen_d
        ).sum()
        incoming_live = cls._jacobian_kernel_inner(
            u, live_f, live_d, live_f, live_d
        ).sum()
        incoming_frozen = cls._jacobian_kernel_inner(
            u, frozen_f, frozen_d, frozen_f, frozen_d
        ).sum()

        live_h = rms * live_f
        frozen_h = rms * frozen_f
        outgoing_cross = (live_h * frozen_h).sum(dim=-1).square().sum()
        outgoing_live = live_h.square().sum(dim=-1).square().sum()
        outgoing_frozen = frozen_h.square().sum(dim=-1).square().sum()
        return torch.stack(
            (
                torch.stack((incoming_cross, incoming_live, incoming_frozen)),
                torch.stack((outgoing_cross, outgoing_live, outgoing_frozen)),
            )
        )

    @staticmethod
    def _alignment_from_statistics(statistics, exact_initializer=None):
        numerator = statistics[..., 0]
        denominator = torch.sqrt(
            statistics[..., 1] * statistics[..., 2]
        )
        if torch.any(denominator <= 0.0) or torch.any(~torch.isfinite(statistics)):
            raise RuntimeError("R05 response alignment lost finite positive scale")
        alignment = (numerator / denominator).clamp_(0.0, 1.0)
        if exact_initializer is not None:
            alignment = torch.where(exact_initializer, torch.ones_like(alignment), alignment)
        return alignment

    def _layer_response_metric(self, layer_index):
        # R05 deletes the parent's A metric unconditionally.  Consume its probe
        # directly so the router sees the identical rows without executing the
        # outgoing-Gram/Jacobian metric that would immediately be discarded.
        z = self._consume_probe(layer_index).float().view(
            self.probe_count, self.groups, self.width
        )
        pair = self.pairs[layer_index]
        self._router_local_statistics[layer_index] = self._response_statistics(
            z,
            self.rlb_eps,
            pair["numerator"],
            pair["denominator"],
            self._frozen_numerators[layer_index],
            self._frozen_denominators[layer_index],
        )
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
        identity = torch.eye(self.width, device=z.device, dtype=z.dtype)
        return identity.expand(self.groups, self.width, self.width).clone()

    def _consume_router_alignments(self):
        if any(value is None for value in self._router_local_statistics):
            raise RuntimeError("R05 did not form every layer response alignment")
        statistics = torch.stack(self._router_local_statistics)
        exact = torch.tensor(
            self._router_exact_initializer,
            device=statistics.device,
            dtype=torch.int32,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            dist.all_reduce(exact, op=dist.ReduceOp.MIN)
        exact = exact.bool().unsqueeze(-1).expand(-1, 2)
        alignments = self._alignment_from_statistics(statistics, exact)
        self._router_local_statistics = [None for _ in self.pairs]
        self._router_exact_initializer = [False for _ in self.pairs]
        if not self.use_response_router:
            alignments = torch.ones_like(alignments)
        return alignments

    def _coordinate_inverse_scale(self, coordinate_gradient, *, incoming):
        if incoming:
            moment = self._incoming_coordinate_second_moment
        else:
            moment = self._outgoing_coordinate_second_moment
        if moment is None:
            moment = torch.zeros_like(coordinate_gradient)
            if incoming:
                self._incoming_coordinate_second_moment = moment
                self.state[self.incoming[0]][
                    "r05_incoming_coordinate_second_moment"
                ] = moment
            else:
                self._outgoing_coordinate_second_moment = moment
                self.state[self.outgoing[0]][
                    "r05_outgoing_coordinate_second_moment"
                ] = moment
        if moment.shape != coordinate_gradient.shape:
            raise RuntimeError("R05 coordinate second-moment inventory changed")
        moment.mul_(self.beta2).addcmul_(
            coordinate_gradient,
            coordinate_gradient,
            value=1.0 - self.beta2,
        )
        correction = 1.0 - self.beta2 ** self._r05_step
        return torch.reciprocal(moment.sqrt() / math.sqrt(correction) + self.adaptive_eps)

    @staticmethod
    def _scaled_blockwise_fp64_inner(left, right, block_elements=1 << 20):
        """Accurately reduce one inner product per leading batch item.

        This is a fail-only certificate path.  It never feeds an optimizer
        update: the ordinary FP32 path remains the method.  Scaling each
        operand before the blockwise FP64 accumulation prevents the first-step
        adaptive coordinate from making a valid adjoint identity appear lost
        solely through a large dynamic range and FP32 reduction order.
        """
        if left.shape != right.shape or left.ndim < 2:
            raise RuntimeError("R05 FP64 adjoint certificate shape changed")
        left_flat = left.reshape(left.shape[0], -1)
        right_flat = right.reshape(right.shape[0], -1)
        left_scale = left_flat.abs().amax(dim=1).to(torch.float64)
        right_scale = right_flat.abs().amax(dim=1).to(torch.float64)
        safe_left_scale = left_scale.clamp_min(torch.finfo(torch.float64).tiny)
        safe_right_scale = right_scale.clamp_min(torch.finfo(torch.float64).tiny)
        accumulated = torch.zeros_like(left_scale)
        for start in range(0, left_flat.shape[1], int(block_elements)):
            stop = min(start + int(block_elements), left_flat.shape[1])
            left_block = left_flat[:, start:stop].to(torch.float64)
            right_block = right_flat[:, start:stop].to(torch.float64)
            left_block.div_(safe_left_scale[:, None])
            right_block.div_(safe_right_scale[:, None])
            left_block.mul_(right_block)
            accumulated.add_(left_block.sum(dim=1))
        return accumulated * left_scale * right_scale

    @staticmethod
    def _power_of_two_coordinate_unit(inverse):
        """Remove only a per-matrix positive scalar from ``D``.

        The adaptive branch is matched back to the parent's Frobenius budget,
        so a positive scalar shared by every coordinate of one matrix cancels
        exactly.  Choosing that scalar as a power of two preserves the FP32 and
        BF16 significands while preventing ``C* D`` from forming an enormous
        intermediate on the first step.  Relative coordinate geometry is
        unchanged.
        """
        maximum = inverse.amax(dim=(-2, -1), keepdim=True)
        _mantissa, exponent = torch.frexp(maximum)
        unit = torch.ldexp(torch.ones_like(maximum), exponent)
        return inverse / unit, unit

    @staticmethod
    def _roundoff_compensated_adjoint(
        primal, coordinate_primal, coordinate_dual, adjoint
    ):
        """Restore the bilinear identity lost only to finite precision.

        The hand adjoint is the same VJP returned by autograd for the executed
        triangular-solve composition.  With a highly anisotropic ``D``, its
        parameter-space dot can nevertheless lose digits through cancellation.
        The correction below is the minimum-norm change parallel to the one
        primal being paired and is identically zero in exact arithmetic.
        Two compensated passes account for rounding in the first FP32 axpy.
        """
        target = (coordinate_primal * coordinate_dual).sum(dim=(-2, -1))
        energy = primal.square().sum(dim=(-2, -1)).clamp_min(
            torch.finfo(primal.dtype).tiny
        )
        corrected = adjoint
        total_coefficient = torch.zeros_like(target)
        for _ in range(2):
            observed = (primal * corrected).sum(dim=(-2, -1))
            coefficient = (target - observed) / energy
            corrected = corrected + coefficient[:, None, None] * primal
            total_coefficient = total_coefficient + coefficient
        return corrected, total_coefficient

    @staticmethod
    def _blend_equalized(ordinary, adaptive_equal, alignment):
        """Blend an alternate already matched to the parent budget."""
        ordinary_norm = torch.linalg.vector_norm(
            ordinary, dim=(-2, -1), keepdim=True
        )
        tiny = torch.finfo(ordinary.dtype).tiny
        c = alignment[:, None, None]
        parent_amplitude = torch.sqrt(c)
        adaptive_amplitude = torch.sqrt((1.0 - c).clamp_min(0.0))
        unnormalized = (
            parent_amplitude * ordinary
            + adaptive_amplitude * adaptive_equal
        )
        blend_norm = torch.linalg.vector_norm(
            unnormalized, dim=(-2, -1), keepdim=True
        )
        blend_ratio = ordinary_norm / blend_norm.clamp_min(tiny)
        normalized = unnormalized * blend_ratio
        # This explicit mathematical limit makes c=1 a bitwise parent update.
        direction = torch.where(c == 1.0, ordinary, normalized)
        cosine = (ordinary * adaptive_equal).sum(
            dim=(-2, -1), keepdim=True
        ) / ordinary_norm.square().clamp_min(tiny)
        return (
            direction,
            blend_ratio,
            cosine,
            parent_amplitude,
            adaptive_amplitude,
        )

    @staticmethod
    def _equal_budget_blend(ordinary, adaptive, alignment):
        ordinary_norm = torch.linalg.vector_norm(
            ordinary, dim=(-2, -1), keepdim=True
        )
        adaptive_norm = torch.linalg.vector_norm(
            adaptive, dim=(-2, -1), keepdim=True
        )
        tiny = torch.finfo(ordinary.dtype).tiny
        adaptive_ratio = ordinary_norm / adaptive_norm.clamp_min(tiny)
        adaptive_equal = adaptive * adaptive_ratio
        (
            direction,
            blend_ratio,
            cosine,
            parent_amplitude,
            adaptive_amplitude,
        ) = R05Core._blend_equalized(ordinary, adaptive_equal, alignment)
        return (
            direction,
            adaptive_equal,
            adaptive_ratio,
            blend_ratio,
            cosine,
            parent_amplitude,
            adaptive_amplitude,
        )

    @staticmethod
    def _descent_safe_endpoint(parent, adaptive, momentum, alignment):
        """Move from ``parent`` toward ``adaptive`` on the same Frobenius sphere.

        The half-angle is bounded by the learned-response angle, by the
        parent-to-adaptive arc, and by the exact first-order-descent boundary.
        There is no tunable coefficient.  The implementation uses an
        equivalent two-vector formula that uses and reuses one residual
        workspace; it does not retain a separate full-sized orthogonal basis
        tensor.
        """
        if parent.shape != adaptive.shape or parent.shape != momentum.shape:
            raise RuntimeError("R05 endpoint tensor shapes differ")
        if parent.ndim != 3 or alignment.shape != (parent.shape[0],):
            raise RuntimeError("R05 endpoint batch geometry changed")
        dims = (-2, -1)
        parent_norm = torch.linalg.vector_norm(parent, dim=dims, keepdim=True)
        adaptive_norm = torch.linalg.vector_norm(adaptive, dim=dims, keepdim=True)
        parent_descent = (momentum * parent).sum(dim=dims, keepdim=True)
        cross = (parent * adaptive).sum(dim=dims, keepdim=True)
        tiny = torch.finfo(parent.dtype).tiny
        nonzero = parent_norm > 0.0
        safe_parent_norm = parent_norm.clamp_min(tiny)
        safe_adaptive_norm = adaptive_norm.clamp_min(tiny)
        raw_gamma = torch.where(
            nonzero,
            cross / (safe_parent_norm * safe_adaptive_norm),
            torch.ones_like(cross),
        )
        gamma = raw_gamma.clamp(-1.0, 1.0)
        c = alignment[:, None, None]

        finite = (
            torch.isfinite(parent_norm)
            & torch.isfinite(adaptive_norm)
            & torch.isfinite(parent_descent)
            & torch.isfinite(raw_gamma)
            & torch.isfinite(c)
        )
        valid = finite & ((~nonzero) | (
            (adaptive_norm > 0.0) & (parent_descent > 0.0)
        ))
        valid = valid & (c >= 0.0) & (c <= 1.0)
        torch._assert_async(valid.all())

        machine = torch.finfo(parent.dtype).eps
        collinear_tolerance = machine * (1.0 + gamma.abs())
        antipodal = nonzero & ((1.0 + gamma) <= collinear_tolerance)
        torch._assert_async((~antipodal).all())
        adaptive_scale = parent_norm / safe_adaptive_norm
        # Form the orthogonal residual directly.  ``sqrt(1-gamma^2)`` loses
        # most of its significant bits near either collinear endpoint in
        # FP32, while this executed residual is the quantity the endpoint
        # actually uses and remains well scaled.
        residual = adaptive * adaptive_scale - gamma * parent
        sine = (
            torch.linalg.vector_norm(residual, dim=dims, keepdim=True)
            / safe_parent_norm
        )
        collinear = sine <= collinear_tolerance

        residual_descent = (momentum * residual).sum(dim=dims, keepdim=True)
        safe_sine = sine.clamp_min(tiny)
        safe_descent = parent_descent.clamp_min(tiny)
        descent_cap = (
            residual_descent / (safe_sine * safe_descent)
        ).clamp_min(0.0)
        root_c = torch.sqrt(c)
        response_cap = torch.sqrt((1.0 - c).clamp_min(0.0)) / (1.0 + root_c)
        branch_cap = sine / (1.0 + gamma).clamp_min(tiny)
        half_angle = torch.minimum(
            torch.minimum(response_cap, branch_cap), descent_cap
        )
        active = nonzero & (~collinear) & (residual_descent > 0.0)
        half_angle = torch.where(active, half_angle, torch.zeros_like(half_angle))

        denominator = 1.0 + half_angle.square()
        cosine = (1.0 - half_angle.square()) / denominator
        sine_angle = 2.0 * half_angle / denominator
        sine_ratio = sine_angle / safe_sine
        parent_coefficient = cosine - gamma * sine_ratio
        adaptive_coefficient = sine_ratio * adaptive_scale
        # ``residual`` is dead after the cap calculation.  Reuse its storage
        # for the exact two-vector endpoint so the numerical closure adds one
        # full role tensor rather than several simultaneous copies.
        provisional = residual
        provisional.copy_(parent).mul_(parent_coefficient)
        provisional.addcmul_(adaptive, adaptive_coefficient)
        provisional_norm = torch.linalg.vector_norm(
            provisional, dim=dims, keepdim=True
        )
        parent_endpoint_cross = (parent * provisional).sum(
            dim=dims, keepdim=True
        )
        adaptive_endpoint_cross = (adaptive * provisional).sum(
            dim=dims, keepdim=True
        ) * adaptive_scale
        realized_cosine = parent_endpoint_cross / (
            safe_parent_norm * provisional_norm.clamp_min(tiny)
        )
        realized_orientation = (
            adaptive_endpoint_cross - gamma * parent_endpoint_cross
        )
        angle_tolerance = 64.0 * machine
        angle_valid = (
            (realized_cosine + angle_tolerance >= root_c)
            & (realized_cosine + angle_tolerance >= gamma)
            & (
                realized_orientation
                >= -angle_tolerance * parent_norm.square().clamp_min(1.0)
            )
        )
        provisional.mul_(
            parent_norm / provisional_norm.clamp_min(tiny)
        )
        rematched_descent = (momentum * provisional).sum(
            dim=dims, keepdim=True
        )
        accepted = (
            (half_angle > 0.0)
            & torch.isfinite(provisional_norm)
            & torch.isfinite(realized_cosine)
            & torch.isfinite(realized_orientation)
            & angle_valid
            & torch.isfinite(rematched_descent)
            & (rematched_descent >= parent_descent)
        )
        endpoint = torch.where(accepted, provisional, parent)
        half_angle = torch.where(
            accepted, half_angle, torch.zeros_like(half_angle)
        )
        torch._assert_async(torch.isfinite(endpoint).all())

        endpoint_norm = torch.linalg.vector_norm(endpoint, dim=dims, keepdim=True)
        endpoint_descent = (momentum * endpoint).sum(dim=dims, keepdim=True)
        budget_residual = (
            (endpoint_norm - parent_norm).abs()
            / parent_norm.clamp_min(1.0)
        )
        descent_margin = endpoint_descent - parent_descent
        torch._assert_async((budget_residual <= 64.0 * machine).all())
        torch._assert_async((descent_margin >= 0.0).all())
        return endpoint, {
            "half_angle": half_angle.flatten(),
            "response_cap": response_cap.flatten(),
            "branch_cap": branch_cap.flatten(),
            "descent_cap": descent_cap.flatten(),
            "gamma": gamma.flatten(),
            "budget_residual": budget_residual.flatten(),
            "descent_margin": descent_margin.flatten(),
            "parent_descent": parent_descent.flatten(),
            "endpoint_descent": endpoint_descent.flatten(),
        }

    def _functional_jvp_factors(self, preactivations):
        """Cache the exact normalized P5/Q4 factors shared by both JVPs."""
        layers, samples, hidden = preactivations.shape
        if layers != len(self.pairs) or hidden != self.hidden:
            raise RuntimeError("R05 functional JVP inventory changed")
        z = preactivations.view(layers, samples, self.groups, self.width)
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        u = z / rms
        u2 = u.square()
        u3 = u2 * u
        u4 = u2.square()
        u5 = u4 * u
        abs_u = u.abs()
        numerator = torch.stack(
            [pair["numerator"] for pair in self.pairs]
        ).float()[:, None, :, None, :]
        denominator = torch.stack(
            [pair["denominator"] for pair in self.pairs]
        ).float().abs()[:, None, :, None, :]
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

    def _functional_jvp(self, preactivations, tangent, factors=None):
        """Exact batched JVP of the installed normalized grouped P5/Q4."""
        if preactivations.shape != tangent.shape:
            raise RuntimeError("R05 functional JVP shapes differ")
        if factors is None:
            factors = self._functional_jvp_factors(preactivations)
        u, derivative, radial = factors
        value = tangent.view_as(u)
        projected = (u * value).mean(dim=-1, keepdim=True)
        return (derivative * value + radial * projected).reshape_as(tangent)

    def _incoming_functional_image(
        self, inputs, preactivations, direction, outgoing_weights, factors=None
    ):
        perturbation = torch.bmm(inputs, direction.transpose(1, 2))
        response = self._functional_jvp(
            preactivations, perturbation, factors=factors
        )
        return torch.bmm(response, outgoing_weights.transpose(1, 2))

    @staticmethod
    def _functional_corner_scores(
        incoming_parent_image,
        incoming_endpoint_image,
        outgoing_parent_image,
        outgoing_endpoint_image,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
    ):
        responses = torch.stack(
            (
                incoming_parent_image + outgoing_parent_image,
                incoming_endpoint_image + outgoing_parent_image,
                incoming_parent_image + outgoing_endpoint_image,
                incoming_endpoint_image + outgoing_endpoint_image,
            ),
            dim=1,
        )
        energy_sums = responses.square().sum(dim=(-2, -1))
        descents = torch.stack(
            (
                incoming_parent_descent + outgoing_parent_descent,
                incoming_endpoint_descent + outgoing_parent_descent,
                incoming_parent_descent + outgoing_endpoint_descent,
                incoming_endpoint_descent + outgoing_endpoint_descent,
            ),
            dim=1,
        )
        local_count = torch.tensor(
            [responses.shape[-2]],
            device=responses.device,
            dtype=energy_sums.dtype,
        )
        local_rank_count = torch.ones_like(local_count)
        packet = torch.cat((
            energy_sums.reshape(-1),
            descents.reshape(-1),
            local_count,
            local_rank_count,
        ))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(packet, op=dist.ReduceOp.SUM)
        element_count = energy_sums.numel()
        global_count = packet[-2]
        global_rank_count = packet[-1]
        energy_means = (
            packet[:element_count].view_as(energy_sums) / global_count
        )
        descent_means = (
            packet[element_count:2 * element_count].view_as(descents)
            / global_rank_count
        )
        scores = (
            -float(lr) * descent_means
            + 0.5 * float(lr) ** 2 * energy_means
        )
        torch._assert_async(torch.isfinite(scores).all())
        return scores, energy_means, global_count

    def _select_functional_corner(
        self,
        inputs,
        preactivations,
        features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        force_parent=None,
    ):
        outgoing_weights = torch.stack(self.outgoing).float()
        jvp_factors = self._functional_jvp_factors(preactivations)
        incoming_parent_image = self._incoming_functional_image(
            inputs, preactivations, incoming_parent, outgoing_weights,
            factors=jvp_factors,
        )
        incoming_endpoint_image = self._incoming_functional_image(
            inputs, preactivations, incoming_endpoint, outgoing_weights,
            factors=jvp_factors,
        )
        outgoing_parent_image = torch.bmm(features, outgoing_parent_transpose)
        outgoing_endpoint_image = torch.bmm(features, outgoing_endpoint_transpose)
        scores, energies, global_count = self._functional_corner_scores(
            incoming_parent_image,
            incoming_endpoint_image,
            outgoing_parent_image,
            outgoing_endpoint_image,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
            lr,
        )
        choices = torch.argmin(scores, dim=1)
        if force_parent is not None:
            if force_parent.shape != choices.shape:
                raise RuntimeError("R05 functional-limit mask changed shape")
            choices = torch.where(force_parent, torch.zeros_like(choices), choices)
        if not self.use_functional_selector:
            choices = torch.zeros_like(choices)
        choose_incoming_endpoint = ((choices == 1) | (choices == 3))[:, None, None]
        choose_outgoing_endpoint = (choices >= 2)[:, None, None]
        torch.where(
            choose_incoming_endpoint,
            incoming_endpoint,
            incoming_parent,
            out=incoming_parent,
        )
        torch.where(
            choose_outgoing_endpoint,
            outgoing_endpoint_transpose,
            outgoing_parent_transpose,
            out=outgoing_parent_transpose,
        )
        ordered = torch.topk(scores, k=2, dim=1, largest=False).values
        metadata = {
            "choices": choices,
            "scores": scores,
            "score_margin": ordered[:, 1] - ordered[:, 0],
            "energies": energies,
            "global_count": global_count,
        }
        return incoming_parent, outgoing_parent_transpose, metadata

    def load_state_dict(self, state_dict):
        self._assert_functional_capture_quiescent()
        try:
            raw_groups = state_dict["param_groups"]
            raw_state = state_dict["state"]
            if len(raw_groups) != 1:
                raise RuntimeError("R05 checkpoint optimizer group count changed")
            raw_group = raw_groups[0]
            raw_parameter_ids = list(raw_group["params"])
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("R05 checkpoint optimizer inventory changed") from error

        parameters = self.incoming + self.outgoing
        if len(raw_parameter_ids) != len(parameters):
            raise RuntimeError("R05 checkpoint parameter inventory changed")
        raw_parameter_states = []
        for parameter_id in raw_parameter_ids:
            parameter_state = raw_state.get(parameter_id, {})
            if not isinstance(parameter_state, dict):
                raise RuntimeError("R05 checkpoint parameter state changed")
            raw_parameter_states.append(parameter_state)
        raw_anchor = raw_parameter_states[0]
        raw_outgoing_anchor = raw_parameter_states[len(self.incoming)]

        raw_step_value = raw_anchor.get("r05_step")
        active_payload = any(
            "momentum_buffer" in parameter_state
            for parameter_state in raw_parameter_states
        ) or any(
            key in raw_anchor
            for key in (
                "r05_incoming_coordinate_second_moment",
                "r05_last_corner_choices",
            )
        ) or "r05_outgoing_coordinate_second_moment" in raw_outgoing_anchor
        if raw_step_value is None:
            if active_payload:
                raise RuntimeError("R05 checkpoint lost its active step")
            raw_step = 0
        elif type(raw_step_value) is not int or raw_step_value < 0:
            raise RuntimeError("R05 checkpoint step is invalid")
        else:
            raw_step = raw_step_value
        if raw_step == 0 and active_payload:
            raise RuntimeError("R05 checkpoint active state has step zero")

        def require_tensor(name, value, shape, dtype, *, nonnegative=False):
            if (
                not torch.is_tensor(value)
                or value.shape != shape
                or value.dtype != dtype
            ):
                raise RuntimeError(f"R05 checkpoint {name} inventory changed")
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"R05 checkpoint {name} is nonfinite")
            if nonnegative and not bool((value >= 0).all().item()):
                raise RuntimeError(f"R05 checkpoint {name} is negative")

        raw_frozen_numerators = raw_anchor.get("r05_frozen_numerators")
        raw_frozen_denominators = raw_anchor.get("r05_frozen_denominators")
        require_tensor(
            "frozen numerator",
            raw_frozen_numerators,
            (len(self.pairs), self.groups, 6),
            torch.float32,
        )
        require_tensor(
            "frozen denominator",
            raw_frozen_denominators,
            (len(self.pairs), self.groups, 4),
            torch.float32,
        )

        raw_choices = raw_anchor.get("r05_last_corner_choices")
        raw_incoming_moment = raw_anchor.get(
            "r05_incoming_coordinate_second_moment"
        )
        raw_outgoing_moment = raw_outgoing_anchor.get(
            "r05_outgoing_coordinate_second_moment"
        )
        if raw_step > 0:
            require_tensor(
                "selector",
                raw_choices,
                (len(self.pairs),),
                torch.int64,
            )
            raw_valid_choices = (raw_choices >= 0) & (raw_choices <= 3)
            if not bool(raw_valid_choices.all().item()):
                raise RuntimeError("R05 checkpoint selector choices are invalid")
            coordinate_shape = (
                len(self.pairs), self.hidden, self.external_width
            )
            require_tensor(
                "incoming coordinate moment",
                raw_incoming_moment,
                coordinate_shape,
                torch.float32,
                nonnegative=True,
            )
            require_tensor(
                "outgoing coordinate moment",
                raw_outgoing_moment,
                coordinate_shape,
                torch.float32,
                nonnegative=True,
            )
            for parameter, parameter_state in zip(
                parameters, raw_parameter_states
            ):
                require_tensor(
                    "momentum buffer",
                    parameter_state.get("momentum_buffer"),
                    parameter.shape,
                    parameter.dtype,
                )
        elif any(
            value is not None
            for value in (raw_choices, raw_incoming_moment, raw_outgoing_moment)
        ):
            raise RuntimeError("R05 checkpoint inactive state is inconsistent")

        result = super().load_state_dict(state_dict)
        group = self.param_groups[0]
        if (
            float(group.get("r05_beta2", -1.0)) != self.beta2
            or float(group.get("r05_eps", -1.0)) != self.adaptive_eps
            or bool(group.get("r05_use_response_router", False))
            != self.use_response_router
            or bool(group.get("r05_use_functional_selector", False))
            != self.use_functional_selector
        ):
            raise RuntimeError("R05 checkpoint method constants changed")

        anchor_state = self.state[self.incoming[0]]
        outgoing_anchor_state = self.state[self.outgoing[0]]
        frozen_numerators = raw_frozen_numerators.detach().to(
            device=self.incoming[0].device, dtype=torch.float32, copy=True
        )
        frozen_denominators = raw_frozen_denominators.detach().to(
            device=self.incoming[0].device, dtype=torch.float32, copy=True
        )
        anchor_state["r05_frozen_numerators"] = frozen_numerators
        anchor_state["r05_frozen_denominators"] = frozen_denominators
        self._frozen_numerators = list(frozen_numerators.unbind(0))
        self._frozen_denominators = list(frozen_denominators.unbind(0))

        if raw_step > 0:
            for parameter in parameters:
                buffer = self.state[parameter].get("momentum_buffer")
                if (
                    buffer is None
                    or buffer.shape != parameter.shape
                    or buffer.dtype != parameter.dtype
                    or buffer.device != parameter.device
                ):
                    raise RuntimeError(
                        "R05 loaded momentum buffer inventory changed"
                    )
            incoming_moment = raw_incoming_moment.detach().to(
                device=self.incoming[0].device,
                dtype=torch.float32,
                copy=True,
            )
            outgoing_moment = raw_outgoing_moment.detach().to(
                device=self.outgoing[0].device,
                dtype=torch.float32,
                copy=True,
            )
            saved_choices = raw_choices.detach().to(
                device=self.incoming[0].device,
                dtype=torch.int64,
                copy=True,
            )
            anchor_state[
                "r05_incoming_coordinate_second_moment"
            ] = incoming_moment
            outgoing_anchor_state[
                "r05_outgoing_coordinate_second_moment"
            ] = outgoing_moment
            anchor_state["r05_last_corner_choices"] = saved_choices
        else:
            incoming_moment = None
            outgoing_moment = None
            saved_choices = None

        self._incoming_coordinate_second_moment = incoming_moment
        self._outgoing_coordinate_second_moment = outgoing_moment
        self._r05_step = raw_step
        anchor_state["r05_step"] = raw_step
        self._last_corner_choices = saved_choices
        self._functional_pending_inputs = [None for _ in self.pairs]
        self._functional_records = [[] for _ in self.pairs]
        self._functional_packets = [None for _ in self.pairs]
        return result

    def _assert_functional_capture_quiescent(self):
        if any(value is not None for value in self._functional_pending_inputs):
            raise RuntimeError("R05 checkpoint encountered a pending aligned input")
        if any(records for records in self._functional_records):
            raise RuntimeError("R05 checkpoint encountered aligned forward records")
        if any(packet is not None for packet in self._functional_packets):
            raise RuntimeError("R05 checkpoint encountered an aligned sample packet")

    def state_dict(self):
        self._assert_functional_capture_quiescent()
        return super().state_dict()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R05 refuses a nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        functional_inputs, functional_preactivations, functional_features = (
            self._consume_functional_samples()
        )

        incoming_metrics = torch.stack([
            self._layer_response_metric(index) for index in range(len(self.pairs))
        ])
        alignments = self._consume_router_alignments()
        feature_items = [
            self._consume_feature_moment(index) for index in range(len(self.pairs))
        ]
        outgoing_metrics = torch.stack([item[0] for item in feature_items])
        outgoing_counts = torch.tensor(
            [item[1] for item in feature_items],
            device=outgoing_metrics.device,
            dtype=outgoing_metrics.dtype,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(incoming_metrics, op=dist.ReduceOp.SUM)
            incoming_metrics.div_(dist.get_world_size())
            dist.all_reduce(outgoing_metrics, op=dist.ReduceOp.SUM)
            dist.all_reduce(outgoing_counts, op=dist.ReduceOp.SUM)
        outgoing_metrics.div_(outgoing_counts[:, None, None, None])
        input_metrics, input_counts = self._consume_input_moments()

        (
            incoming_lower,
            incoming_volume,
            hidden_shift,
            _,
            incoming_volume_residual,
        ) = self._unit_volume_cholesky(incoming_metrics, capture_spectrum=False)
        (
            outgoing_lower,
            outgoing_volume,
            _,
            _,
            outgoing_volume_residual,
        ) = self._unit_volume_cholesky(outgoing_metrics, capture_spectrum=False)
        (
            input_lower,
            input_volume,
            input_shift,
            _,
            input_volume_residual,
        ) = self._unit_volume_cholesky(input_metrics, capture_spectrum=False)

        incoming_gradients = torch.stack(
            [parameter.grad for parameter in self.incoming]
        ).float()
        outgoing_gradients = torch.stack(
            [parameter.grad for parameter in self.outgoing]
        ).float()
        incoming_momenta = torch.stack(
            [self._nesterov(parameter) for parameter in self.incoming]
        ).float()
        outgoing_momenta = torch.stack(
            [self._nesterov(parameter) for parameter in self.outgoing]
        ).float()
        layer_count = len(self.pairs)

        incoming_blocks = incoming_momenta.view(
            layer_count, self.groups, self.width, self.external_width
        )
        incoming_work_blocks = self._left_coordinate(
            incoming_lower, incoming_volume, incoming_blocks
        )
        incoming_work = self._right_coordinate(
            input_lower,
            input_volume,
            incoming_work_blocks.reshape_as(incoming_momenta),
        )
        incoming_polar = _batched_zero_power(incoming_work, self.ns_steps).float()
        incoming_pullback = self._right_adjoint(
            input_lower, input_volume, incoming_polar
        )
        incoming_ordinary = self._left_adjoint(
            incoming_lower,
            incoming_volume,
            incoming_pullback.view_as(incoming_blocks),
        ).reshape_as(incoming_momenta)

        outgoing_transpose = outgoing_momenta.transpose(-2, -1)
        outgoing_blocks = outgoing_transpose.view(
            layer_count, self.groups, self.width, self.external_width
        )
        outgoing_work_blocks = self._left_coordinate(
            outgoing_lower, outgoing_volume, outgoing_blocks
        )
        outgoing_work = outgoing_work_blocks.reshape_as(outgoing_transpose)
        outgoing_polar = _batched_zero_power(outgoing_work, self.ns_steps).float()
        outgoing_ordinary_transpose = self._left_adjoint(
            outgoing_lower,
            outgoing_volume,
            outgoing_polar.view_as(outgoing_blocks),
        ).reshape_as(outgoing_transpose)

        incoming_gradient_blocks = incoming_gradients.view_as(incoming_blocks)
        incoming_gradient_work_blocks = self._left_coordinate(
            incoming_lower, incoming_volume, incoming_gradient_blocks
        )
        incoming_gradient_work = self._right_coordinate(
            input_lower,
            input_volume,
            incoming_gradient_work_blocks.reshape_as(incoming_gradients),
        )
        outgoing_gradient_transpose = outgoing_gradients.transpose(-2, -1)
        outgoing_gradient_blocks = outgoing_gradient_transpose.view_as(outgoing_blocks)
        outgoing_gradient_work = self._left_coordinate(
            outgoing_lower, outgoing_volume, outgoing_gradient_blocks
        ).reshape_as(outgoing_gradient_transpose)

        self._r05_step += 1
        self.state[self.incoming[0]]["r05_step"] = self._r05_step
        incoming_inverse = self._coordinate_inverse_scale(
            incoming_gradient_work, incoming=True
        )
        outgoing_inverse = self._coordinate_inverse_scale(
            outgoing_gradient_work, incoming=False
        )
        incoming_stable_inverse, incoming_coordinate_unit = (
            self._power_of_two_coordinate_unit(incoming_inverse)
        )
        outgoing_stable_inverse, outgoing_coordinate_unit = (
            self._power_of_two_coordinate_unit(outgoing_inverse)
        )
        adaptive_coordinate = torch.cat(
            (
                incoming_stable_inverse * incoming_work,
                outgoing_stable_inverse * outgoing_work,
            ),
            dim=0,
        )
        adaptive_polar = _batched_zero_power(
            adaptive_coordinate, self.ns_steps
        ).float()
        incoming_adaptive_polar, outgoing_adaptive_polar = adaptive_polar.split(
            layer_count, dim=0
        )

        incoming_adaptive_coordinate = (
            incoming_stable_inverse * incoming_adaptive_polar
        )
        incoming_adaptive_pullback = self._right_adjoint(
            input_lower, input_volume, incoming_adaptive_coordinate
        )
        incoming_adaptive = self._left_adjoint(
            incoming_lower,
            incoming_volume,
            incoming_adaptive_pullback.view_as(incoming_blocks),
        ).reshape_as(incoming_momenta)
        outgoing_adaptive_coordinate = (
            outgoing_stable_inverse * outgoing_adaptive_polar
        )
        outgoing_adaptive_transpose = self._left_adjoint(
            outgoing_lower,
            outgoing_volume,
            outgoing_adaptive_coordinate.view_as(outgoing_blocks),
        ).reshape_as(outgoing_transpose)

        # Determine the same equal-budget scalar from the provisional
        # pullback, then push it inside the linear coordinate adjoint.  This
        # is algebraically identical to scaling C*D P afterwards, but the
        # realized update and its adjoint certificate now share the exact
        # well-scaled coordinate tensor instead of subtracting huge rounded
        # pullback entries after the fact.
        incoming_ordinary_norm = torch.linalg.vector_norm(
            incoming_ordinary, dim=(-2, -1), keepdim=True
        )
        outgoing_ordinary_norm = torch.linalg.vector_norm(
            outgoing_ordinary_transpose, dim=(-2, -1), keepdim=True
        )
        incoming_adaptive_norm = torch.linalg.vector_norm(
            incoming_adaptive, dim=(-2, -1), keepdim=True
        )
        outgoing_adaptive_norm = torch.linalg.vector_norm(
            outgoing_adaptive_transpose, dim=(-2, -1), keepdim=True
        )
        tiny = torch.finfo(incoming_ordinary.dtype).tiny
        incoming_adaptive_ratio = (
            incoming_ordinary_norm / incoming_adaptive_norm.clamp_min(tiny)
        )
        outgoing_adaptive_ratio = (
            outgoing_ordinary_norm / outgoing_adaptive_norm.clamp_min(tiny)
        )
        incoming_adaptive_equal_coordinate = (
            incoming_adaptive_coordinate * incoming_adaptive_ratio
        )
        incoming_adaptive_equal_pullback = self._right_adjoint(
            input_lower, input_volume, incoming_adaptive_equal_coordinate
        )
        incoming_adaptive_equal = self._left_adjoint(
            incoming_lower,
            incoming_volume,
            incoming_adaptive_equal_pullback.view_as(incoming_blocks),
        ).reshape_as(incoming_momenta)
        outgoing_adaptive_equal_coordinate = (
            outgoing_adaptive_coordinate * outgoing_adaptive_ratio
        )
        outgoing_adaptive_equal_transpose = self._left_adjoint(
            outgoing_lower,
            outgoing_volume,
            outgoing_adaptive_equal_coordinate.view_as(outgoing_blocks),
        ).reshape_as(outgoing_transpose)
        (
            incoming_adaptive_equal,
            incoming_adjoint_compensation,
        ) = self._roundoff_compensated_adjoint(
            incoming_momenta,
            incoming_work,
            incoming_adaptive_equal_coordinate,
            incoming_adaptive_equal,
        )
        (
            outgoing_adaptive_equal_transpose,
            outgoing_adjoint_compensation,
        ) = self._roundoff_compensated_adjoint(
            outgoing_transpose,
            outgoing_work,
            outgoing_adaptive_equal_coordinate,
            outgoing_adaptive_equal_transpose,
        )

        (
            incoming_parent_direction,
            incoming_blend_ratio,
            incoming_cosine,
            incoming_parent_amplitude,
            incoming_adaptive_amplitude,
        ) = self._blend_equalized(
            incoming_ordinary, incoming_adaptive_equal, alignments[:, 0]
        )
        (
            outgoing_parent_direction_transpose,
            outgoing_blend_ratio,
            outgoing_cosine,
            outgoing_parent_amplitude,
            outgoing_adaptive_amplitude,
        ) = self._blend_equalized(
            outgoing_ordinary_transpose,
            outgoing_adaptive_equal_transpose,
            alignments[:, 1],
        )
        outgoing_adaptive_equal = outgoing_adaptive_equal_transpose.transpose(-2, -1)

        incoming_endpoint_direction, incoming_endpoint_metadata = (
            self._descent_safe_endpoint(
                incoming_parent_direction,
                incoming_adaptive_equal,
                incoming_momenta,
                alignments[:, 0],
            )
        )
        outgoing_endpoint_direction_transpose, outgoing_endpoint_metadata = (
            self._descent_safe_endpoint(
                outgoing_parent_direction_transpose,
                outgoing_adaptive_equal_transpose,
                outgoing_transpose,
                alignments[:, 1],
            )
        )

        matched_scale = 0.2 * math.sqrt(max(self.hidden, self.external_width))
        incoming_parent_direction.mul_(matched_scale)
        incoming_endpoint_direction.mul_(matched_scale)
        outgoing_parent_direction_transpose.mul_(matched_scale)
        outgoing_endpoint_direction_transpose.mul_(matched_scale)
        previous_corner_choices = self._last_corner_choices
        incoming_direction, outgoing_direction_transpose, selector_metadata = (
            self._select_functional_corner(
                functional_inputs,
                functional_preactivations,
                functional_features,
                incoming_parent_direction,
                incoming_endpoint_direction,
                outgoing_parent_direction_transpose,
                outgoing_endpoint_direction_transpose,
                incoming_endpoint_metadata["parent_descent"] * matched_scale,
                incoming_endpoint_metadata["endpoint_descent"] * matched_scale,
                outgoing_endpoint_metadata["parent_descent"] * matched_scale,
                outgoing_endpoint_metadata["endpoint_descent"] * matched_scale,
                lr,
                force_parent=(alignments[:, 0] == 1.0)
                & (alignments[:, 1] == 1.0),
            )
        )
        outgoing_direction = outgoing_direction_transpose.transpose(-2, -1)
        self._last_corner_choices = selector_metadata["choices"].detach().clone()
        self.state[self.incoming[0]]["r05_last_corner_choices"] = (
            self._last_corner_choices
        )
        if previous_corner_choices is None:
            selector_flip_count = torch.zeros(
                (), device=incoming_direction.device, dtype=torch.int64
            )
        else:
            selector_flip_count = (
                previous_corner_choices != selector_metadata["choices"]
            ).sum()

        # These two assertions stay on every transition but remain queued on
        # the CUDA stream; unlike Python ``torch.any`` branches they do not
        # introduce a host synchronization.  All descent/adjoint identities
        # are algebraic properties of the unchanged update and are evaluated
        # on the preregistered telemetry transitions below.  They never feed
        # an update tensor, so omitting their redundant ordinary-step
        # reductions is an equation-preserving execution optimization.
        torch._assert_async(torch.isfinite(incoming_direction).all())
        torch._assert_async(torch.isfinite(outgoing_direction).all())

        if self._capture_telemetry_next_step:
            incoming_ordinary_descent = (
                incoming_momenta * incoming_ordinary
            ).sum(dim=(-2, -1))
            outgoing_ordinary_descent = (
                outgoing_transpose * outgoing_ordinary_transpose
            ).sum(dim=(-2, -1))
            incoming_adaptive_descent = (
                incoming_momenta * incoming_adaptive_equal
            ).sum(dim=(-2, -1))
            outgoing_adaptive_descent = (
                outgoing_momenta * outgoing_adaptive_equal
            ).sum(dim=(-2, -1))
            incoming_descent = (
                incoming_momenta * incoming_direction
            ).sum(dim=(-2, -1))
            outgoing_descent = (
                outgoing_momenta * outgoing_direction
            ).sum(dim=(-2, -1))

            incoming_coordinate_descent = (
                incoming_work * incoming_polar
            ).sum(dim=(-2, -1))
            outgoing_coordinate_descent = (
                outgoing_work * outgoing_polar
            ).sum(dim=(-2, -1))
            incoming_adaptive_coordinate_descent = (
                incoming_work * incoming_adaptive_equal_coordinate
            ).sum(dim=(-2, -1))
            outgoing_adaptive_coordinate_descent = (
                outgoing_work * outgoing_adaptive_equal_coordinate
            ).sum(dim=(-2, -1))
            incoming_adjoint_residual = (
                (incoming_ordinary_descent - incoming_coordinate_descent).abs()
                / incoming_coordinate_descent.abs().clamp_min(1.0e-20)
            )
            outgoing_adjoint_residual = (
                (outgoing_ordinary_descent - outgoing_coordinate_descent).abs()
                / outgoing_coordinate_descent.abs().clamp_min(1.0e-20)
            )
            incoming_adaptive_adjoint_residual = (
                (incoming_adaptive_descent - incoming_adaptive_coordinate_descent).abs()
                / incoming_adaptive_coordinate_descent.abs().clamp_min(1.0e-20)
            )
            outgoing_adaptive_adjoint_residual = (
                (outgoing_adaptive_descent - outgoing_adaptive_coordinate_descent).abs()
                / outgoing_adaptive_coordinate_descent.abs().clamp_min(1.0e-20)
            )

            if (
                torch.any(alignments < 0.0)
                or torch.any(alignments > 1.0)
                or torch.any(incoming_ordinary_descent <= 0.0)
                or torch.any(outgoing_ordinary_descent <= 0.0)
                or torch.any(incoming_adaptive_descent <= 0.0)
                or torch.any(outgoing_adaptive_descent <= 0.0)
                or torch.any(incoming_descent <= 0.0)
                or torch.any(outgoing_descent <= 0.0)
            ):
                raise RuntimeError("R05 response-routed LMO lost finite descent")
            all_adjoint_residuals = torch.stack(
                (
                    incoming_adjoint_residual,
                    outgoing_adjoint_residual,
                    incoming_adaptive_adjoint_residual,
                    outgoing_adaptive_adjoint_residual,
                )
            )
            adjoint_fp64_fallback = False
        if (
            self._capture_telemetry_next_step
            and torch.any(all_adjoint_residuals > 3.0e-4)
        ):
            # The first-step D coordinate can have a very large dynamic range.
            # Recheck only a failing FP32 certificate with a scaled blockwise
            # FP64 reduction; none of these values feeds the update itself.
            adjoint_fp64_fallback = True
            fp32_adjoint_residuals = all_adjoint_residuals
            incoming_ordinary_lhs64 = self._scaled_blockwise_fp64_inner(
                incoming_momenta, incoming_ordinary
            )
            incoming_ordinary_rhs64 = self._scaled_blockwise_fp64_inner(
                incoming_work, incoming_polar
            )
            outgoing_ordinary_lhs64 = self._scaled_blockwise_fp64_inner(
                outgoing_momenta,
                outgoing_ordinary_transpose.transpose(-2, -1),
            )
            outgoing_ordinary_rhs64 = self._scaled_blockwise_fp64_inner(
                outgoing_work, outgoing_polar
            )
            incoming_adaptive_lhs64 = self._scaled_blockwise_fp64_inner(
                incoming_momenta, incoming_adaptive_equal
            )
            incoming_adaptive_rhs64 = self._scaled_blockwise_fp64_inner(
                incoming_work, incoming_adaptive_equal_coordinate
            )
            outgoing_adaptive_lhs64 = self._scaled_blockwise_fp64_inner(
                outgoing_momenta, outgoing_adaptive_equal
            )
            outgoing_adaptive_rhs64 = self._scaled_blockwise_fp64_inner(
                outgoing_work, outgoing_adaptive_equal_coordinate
            )
            fp64_lhs = torch.stack((
                incoming_ordinary_lhs64,
                outgoing_ordinary_lhs64,
                incoming_adaptive_lhs64,
                outgoing_adaptive_lhs64,
            ))
            fp64_rhs = torch.stack((
                incoming_ordinary_rhs64,
                outgoing_ordinary_rhs64,
                incoming_adaptive_rhs64,
                outgoing_adaptive_rhs64,
            ))
            all_adjoint_residuals = (
                (fp64_lhs - fp64_rhs).abs()
                / fp64_rhs.abs().clamp_min(torch.finfo(torch.float64).tiny)
            )
            residual_names = (
                "incoming_ordinary",
                "outgoing_ordinary",
                "incoming_adaptive",
                "outgoing_adaptive",
            )
            diagnostics = {
                "event": "r05_adjoint_fp64_fallback",
                "names": residual_names,
                "fp32_max": [
                    float(value) for value in fp32_adjoint_residuals.amax(dim=1).tolist()
                ],
                "fp32_worst_layer": [
                    int(value) for value in fp32_adjoint_residuals.argmax(dim=1).tolist()
                ],
                "fp64_max": [
                    float(value) for value in all_adjoint_residuals.amax(dim=1).tolist()
                ],
                "fp64_worst_layer": [
                    int(value) for value in all_adjoint_residuals.argmax(dim=1).tolist()
                ],
            }
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            if rank == 0:
                print(diagnostics, flush=True)
            if torch.any(all_adjoint_residuals > 3.0e-4):
                raise RuntimeError(
                    f"R05 coordinate adjoint identity was lost: {diagnostics}"
                )

        for index, (incoming, outgoing) in enumerate(zip(self.incoming, self.outgoing)):
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming.add_(incoming_direction[index].to(incoming.dtype), alpha=-lr)
            outgoing.add_(outgoing_direction[index].to(outgoing.dtype), alpha=-lr)

        if self._capture_telemetry_next_step:
            disagreement = torch.sqrt((1.0 - torch.cat(
                (incoming_cosine, outgoing_cosine), dim=0
            ).square()).clamp_min(0.0))
            self._last_telemetry = {
                "rlb_r05_group_count": int(self.groups),
                "rlb_r05_group_width": int(self.width),
                "rlb_r05_probe_microbatches": int(self._last_probe_record_count),
                "rlb_r05_feature_microbatches": int(self._last_feature_record_count),
                "rlb_r05_feature_local_sample_count": int(self._last_feature_sample_count),
                "rlb_r05_input_capture_per_microbatch": int(self.input_capture_count),
                "rlb_r05_input_microbatches": int(self._last_input_record_count),
                "rlb_r05_input_local_sample_count": int(self._last_input_local_sample_count),
                "rlb_r05_input_global_sample_count": int(input_counts[0].item()),
                "rlb_r05_functional_local_sample_count": int(self.probe_count),
                "rlb_r05_functional_global_sample_count": int(
                    selector_metadata["global_count"].item()
                ),
                "rlb_r05_step": int(self._r05_step),
                "rlb_r05_incoming_alignment_min": float(alignments[:, 0].amin().item()),
                "rlb_r05_incoming_alignment_median": float(alignments[:, 0].median().item()),
                "rlb_r05_incoming_alignment_max": float(alignments[:, 0].amax().item()),
                "rlb_r05_outgoing_alignment_min": float(alignments[:, 1].amin().item()),
                "rlb_r05_outgoing_alignment_median": float(alignments[:, 1].median().item()),
                "rlb_r05_outgoing_alignment_max": float(alignments[:, 1].amax().item()),
                "rlb_r05_cka_departure_max": float((1.0 - alignments).amax().item()),
                "rlb_r05_router_activity_max": float(torch.maximum(
                    incoming_adaptive_amplitude,
                    outgoing_adaptive_amplitude,
                ).amax().item()),
                "rlb_r05_parent_amplitude_min": float(torch.minimum(
                    incoming_parent_amplitude,
                    outgoing_parent_amplitude,
                ).amin().item()),
                "rlb_r05_branch_disagreement_max": float(disagreement.amax().item()),
                "rlb_r05_incoming_adaptive_norm_ratio_max": float(incoming_adaptive_ratio.amax().item()),
                "rlb_r05_outgoing_adaptive_norm_ratio_max": float(outgoing_adaptive_ratio.amax().item()),
                "rlb_r05_blend_norm_ratio_max": float(torch.maximum(
                    incoming_blend_ratio, outgoing_blend_ratio
                ).amax().item()),
                "rlb_r05_incoming_endpoint_half_angle_max": float(
                    incoming_endpoint_metadata["half_angle"].amax().item()
                ),
                "rlb_r05_outgoing_endpoint_half_angle_max": float(
                    outgoing_endpoint_metadata["half_angle"].amax().item()
                ),
                "rlb_r05_endpoint_budget_residual_max": float(torch.maximum(
                    incoming_endpoint_metadata["budget_residual"].amax(),
                    outgoing_endpoint_metadata["budget_residual"].amax(),
                ).item()),
                "rlb_r05_endpoint_descent_margin_min": float(torch.minimum(
                    incoming_endpoint_metadata["descent_margin"].amin(),
                    outgoing_endpoint_metadata["descent_margin"].amin(),
                ).item()),
                "rlb_r05_endpoint_response_cap_max": float(torch.maximum(
                    incoming_endpoint_metadata["response_cap"].amax(),
                    outgoing_endpoint_metadata["response_cap"].amax(),
                ).item()),
                "rlb_r05_endpoint_branch_cap_max": float(torch.maximum(
                    incoming_endpoint_metadata["branch_cap"].amax(),
                    outgoing_endpoint_metadata["branch_cap"].amax(),
                ).item()),
                "rlb_r05_endpoint_descent_cap_max": float(torch.maximum(
                    incoming_endpoint_metadata["descent_cap"].amax(),
                    outgoing_endpoint_metadata["descent_cap"].amax(),
                ).item()),
                "rlb_r05_selector_parent_count": int(
                    (selector_metadata["choices"] == 0).sum().item()
                ),
                "rlb_r05_selector_incoming_count": int(
                    (selector_metadata["choices"] == 1).sum().item()
                ),
                "rlb_r05_selector_outgoing_count": int(
                    (selector_metadata["choices"] == 2).sum().item()
                ),
                "rlb_r05_selector_joint_count": int(
                    (selector_metadata["choices"] == 3).sum().item()
                ),
                "rlb_r05_selector_flip_count": int(selector_flip_count.item()),
                "rlb_r05_selector_score_margin_min": float(
                    selector_metadata["score_margin"].amin().item()
                ),
                "rlb_r05_selector_score_margin_median": float(
                    selector_metadata["score_margin"].median().item()
                ),
                "rlb_r05_selector_selected_minus_parent_max": float((
                    selector_metadata["scores"].gather(
                        1, selector_metadata["choices"][:, None]
                    ).squeeze(1) - selector_metadata["scores"][:, 0]
                ).amax().item()),
                "rlb_r05_hidden_precision_shift_relative": float(hidden_shift),
                "rlb_r05_input_precision_shift_relative": float(input_shift),
                "rlb_r05_incoming_volume_residual_max": float(incoming_volume_residual.amax().item()),
                "rlb_r05_outgoing_volume_residual_max": float(outgoing_volume_residual.amax().item()),
                "rlb_r05_input_volume_residual_max": float(input_volume_residual.amax().item()),
                "rlb_r05_incoming_ordinary_descent_min": float(incoming_ordinary_descent.amin().item()),
                "rlb_r05_outgoing_ordinary_descent_min": float(outgoing_ordinary_descent.amin().item()),
                "rlb_r05_incoming_adaptive_descent_min": float(incoming_adaptive_descent.amin().item()),
                "rlb_r05_outgoing_adaptive_descent_min": float(outgoing_adaptive_descent.amin().item()),
                "rlb_r05_incoming_blend_descent_min": float(incoming_descent.amin().item()),
                "rlb_r05_outgoing_blend_descent_min": float(outgoing_descent.amin().item()),
                "rlb_r05_adjoint_residual_max": float(all_adjoint_residuals.amax().item()),
                "rlb_r05_adjoint_fp64_fallback": int(adjoint_fp64_fallback),
                "rlb_r05_incoming_coordinate_unit_max": float(
                    incoming_coordinate_unit.amax().item()
                ),
                "rlb_r05_outgoing_coordinate_unit_max": float(
                    outgoing_coordinate_unit.amax().item()
                ),
                "rlb_r05_adjoint_compensation_max": float(torch.maximum(
                    incoming_adjoint_compensation.abs().amax(),
                    outgoing_adjoint_compensation.abs().amax(),
                ).item()),
            }
        self._capture_telemetry_next_step = False
        return loss
