"""Learned-response radial natural geometry on complete R01.

Global-RLB gives hidden coordinate ``i`` two canonical radial variables: the
norm of incoming row ``A[i,:]`` and the norm of the paired outgoing column
``B[:,i]``.  Their sampled first-order response signals are different:

    s_A = z_i * (f'_i(u) + (f_i(u)-u_i f'_i(u)) u_i / width)
    s_B = rho * f_i(u).

The first expression is the exact self-coordinate diagonal of the installed
group-RMS P5/Q4 Jacobian; the second is the exact current RLB feature.  Their
2 x 2 empirical Gram matrix therefore says whether the two radial variables
are functionally redundant (the near-linear rank-one case) or whether the
current learned rational shape exposes a second response direction.

R08 applies the Moore--Penrose inverse square root of that current response
Gram to the matched-beta Adam direction of the two radial gradients.  There
is no damping or route threshold: numerical rank is determined only at the
standard floating-point pseudoinverse tolerance.  The resulting paired role
direction is made orthogonal to complete R01 and matched to its exact group
Frobenius budget.  The inherited orthogonal sum/difference atlas and global
downstream-loss transaction then select the complete update.  The all-ones
atlas coefficient remains literal R01, scheduled LR and WD are each applied
once, and every internal LR/WD scale is one.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_r01_core import R01Core
from .rlb_r05_next_core import R05NextCore


class R08NextCore(R05NextCore):
    """R01 plus a current-P5/Q4 two-role radial natural atlas."""

    component_code = 15
    checkpoint_schema = "r08_r01_learned_response_radial_natural_atlas_v1"
    inherited_parent = "current_r01_global_cross_layer_rlb_metric"
    new_scientific_components = (
        "paired_hidden_radial_adam_state",
        "current_p5_q4_two_role_response_pseudoinverse_geometry",
    )

    def __init__(self, pairs, **kwargs):
        self._r08_inverse_sqrt = None
        self._r08_role_direction = None
        self._r08_response_metadata = None
        super().__init__(pairs, **kwargs)

    def lr_wd_fairness_audit(self):
        report = R01Core.lr_wd_fairness_audit(self)
        report.update({
            "paired_radial_adam_lr_scale": 1.0,
            "paired_radial_adam_weight_decay_scale": 1.0,
            "current_p5_q4_response_geometry_lr_scale": 1.0,
            "response_pseudoinverse_lr_scale": 1.0,
            "radial_natural_atlas_budget_lr_scale": 1.0,
            "inherited_r01_budget_lr_scale": 1.0,
        })
        return report

    @staticmethod
    def _response_pseudoinverse_sqrt(a_signal, b_signal):
        """Return the all-rank 2 x 2 response Gram pseudoinverse square root."""
        if a_signal.shape != b_signal.shape or a_signal.ndim != 4:
            raise RuntimeError("R08 response-signal inventory changed")
        sums = torch.stack((
            a_signal.square().sum(dim=1),
            (a_signal * b_signal).sum(dim=1),
            b_signal.square().sum(dim=1),
        ), dim=-1)
        count = torch.tensor(
            float(a_signal.shape[1]),
            device=a_signal.device,
            dtype=a_signal.dtype,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        torch._assert_async(torch.isfinite(count) & (count > 0.0))
        moments = sums / count
        gram = torch.empty(
            *moments.shape[:-1], 2, 2,
            device=moments.device,
            dtype=moments.dtype,
        )
        gram[..., 0, 0] = moments[..., 0]
        gram[..., 0, 1] = moments[..., 1]
        gram[..., 1, 0] = moments[..., 1]
        gram[..., 1, 1] = moments[..., 2]
        # Exact analytic spectral calculus for symmetric 2 x 2 matrices.
        # Besides removing launch overhead, this avoids cuSOLVER's invalid-
        # batch limitation for the exact-M1 82,944-matrix inventory.  It is
        # algebraically the same Moore--Penrose inverse square root.
        aa, ab, bb = moments.unbind(dim=-1)
        trace = aa + bb
        gap = torch.sqrt((aa - bb).square() + 4.0 * ab.square())
        low = (0.5 * (trace - gap)).clamp_min(0.0)
        high = (0.5 * (trace + gap)).clamp_min(0.0)
        maximum = high
        tolerance = maximum * (2.0 * torch.finfo(gram.dtype).eps)
        low_active = low > tolerance
        high_active = high > tolerance
        low_inverse_root = torch.where(
            low_active,
            torch.rsqrt(low.clamp_min(torch.finfo(gram.dtype).tiny)),
            torch.zeros_like(low),
        )
        high_inverse_root = torch.where(
            high_active,
            torch.rsqrt(high.clamp_min(torch.finfo(gram.dtype).tiny)),
            torch.zeros_like(high),
        )
        distinct = gap > tolerance
        safe_gap = gap.clamp_min(torch.finfo(gram.dtype).tiny)
        high_projector_00 = (aa - low) / safe_gap
        high_projector_01 = ab / safe_gap
        high_projector_11 = (bb - low) / safe_gap
        difference = high_inverse_root - low_inverse_root
        inverse_sqrt = torch.empty_like(gram)
        inverse_sqrt[..., 0, 0] = (
            low_inverse_root + difference * high_projector_00
        )
        inverse_sqrt[..., 0, 1] = difference * high_projector_01
        inverse_sqrt[..., 1, 0] = difference * high_projector_01
        inverse_sqrt[..., 1, 1] = (
            low_inverse_root + difference * high_projector_11
        )
        # When both eigenvalues coincide, the matrix is a scalar identity and
        # the projector orientation is immaterial.
        repeated = high_inverse_root
        inverse_sqrt[..., 0, 0] = torch.where(
            distinct, inverse_sqrt[..., 0, 0], repeated
        )
        inverse_sqrt[..., 0, 1] = torch.where(
            distinct, inverse_sqrt[..., 0, 1], torch.zeros_like(repeated)
        )
        inverse_sqrt[..., 1, 0] = torch.where(
            distinct, inverse_sqrt[..., 1, 0], torch.zeros_like(repeated)
        )
        inverse_sqrt[..., 1, 1] = torch.where(
            distinct, inverse_sqrt[..., 1, 1], repeated
        )
        rank = low_active.to(torch.int64) + high_active.to(torch.int64)
        positive_minimum = torch.where(low_active, low, high)
        condition = maximum / positive_minimum.clamp_min(
            torch.finfo(gram.dtype).tiny
        )
        condition = torch.where(rank > 0, condition, torch.ones_like(condition))
        torch._assert_async(
            torch.isfinite(gram).all()
            & torch.isfinite(inverse_sqrt).all()
            & torch.isfinite(condition).all()
            & (rank > 0).all()
        )
        return inverse_sqrt, {
            "rank": rank,
            "condition": condition,
            "count": count,
            "cross_correlation": moments[..., 1] / torch.sqrt(
                moments[..., 0] * moments[..., 2]
            ).clamp_min(torch.finfo(gram.dtype).tiny),
        }

    def _paired_magnitude_state(self, summed_gradients):
        """Matched-beta two-role Adam followed by current-response whitening."""
        expected = (len(self.pairs), self.groups, self.width)
        if summed_gradients.shape != expected or self._r08_inverse_sqrt is None:
            raise RuntimeError("R08 paired-radial state inventory changed")
        shape = expected + (self.external_width,)
        incoming = torch.stack(self.incoming).float().view(shape)
        outgoing = torch.stack(self.outgoing).float().transpose(-2, -1).view(shape)
        incoming_unit = incoming / torch.linalg.vector_norm(
            incoming, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(incoming.dtype).tiny)
        outgoing_unit = outgoing / torch.linalg.vector_norm(
            outgoing, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(outgoing.dtype).tiny)
        incoming_gradients = torch.stack([
            parameter.grad for parameter in self.incoming
        ]).float().view(shape)
        outgoing_gradients = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float().transpose(-2, -1).view(shape)
        radial_gradients = torch.stack((
            (incoming_gradients * incoming_unit).sum(dim=-1),
            (outgoing_gradients * outgoing_unit).sum(dim=-1),
        ), dim=-1)
        reconstructed = radial_gradients.sum(dim=-1)
        error = (reconstructed - summed_gradients).abs().amax()
        scale = reconstructed.abs().amax().clamp_min(1.0)
        torch._assert_async(
            error <= 512.0 * torch.finfo(reconstructed.dtype).eps * scale
        )

        anchor = self.state[self.incoming[0]]
        first = anchor.get("r08_radial_first")
        second = anchor.get("r08_radial_second")
        step = anchor.get("r08_radial_step", 0)
        if first is None:
            first = torch.zeros_like(radial_gradients)
            second = torch.zeros_like(radial_gradients)
        if (
            first.shape != radial_gradients.shape
            or second is None
            or second.shape != radial_gradients.shape
            or type(step) is not int
            or step < 0
        ):
            raise RuntimeError("R08 paired-radial checkpoint state changed")
        next_step = step + 1
        first.mul_(self._r05_beta1).add_(
            radial_gradients, alpha=1.0 - self._r05_beta1
        )
        second.mul_(self._r05_beta2).addcmul_(
            radial_gradients,
            radial_gradients,
            value=1.0 - self._r05_beta2,
        )
        bias1 = 1.0 - self._r05_beta1 ** next_step
        bias2 = 1.0 - self._r05_beta2 ** next_step
        adam_direction = (first / bias1) / (
            torch.sqrt(second / bias2) + self._r05_eps
        )
        role_direction = torch.einsum(
            "...ij,...j->...i", self._r08_inverse_sqrt, adam_direction
        )
        torch._assert_async(
            torch.isfinite(adam_direction).all()
            & torch.isfinite(role_direction).all()
        )
        anchor["r08_radial_first"] = first
        anchor["r08_radial_second"] = second
        anchor["r08_radial_step"] = next_step
        self._r08_role_direction = role_direction
        # R05's outer atlas inventory expects one channel scalar.  The R08
        # lift below consumes the complete two-role tensor; this norm is only
        # a finite, shape-correct carrier and has no optimizer action.
        carrier = torch.linalg.vector_norm(role_direction, dim=-1)
        return carrier, next_step

    def _orthogonal_equal_budget_magnitude(
        self,
        parent_incoming,
        parent_outgoing,
        unit_incoming,
        unit_outgoing,
        magnitude_direction,
    ):
        if (
            self._r08_role_direction is None
            or self._r08_role_direction.shape != magnitude_direction.shape + (2,)
        ):
            raise RuntimeError("R08 paired-role lift inventory changed")
        role = self._r08_role_direction
        ones = torch.ones_like(magnitude_direction)
        return R05NextCore._orthogonal_equal_budget_magnitude(
            parent_incoming,
            parent_outgoing,
            unit_incoming * role[..., 0, None],
            unit_outgoing * role[..., 1, None],
            ones,
        )

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        *args,
        **kwargs,
    ):
        if functional_preactivations is None or functional_features is None:
            raise RuntimeError("R08 did not receive current response samples")
        factors = self._functional_jvp_factors(functional_preactivations)
        u, derivative, radial = factors
        z = functional_preactivations.view_as(u)
        features = functional_features.view_as(u)
        self_derivative = derivative + radial * u / float(self.width)
        a_signal = self_derivative * z
        b_signal = features
        inverse_sqrt, metadata = self._response_pseudoinverse_sqrt(
            a_signal, b_signal
        )
        self._r08_inverse_sqrt = inverse_sqrt
        self._r08_response_metadata = metadata
        return super()._select_functional_corner(
            functional_inputs,
            functional_preactivations,
            functional_features,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def step(self, closure=None):
        self._r08_inverse_sqrt = None
        self._r08_role_direction = None
        self._r08_response_metadata = None
        loss = super().step(closure)
        metadata = self._r08_response_metadata
        if metadata is None:
            raise RuntimeError("R08 did not execute its learned-response geometry")
        if "rlb_r05_component_code" in self._last_telemetry:
            inherited = {
                key.replace("rlb_r05_", "rlb_r08_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r05_")
            }
            self._last_telemetry.update(inherited)
            for key in tuple(self._last_telemetry):
                if key.startswith("rlb_r05_"):
                    del self._last_telemetry[key]
            rank = metadata["rank"]
            condition = metadata["condition"]
            correlation = metadata["cross_correlation"]
            self._last_telemetry.update({
                "rlb_r08_component_code": self.component_code,
                "rlb_r08_parent_is_complete_r01": 1,
                "rlb_r08_paired_radial_adam_enabled": 1,
                "rlb_r08_current_p5_q4_response_geometry_enabled": 1,
                "rlb_r08_response_pseudoinverse_enabled": 1,
                "rlb_r08_response_rank_min": int(rank.amin().item()),
                "rlb_r08_response_rank_median": int(rank.median().item()),
                "rlb_r08_response_rank_max": int(rank.amax().item()),
                "rlb_r08_response_condition_max": float(condition.amax().item()),
                "rlb_r08_response_correlation_min": float(correlation.amin().item()),
                "rlb_r08_response_correlation_median": float(correlation.median().item()),
                "rlb_r08_response_correlation_max": float(correlation.amax().item()),
                "rlb_r08_response_global_sample_count": int(
                    metadata["count"].item()
                ),
            })
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._r08_inverse_sqrt = None
        self._r08_role_direction = None
        self._r08_response_metadata = None
        return result


__all__ = ("R08NextCore",)
