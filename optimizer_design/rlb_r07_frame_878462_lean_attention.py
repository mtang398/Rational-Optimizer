"""Method1 attention with certificates evaluated only when published.

The Method1 R02 attention transaction computes condition numbers, branch
angles, per-branch descents, and residual certificates after its update
direction has already been fixed.  Those tensors are read only by telemetry.
This execution-only subclass retains the complete historical path on every
telemetry transition and stops the ordinary path at the same direction.

The factorized moment recurrence, R05/R06 branches, two NS5 calls per role,
chord coefficients, parameter update, LR, WD, and checkpoint state are
unchanged.  Full-tensor guard reductions that only raise after those values
are determined also remain on telemetry transitions; ordinary finite-domain
transitions omit them.  Ordinary transitions are accepted only after bitwise
parameter-and-state equivalence tests against the qualified Method1 path.
"""

from __future__ import annotations

import torch

from ._method1_metric2_approx.rlb_group_muon_core import (
    _batched_zero_power,
    _match_rms_adamw_scale,
)
from ._method1_metric2_approx.rlb_r04_core import R04AttentionCore
from ._method1_metric2_approx.rlb_r05_revision_core import (
    R05RevisionRouterCore,
)
from .rlb_r07_frame_878462_metric2 import Method1Metric2AttentionOptimizer


FAMILY_ID = "method1_878462_telemetry_certificate_elision_v1"


class _LeanR02AttentionMixin:
    """Reusable execution repair for the qualified R02 attention equation."""

    def _ordinary_adaptive_source(self, role, gradients, momenta, step):
        anchor_state = self.state[self.role_parameters[role][0]]
        row_key = f"r04_{role}_row_second_moment"
        column_key = f"r04_{role}_column_second_moment"
        rows = anchor_state.get(row_key)
        columns = anchor_state.get(column_key)
        row_shape = gradients.shape[:-1]
        column_shape = (gradients.shape[0], gradients.shape[-1])
        if rows is None:
            rows = torch.zeros(
                row_shape, device=gradients.device, dtype=torch.float32
            )
            columns = torch.zeros(
                column_shape, device=gradients.device, dtype=torch.float32
            )
            anchor_state[row_key] = rows
            anchor_state[column_key] = columns
        if (
            rows.shape != row_shape
            or columns is None
            or columns.shape != column_shape
        ):
            raise RuntimeError("R04 factorized second-moment inventory changed")

        # Preserve the qualified operation sequence up to adaptive_equal.
        squared = gradients.square()
        rows.mul_(self.beta2).add_(
            squared.sum(dim=-1), alpha=1.0 - self.beta2
        )
        columns.mul_(self.beta2).add_(
            squared.sum(dim=-2), alpha=1.0 - self.beta2
        )
        correction = 1.0 - self.beta2 ** int(step)
        row_total = rows.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(rows.dtype).tiny
        )
        variance = (
            rows[:, :, None] * columns[:, None, :] / row_total[:, :, None]
        )
        variance.div_(correction)
        inverse_root = torch.reciprocal(variance.sqrt() + self.adaptive_eps)
        adaptive = momenta * inverse_root
        momentum_norm = torch.linalg.vector_norm(
            momenta, dim=(-2, -1), keepdim=True
        )
        adaptive_norm = torch.linalg.vector_norm(
            adaptive, dim=(-2, -1), keepdim=True
        )
        tiny = torch.finfo(momenta.dtype).tiny
        return adaptive * (momentum_norm / adaptive_norm.clamp_min(tiny))

    @staticmethod
    def _ordinary_family_route(parent, momentum, participation, *, groups, width):
        if parent.shape != momentum.shape or parent.ndim != 3:
            raise RuntimeError("R05 optimizer-family tensor inventory changed")
        layers, hidden, external = parent.shape
        if hidden != groups * width or participation.shape != (layers, groups):
            raise RuntimeError("R05 optimizer-family group inventory changed")
        shape = (layers, groups, width, external)
        p = parent.view(shape)
        m = momentum.view(shape)
        dims = (-2, -1)
        tiny = torch.finfo(parent.dtype).tiny
        p_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
        sign = torch.sign(m)
        sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
        valid_norm = (p_norm > 0.0) & (sign_norm > 0.0)
        coordinate = sign * (p_norm / sign_norm.clamp_min(tiny))
        c = participation[:, :, None, None]
        source = torch.sqrt(c) * p
        source.add_(torch.sqrt((1.0 - c).clamp_min(0.0)) * coordinate)
        source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
        provisional = source * (p_norm / source_norm.clamp_min(tiny))
        provisional = torch.where(c == 1.0, p, provisional)
        finite = valid_norm & torch.isfinite(provisional).all(
            dim=dims, keepdim=True
        )
        return torch.where(finite, provisional, p).reshape_as(parent)

    @staticmethod
    def _ordinary_chord(
        u6, u5, literal_parent, momentum, congruence, *, groups, width
    ):
        if not (
            u6.shape == u5.shape == literal_parent.shape == momentum.shape
        ):
            raise RuntimeError("R02 chord tensors differ")
        if u6.ndim != 3:
            raise RuntimeError("R02 chord requires a layer batch of matrices")
        layers, hidden, external = u6.shape
        if hidden != int(groups) * int(width):
            raise RuntimeError("R02 chord group inventory changed")
        if congruence.shape != (layers,):
            raise RuntimeError("R02 chord congruence inventory changed")

        shape = (layers, int(groups), int(width), external)
        d6 = u6.reshape(shape)
        d5 = u5.reshape(shape)
        target = literal_parent.reshape(shape)
        dims = (-2, -1)
        tiny = torch.finfo(u6.dtype).tiny
        a = congruence[:, None, None, None]
        delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))

        target_norm = torch.linalg.vector_norm(target, dim=dims, keepdim=True)
        u6_norm = torch.linalg.vector_norm(d6, dim=dims, keepdim=True)
        u5_norm = torch.linalg.vector_norm(d5, dim=dims, keepdim=True)
        d6 = d6 * (target_norm / u6_norm.clamp_min(tiny))
        d5 = d5 * (target_norm / u5_norm.clamp_min(tiny))
        source = a * d6 + delta * d5
        source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
        mixed = source * (target_norm / source_norm.clamp_min(tiny))
        direction = torch.where(a == 1.0, d6, mixed)
        direction = torch.where(a == 0.0, d5, direction)
        return direction.reshape_as(u6)

    def _ordinary_r02_step(self, closure=None):
        """Execute the exact R02 update without telemetry-only products.

        This helper is deliberately separate from ``step`` so a child method
        with its own pre/post transformation (notably Method3 row products)
        can retain that transformation while using the same exact R02 cutoff.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R02 refuses a nonunit attention LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        congruence, r06_intrinsic, r05_intrinsic, router_step = (
            self.router.consume_attention_routes()
        )
        anchor_state = self.state[self.role_parameters["qkv"][0]]
        previous_step = anchor_state.get("r02_attention_step", 0)
        if type(previous_step) is not int or router_step != previous_step + 1:
            raise RuntimeError("R02 attention did not consume one shared router state")
        anchor_state["r02_attention_step"] = router_step

        for role in self._ROLES:
            parameters = self.role_parameters[role]
            gradients = torch.stack([
                parameter.grad for parameter in parameters
            ]).float()
            momenta = torch.stack([
                self._nesterov(parameter) for parameter in parameters
            ]).float()
            adaptive_equal = self._ordinary_adaptive_source(
                role, gradients, momenta, router_step
            )
            r06_source = R04AttentionCore._route_source(
                momenta, adaptive_equal, r06_intrinsic
            )
            u6 = _batched_zero_power(r06_source, self.ns_steps).float()
            scale = _match_rms_adamw_scale(
                momenta.shape[-2], momenta.shape[-1]
            )
            u6.mul_(scale)
            literal_parent = _batched_zero_power(
                momenta, self.ns_steps
            ).float()
            literal_parent.mul_(scale)
            u5 = self._ordinary_family_route(
                literal_parent,
                momenta,
                r05_intrinsic[:, None],
                groups=1,
                width=literal_parent.shape[1],
            )
            direction = self._ordinary_chord(
                u6,
                u5,
                literal_parent,
                momenta,
                congruence,
                groups=1,
                width=literal_parent.shape[1],
            )
            for index, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(direction[index].to(parameter.dtype), alpha=-lr)

        self._capture_telemetry_next_step = False
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        if bool(self._capture_telemetry_next_step):
            return super().step(closure)
        return self._ordinary_r02_step(closure)


class Method1LeanAttentionOptimizer(
    _LeanR02AttentionMixin, Method1Metric2AttentionOptimizer
):
    """Exact Method1 attention update without ordinary telemetry products."""


def lean_attention_report():
    return {
        "family_id": FAMILY_ID,
        "method1_attention_update_changed": False,
        "ns5_changed": False,
        "lr_or_wd_changed": False,
        "telemetry_transitions_use_historical_path": True,
        "ordinary_guard_reductions_elided": True,
        "ordinary_transition_gate": "bitwise_parameters_and_state",
    }


__all__ = (
    "FAMILY_ID",
    "Method1LeanAttentionOptimizer",
    "_LeanR02AttentionMixin",
    "lean_attention_report",
)
