"""Opaque R02: response-homotopy chord of the current R06 and R05 methods.

R02 is one optimizer state, not two optimizers stepped in sequence.  From one
set of clipped gradients, Nesterov buffers, matched-beta2 moments, response
probes, and current-versus-initial P5/Q4 kernel statistics it constructs:

``U6``
    the complete current R06 direction (intrinsic response participation
    routed through the response-coordinate adaptive/spectral family); and

``U5``
    the complete current R05 direction (the response-congruence spectral
    parent routed toward its coordinate-sign LMO by current intrinsic
    participation).

For a role-specific exact response-kernel congruence ``a`` and its canonical
Pythagorean residual ``delta = sqrt(1-a**2)``, R02 executes

    normalize_budget(a * U6 + delta * U5).

Both branches are first normalized to the same literal spectral-parent
budget.  RLB matrices close this budget independently for every learned
rational group; QKV and attention-output close it once per layer and role.
The explicit limits ``a == 1`` and ``a == 0`` return the normalized U6 and U5
directions bitwise.  No response statistic multiplies LR or WD.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_group_muon_core import _batched_zero_power, _match_rms_adamw_scale
from .rlb_r04_core import R04AttentionCore
from .rlb_r05_core import R05Core
from .rlb_r05_revision_core import R05RevisionRouterCore
from .rlb_r06_revision_core import R06AttentionCore, R06RLBRouterCore
from .rlb_response_capture_core import RLBResponseCaptureCore
from .rlb_r08_core import R08RevisionCore


class _R02ChordGeometry:
    """Pure equal-budget chord equations shared by all four matrix roles."""

    @staticmethod
    def budget_normalized_chord(
        u6,
        u5,
        literal_parent,
        momentum,
        congruence,
        *,
        groups,
        width,
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
        moment = momentum.reshape(shape)
        dims = (-2, -1)
        tiny = torch.finfo(u6.dtype).tiny
        machine = torch.finfo(u6.dtype).eps

        a = congruence[:, None, None, None]
        valid_a = torch.isfinite(a) & (a >= 0.0) & (a <= 1.0)
        torch._assert_async(valid_a.all())
        delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))

        target_norm = torch.linalg.vector_norm(target, dim=dims, keepdim=True)
        u6_norm = torch.linalg.vector_norm(d6, dim=dims, keepdim=True)
        u5_norm = torch.linalg.vector_norm(d5, dim=dims, keepdim=True)
        valid_norm = (
            torch.isfinite(target_norm)
            & torch.isfinite(u6_norm)
            & torch.isfinite(u5_norm)
            & (target_norm > 0.0)
            & (u6_norm > 0.0)
            & (u5_norm > 0.0)
        )
        torch._assert_async(valid_norm.all())

        # These are the two scientific branch directions.  Each is put on the
        # literal parent's exact group/role sphere before the outer homotopy.
        d6 = d6 * (target_norm / u6_norm.clamp_min(tiny))
        d5 = d5 * (target_norm / u5_norm.clamp_min(tiny))
        source = a * d6 + delta * d5
        source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
        valid_source = torch.isfinite(source_norm) & (source_norm > 0.0)
        torch._assert_async(valid_source.all())
        mixed = source * (target_norm / source_norm.clamp_min(tiny))

        # Required mathematical limits.  torch.where makes the selected
        # endpoint bitwise identical, rather than merely numerically close.
        direction = torch.where(a == 1.0, d6, mixed)
        direction = torch.where(a == 0.0, d5, direction)

        direction_norm = torch.linalg.vector_norm(
            direction, dim=dims, keepdim=True
        )
        budget_residual = (
            (direction_norm - target_norm).abs()
            / target_norm.clamp_min(1.0)
        )
        pythagorean_residual = (a.square() + delta.square() - 1.0).abs()

        u6_group_descent = (moment * d6).sum(dim=dims)
        u5_group_descent = (moment * d5).sum(dim=dims)
        chord_group_descent = (moment * direction).sum(dim=dims)
        u6_descent = u6_group_descent.sum(dim=-1)
        u5_descent = u5_group_descent.sum(dim=-1)
        chord_descent = chord_group_descent.sum(dim=-1)
        descent_valid = (
            torch.isfinite(u6_group_descent).all(dim=-1)
            & torch.isfinite(u5_group_descent).all(dim=-1)
            & torch.isfinite(chord_group_descent).all(dim=-1)
            & (u6_group_descent > 0.0).all(dim=-1)
            & (u5_group_descent > 0.0).all(dim=-1)
            & (chord_group_descent > 0.0).all(dim=-1)
            & torch.isfinite(u6_descent)
            & torch.isfinite(u5_descent)
            & torch.isfinite(chord_descent)
            & (u6_descent > 0.0)
            & (u5_descent > 0.0)
            & (chord_descent > 0.0)
        )
        torch._assert_async(descent_valid.all())
        torch._assert_async(torch.isfinite(direction).all())
        torch._assert_async((budget_residual <= 1024.0 * machine).all())
        torch._assert_async((pythagorean_residual <= 8.0 * machine).all())

        d6_norm = torch.linalg.vector_norm(d6, dim=dims, keepdim=True)
        d5_norm = torch.linalg.vector_norm(d5, dim=dims, keepdim=True)
        branch_cosine = (
            (d6 * d5).sum(dim=dims, keepdim=True)
            / (d6_norm * d5_norm).clamp_min(tiny)
        ).clamp(-1.0, 1.0)
        endpoint_cosine = (
            (d6 * direction).sum(dim=dims, keepdim=True)
            / (d6_norm * direction_norm).clamp_min(tiny)
        ).clamp(-1.0, 1.0)
        half_angle = torch.sqrt(
            (
                (1.0 - endpoint_cosine)
                / (1.0 + endpoint_cosine).clamp_min(tiny)
            ).clamp_min(0.0)
        )
        return direction.reshape_as(u6), {
            "congruence": congruence,
            "delta": delta[:, 0, 0, 0],
            "pythagorean_residual": pythagorean_residual.flatten(),
            "budget_residual": budget_residual.flatten(),
            "u6_descent": u6_descent,
            "u5_descent": u5_descent,
            "chord_descent": chord_descent,
            "u6_group_descent": u6_group_descent.flatten(),
            "u5_group_descent": u5_group_descent.flatten(),
            "chord_group_descent": chord_group_descent.flatten(),
            "branch_cosine": branch_cosine.flatten(),
            "half_angle": half_angle.flatten(),
            # Compatibility names consumed by the retained R05 transaction.
            "response_cap": delta.flatten(),
            "branch_cap": half_angle.flatten(),
            "descent_cap": half_angle.flatten(),
            "gamma": endpoint_cosine.flatten(),
            "descent_margin": (chord_descent - u6_descent).flatten(),
            "parent_descent": u6_descent.flatten(),
            "endpoint_descent": chord_descent.flatten(),
            "u6_direction": d6.reshape_as(u6),
            "u5_direction": d5.reshape_as(u5),
        }


class R02Core(_R02ChordGeometry, R06RLBRouterCore):
    """One-state R06/R05 response-homotopy router for RLB matrix pairs."""

    component_code = 4
    checkpoint_schema = "r02_r06_r05_response_chord_v1"

    # R02's endpoint is fixed to the complete U6/U5 chord and therefore never
    # consumes R05's four-corner functional-selector packet.  Install the
    # underlying B+C capture hooks directly so the inherited, scientifically
    # dead x/z/h replay is not copied during every forward pass.  The response
    # probes, feature moments, and residual-input moments used by R02 are
    # retained exactly.
    def _make_input_hook(self, index):
        return R08RevisionCore._make_input_hook(self, index)

    def _make_feature_hook(self, index):
        return RLBResponseCaptureCore._make_feature_hook(self, index)

    def _consume_functional_samples(self):
        return None, None, None

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
    ):
        if float(lr) != 3.0e-4 or float(weight_decay) != 0.10:
            raise ValueError("R02 requires the matched LR/WD contract")
        if float(momentum) != 0.95 or int(ns_steps) != 5:
            raise ValueError("R02 requires clipped beta=.95 Nesterov and NS5")
        if float(beta2) != 0.95 or float(eps) != 1.0e-8:
            raise ValueError("R02 requires matched beta2/epsilon")
        pairs = list(pairs)
        self._r02_local_group_participation = [None for _ in pairs]
        self._r02_group_participation = None
        self._r02_congruences = None
        self._r02_attention_congruence = None
        self._r02_r06_attention_intrinsic = None
        self._r02_r05_attention_intrinsic = None
        self._r02_blend_records = []
        self._r02_endpoint_records = []
        self._r02_endpoint_role = 0
        self._r02_attention_consumed = True
        self._r02_metric_factor_call = None
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
        )
        # R05 deletes the inherited incoming response metric exactly and
        # supplies the same groupwise identity at every layer and step.  Build
        # that immutable value once.  ``R08RevisionCore.step`` stacks the
        # returned tensors before any collective or in-place operation, so
        # sharing this source cannot alias mutable optimizer state.  This is a
        # trajectory-preserving removal of 18 repeated 18x256x256 identity
        # materializations per transition.
        identity = torch.eye(
            self.width,
            device=self.incoming[0].device,
            dtype=torch.float32,
        )
        self._r02_identity_lower = identity.view(
            1, 1, self.width, self.width
        ).expand(len(self.pairs), self.groups, self.width, self.width)
        scalar = torch.ones(
            (), device=identity.device, dtype=identity.dtype
        )
        self._r02_identity_volume = scalar.expand(
            len(self.pairs), self.groups
        )
        self._r02_identity_volume_residual = torch.zeros_like(
            self._r02_identity_volume
        )

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        """Skip the algebraically cancelling factorization of exact identity.

        The first metric in the inherited transaction is R05's deliberately
        deleted incoming metric, namely an exact identity for every
        layer/group.  Its stabilized Cholesky factor is a positive scalar
        times identity and its unit-volume coordinate and adjoint therefore
        cancel to the identity map.  Returning that normalized identity
        directly removes a batched 256x256 Cholesky without changing R02's
        mathematical direction.  The outgoing and residual-input metrics are
        still factored by the literal parent implementation.
        """
        call = self._r02_metric_factor_call
        if call is None:
            return RLBResponseCaptureCore._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
        self._r02_metric_factor_call = call + 1
        if call != 0:
            return RLBResponseCaptureCore._unit_volume_cholesky(
                metric, capture_spectrum=capture_spectrum
            )
        if capture_spectrum:
            raise RuntimeError("R02 identity fast path does not capture a spectrum")
        # ``_layer_response_metric`` returns one symbolic scalar per layer;
        # the inherited stack/all-reduce therefore transports only L values
        # instead of L*G*W*W redundant identity entries.
        expected = (len(self.pairs),)
        if metric.shape != expected:
            raise RuntimeError("R02 incoming identity inventory changed")
        torch._assert_async(torch.isfinite(metric).all() & (metric == 1.0).all())
        relative_shift = torch.finfo(metric.dtype).eps * self.width
        return (
            self._r02_identity_lower,
            self._r02_identity_volume,
            relative_shift,
            None,
            self._r02_identity_volume_residual,
        )

    def _left_coordinate(self, lower, volume, value):
        if lower is self._r02_identity_lower:
            return value
        return R08RevisionCore._left_coordinate(lower, volume, value)

    def _left_adjoint(self, lower, volume, value):
        if lower is self._r02_identity_lower:
            return value
        return R08RevisionCore._left_adjoint(lower, volume, value)

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "incoming_lr_scale": 1.0,
            "outgoing_lr_scale": 1.0,
            "current_r06_direction_lr_scale": 1.0,
            "current_r05_direction_lr_scale": 1.0,
            "response_congruence_chord_lr_scale": 1.0,
            "pythagorean_residual_lr_scale": 1.0,
            "exact_group_budget_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def _layer_response_metric(self, layer_index):
        """Form every R05/R06/R02 response statistic in one shared pass."""
        # Calling the capture owner directly avoids virtual dispatch through
        # R07 and R06, which would reevaluate the identical current quotient
        # solely to reduce its participation to different granularities.
        probe = RLBResponseCaptureCore._consume_probe(self, layer_index)
        z = probe.float().view(self.probe_count, self.groups, self.width)
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        u = z / rms
        pair = self.pairs[layer_index]
        live_f, live_d = self._evaluate_response(
            u, pair["numerator"], pair["denominator"]
        )
        frozen_f, frozen_d = self._evaluate_response(
            u,
            self._frozen_numerators[layer_index],
            self._frozen_denominators[layer_index],
        )

        incoming_cross_values = self._jacobian_kernel_inner(
            u, live_f, live_d, frozen_f, frozen_d
        )
        incoming_live_values = self._jacobian_kernel_inner(
            u, live_f, live_d, live_f, live_d
        )
        incoming_frozen_values = self._jacobian_kernel_inner(
            u, frozen_f, frozen_d, frozen_f, frozen_d
        )
        live_h = rms * live_f
        frozen_h = rms * frozen_f
        outgoing_cross_values = (live_h * frozen_h).sum(dim=-1).square()
        outgoing_live_values = live_h.square().sum(dim=-1).square()
        outgoing_frozen_values = frozen_h.square().sum(dim=-1).square()
        self._router_local_statistics[layer_index] = torch.stack((
            torch.stack((
                incoming_cross_values.sum(),
                incoming_live_values.sum(),
                incoming_frozen_values.sum(),
            )),
            torch.stack((
                outgoing_cross_values.sum(),
                outgoing_live_values.sum(),
                outgoing_frozen_values.sum(),
            )),
        ))
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

        incoming = self._jacobian_participation(u, live_f, live_d)
        outgoing = self._response_participation(live_f)
        sample_count = torch.tensor(
            float(incoming.numel()), device=u.device, dtype=u.dtype
        )
        self._r07_local_participation[layer_index] = torch.stack((
            incoming.sum(), sample_count,
        ))
        self._r06_local_output_participation[layer_index] = torch.stack((
            outgoing.sum(), sample_count,
        ))
        count = torch.full(
            (self.groups,),
            float(self.probe_count),
            device=u.device,
            dtype=u.dtype,
        )
        self._r02_local_group_participation[layer_index] = torch.stack((
            incoming.sum(dim=0), outgoing.sum(dim=0), count,
        ))
        # Symbolic marker for the exact identity metric.  The first
        # ``_unit_volume_cholesky`` call validates it and returns the cached
        # unit-volume identity factor.  This avoids copying and all-reducing
        # 21,233,664 known identity entries on exact M1 every transition.
        return torch.ones((), device=z.device, dtype=z.dtype)

    def _consume_router_alignments(self):
        r06_pair_alignments = super()._consume_router_alignments()
        relative = self._r04_last_alignments
        if relative is None or relative.shape != (len(self.pairs), 2):
            raise RuntimeError("R02 lost the exact response congruences")
        if any(value is None for value in self._r02_local_group_participation):
            raise RuntimeError("R02 did not form every shared group statistic")
        statistics = torch.stack(self._r02_local_group_participation)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        group_participation = statistics[:, :2] / statistics[:, 2:3].clamp_min(1.0)
        valid = (
            torch.isfinite(relative)
            & (relative >= 0.0)
            & (relative <= 1.0)
            & torch.isfinite(group_participation).all(dim=-1)
            & (group_participation >= 0.0).all(dim=-1)
            & (group_participation <= 1.0).all(dim=-1)
        )
        torch._assert_async(valid.all())

        incoming = group_participation[:, 0].mean(dim=-1)
        outgoing = group_participation[:, 1].mean(dim=-1)
        r05_attention_intrinsic = torch.sqrt(
            (incoming * outgoing).clamp_min(0.0)
        )
        if self._r06_attention_alignments is None:
            raise RuntimeError("R02 lost current R06's native attention route")
        r06_attention_intrinsic = self._r06_attention_alignments[:, 0]
        attention_valid = (
            torch.isfinite(r05_attention_intrinsic)
            & (r05_attention_intrinsic >= 0.0)
            & (r05_attention_intrinsic <= 1.0)
            & torch.isfinite(r06_attention_intrinsic)
            & (r06_attention_intrinsic >= 0.0)
            & (r06_attention_intrinsic <= 1.0)
        )
        torch._assert_async(attention_valid.all())
        attention_congruence = torch.sqrt(
            (relative[:, 0] * relative[:, 1]).clamp_min(0.0)
        )
        self._r02_group_participation = group_participation
        self._r02_congruences = relative
        self._r02_r06_attention_intrinsic = r06_attention_intrinsic
        self._r02_r05_attention_intrinsic = r05_attention_intrinsic
        self._r02_attention_congruence = attention_congruence
        self._r02_local_group_participation = [None for _ in self.pairs]
        return r06_pair_alignments

    def _blend_equalized(self, ordinary, adaptive_equal, alignment):
        role = len(self._r02_blend_records)
        if role not in (0, 1) or self._r02_congruences is None:
            raise RuntimeError("R02 shared branch construction order changed")
        # Complete current R06: same branches and product-routed alignment.
        u6_result = R05Core._blend_equalized(
            ordinary, adaptive_equal, alignment
        )
        # Current R05 starts from the response-congruence spectral parent.
        u5_parent = R05Core._blend_equalized(
            ordinary, adaptive_equal, self._r02_congruences[:, role]
        )[0]
        self._r02_blend_records.append((ordinary, u5_parent))
        return u6_result

    def _descent_safe_endpoint(self, parent, adaptive, momentum, alignment):
        del adaptive, alignment
        role = self._r02_endpoint_role
        if (
            role not in (0, 1)
            or len(self._r02_blend_records) != 2
            or self._r02_group_participation is None
            or self._r02_congruences is None
        ):
            raise RuntimeError("R02 shared endpoint construction order changed")
        literal_parent, u5_parent = self._r02_blend_records[role]
        u5, _u5_metadata = R05RevisionRouterCore._family_route(
            u5_parent,
            momentum,
            self._r02_group_participation[:, role],
            groups=self.groups,
            width=self.width,
        )
        direction, metadata = self.budget_normalized_chord(
            parent,
            u5,
            literal_parent,
            momentum,
            self._r02_congruences[:, role],
            groups=self.groups,
            width=self.width,
        )
        self._r02_endpoint_role += 1
        self._r02_endpoint_records.append(metadata)
        return direction, metadata

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        *,
        force_parent,
    ):
        del (
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_parent,
            outgoing_parent_transpose,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
            lr,
            force_parent,
        )
        layers = incoming_endpoint.shape[0]
        zeros = torch.zeros(
            layers, device=incoming_endpoint.device, dtype=incoming_endpoint.dtype
        )
        choices = torch.full(
            (layers,), 3, device=incoming_endpoint.device, dtype=torch.int64
        )
        return incoming_endpoint, outgoing_endpoint_transpose, {
            "choices": choices,
            "scores": torch.zeros(
                (layers, 4),
                device=incoming_endpoint.device,
                dtype=incoming_endpoint.dtype,
            ),
            "score_margin": zeros,
            "energies": torch.zeros(
                (layers, 4),
                device=incoming_endpoint.device,
                dtype=incoming_endpoint.dtype,
            ),
            "global_count": torch.zeros(
                (), device=incoming_endpoint.device, dtype=incoming_endpoint.dtype
            ),
        }

    def consume_attention_routes(self):
        if self._r02_attention_consumed:
            raise RuntimeError("R02 attention consumed its shared state twice")
        if (
            self._r02_attention_congruence is None
            or self._r02_r06_attention_intrinsic is None
            or self._r02_r05_attention_intrinsic is None
        ):
            raise RuntimeError("R02 attention requested an incomplete shared state")
        self._r02_attention_consumed = True
        return (
            self._r02_attention_congruence,
            self._r02_r06_attention_intrinsic,
            self._r02_r05_attention_intrinsic,
            int(self._r05_step),
        )

    def current_attention_routes(self):
        if (
            self._r02_attention_congruence is None
            or self._r02_r06_attention_intrinsic is None
            or self._r02_r05_attention_intrinsic is None
        ):
            raise RuntimeError("R02 has no current attention route")
        return (
            self._r02_attention_congruence,
            self._r02_r06_attention_intrinsic,
            self._r02_r05_attention_intrinsic,
        )

    def _publish_r02_telemetry(self):
        if (
            len(self._r02_endpoint_records) != 2
            or self._r02_congruences is None
            or self._r02_group_participation is None
        ):
            raise RuntimeError("R02 telemetry observed an incomplete chord")
        inherited = {
            key.replace("rlb_r06_", "rlb_r02_", 1): value
            for key, value in self._last_telemetry.items()
            if key.startswith("rlb_r06_")
        }
        incoming, outgoing = self._r02_endpoint_records
        records = (incoming, outgoing)
        a = self._r02_congruences
        delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
        inherited.update({
            "rlb_r02_component_code": self.component_code,
            "rlb_r02_shared_optimizer_state_count": 1,
            "rlb_r02_direction_family_count": 2,
            "rlb_r02_group_count": int(self.groups),
            "rlb_r02_group_width": int(self.width),
            "rlb_r02_structural_matrix_elements": 245_366_784,
            "rlb_r02_response_congruence_min": float(a.amin().item()),
            "rlb_r02_response_congruence_median": float(a.median().item()),
            "rlb_r02_response_congruence_max": float(a.amax().item()),
            "rlb_r02_response_delta_max": float(delta.amax().item()),
            "rlb_r02_group_participation_min": float(
                self._r02_group_participation.amin().item()
            ),
            "rlb_r02_group_participation_max": float(
                self._r02_group_participation.amax().item()
            ),
            "rlb_r02_u6_descent_min": float(min(
                item["u6_descent"].amin().item() for item in records
            )),
            "rlb_r02_u5_descent_min": float(min(
                item["u5_descent"].amin().item() for item in records
            )),
            "rlb_r02_chord_descent_min": float(min(
                item["chord_descent"].amin().item() for item in records
            )),
            "rlb_r02_u6_group_descent_min": float(min(
                item["u6_group_descent"].amin().item() for item in records
            )),
            "rlb_r02_u5_group_descent_min": float(min(
                item["u5_group_descent"].amin().item() for item in records
            )),
            "rlb_r02_chord_group_descent_min": float(min(
                item["chord_group_descent"].amin().item() for item in records
            )),
            "rlb_r02_group_budget_residual_max": float(max(
                item["budget_residual"].amax().item() for item in records
            )),
            "rlb_r02_pythagorean_residual_max": float(max(
                item["pythagorean_residual"].amax().item() for item in records
            )),
            "rlb_r02_branch_disagreement_max": float(max(
                torch.sqrt((1.0 - item["branch_cosine"].square()).clamp_min(0.0))
                .amax().item()
                for item in records
            )),
            "rlb_r02_u6_exact_limit_count": int((a == 1.0).sum().item()),
            "rlb_r02_u5_exact_limit_count": int((a == 0.0).sum().item()),
        })
        self._last_telemetry = inherited

    @torch.no_grad()
    def step(self, closure=None):
        if self._r05_step > 0 and not self._r02_attention_consumed:
            raise RuntimeError("R02 router would overwrite unconsumed attention state")
        publish = bool(self._capture_telemetry_next_step)
        self._r02_group_participation = None
        self._r02_congruences = None
        self._r02_attention_congruence = None
        self._r02_r06_attention_intrinsic = None
        self._r02_r05_attention_intrinsic = None
        self._r02_blend_records = []
        self._r02_endpoint_records = []
        self._r02_endpoint_role = 0
        self._r02_metric_factor_call = 0
        try:
            loss = super().step(closure)
        finally:
            self._r02_metric_factor_call = None
        if self._r02_endpoint_role != 2 or len(self._r02_endpoint_records) != 2:
            raise RuntimeError("R02 did not execute both RLB role chords")
        self._r02_attention_consumed = False
        if publish:
            self._publish_r02_telemetry()
        return loss

    def state_dict(self):
        if not self._r02_attention_consumed and self._r05_step > 0:
            raise RuntimeError("R02 checkpoint split the router/attention transaction")
        state = super().state_dict()
        state["r02_schema"] = self.checkpoint_schema
        return state

    def load_state_dict(self, state_dict):
        if (
            not isinstance(state_dict, dict)
            or state_dict.get("r02_schema") != self.checkpoint_schema
        ):
            raise RuntimeError("R02 checkpoint schema changed")
        translated = dict(state_dict)
        translated.pop("r02_schema")
        result = super().load_state_dict(translated)
        self._r02_local_group_participation = [None for _ in self.pairs]
        self._r02_group_participation = None
        self._r02_congruences = None
        self._r02_attention_congruence = None
        self._r02_r06_attention_intrinsic = None
        self._r02_r05_attention_intrinsic = None
        self._r02_blend_records = []
        self._r02_endpoint_records = []
        self._r02_endpoint_role = 0
        self._r02_attention_consumed = True
        return result


class R02AttentionCore(_R02ChordGeometry, R06AttentionCore):
    """One-state current-R06/current-R05 chord for both attention roles."""

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "current_r06_attention_lr_scale": 1.0,
            "current_r05_attention_lr_scale": 1.0,
            "response_congruence_chord_lr_scale": 1.0,
            "pythagorean_residual_lr_scale": 1.0,
            "exact_role_budget_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
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
        (
            congruence,
            r06_intrinsic,
            r05_intrinsic,
            router_step,
        ) = self.router.consume_attention_routes()
        anchor_state = self.state[self.role_parameters["qkv"][0]]
        previous_step = anchor_state.get("r02_attention_step", 0)
        if type(previous_step) is not int or router_step != previous_step + 1:
            raise RuntimeError("R02 attention did not consume one shared router state")
        anchor_state["r02_attention_step"] = router_step

        records = {}
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            gradients = torch.stack([
                parameter.grad for parameter in parameters
            ]).float()
            momenta = torch.stack([
                self._nesterov(parameter) for parameter in parameters
            ]).float()

            # One matched-beta2 state produces the complete current R06
            # attention direction.
            adaptive_equal, factor_condition = self._factorized_adaptive_source(
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

            # The same Nesterov tensor supplies current R05's spectral parent
            # and coordinate-sign family; no second momentum state is made.
            literal_parent = _batched_zero_power(
                momenta, self.ns_steps
            ).float()
            literal_parent.mul_(scale)
            u5, sign_metadata = R05RevisionRouterCore._family_route(
                literal_parent,
                momenta,
                r05_intrinsic[:, None],
                groups=1,
                width=literal_parent.shape[1],
            )
            direction, chord_metadata = self.budget_normalized_chord(
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
            chord_metadata["factor_condition"] = factor_condition
            chord_metadata["sign_descent"] = sign_metadata["sign_descent"]
            records[role] = chord_metadata

        if self._capture_telemetry_next_step:
            qkv = records["qkv"]
            out = records["attn_out"]
            values = (qkv, out)
            delta = torch.sqrt((1.0 - congruence.square()).clamp_min(0.0))
            self._last_telemetry = {
                "rlb_r02_attention_step": router_step,
                "rlb_r02_attention_congruence_min": float(congruence.amin().item()),
                "rlb_r02_attention_congruence_median": float(congruence.median().item()),
                "rlb_r02_attention_congruence_max": float(congruence.amax().item()),
                "rlb_r02_attention_delta_max": float(delta.amax().item()),
                "rlb_r02_attention_r06_intrinsic_min": float(
                    r06_intrinsic.amin().item()
                ),
                "rlb_r02_attention_r06_intrinsic_median": float(
                    r06_intrinsic.median().item()
                ),
                "rlb_r02_attention_r06_intrinsic_max": float(
                    r06_intrinsic.amax().item()
                ),
                "rlb_r02_attention_r05_intrinsic_min": float(
                    r05_intrinsic.amin().item()
                ),
                "rlb_r02_attention_r05_intrinsic_median": float(
                    r05_intrinsic.median().item()
                ),
                "rlb_r02_attention_r05_intrinsic_max": float(
                    r05_intrinsic.amax().item()
                ),
                "rlb_r02_attention_u6_descent_min": float(min(
                    item["u6_descent"].amin().item() for item in values
                )),
                "rlb_r02_attention_u5_descent_min": float(min(
                    item["u5_descent"].amin().item() for item in values
                )),
                "rlb_r02_attention_chord_descent_min": float(min(
                    item["chord_descent"].amin().item() for item in values
                )),
                "rlb_r02_attention_sign_descent_min": float(min(
                    item["sign_descent"].amin().item() for item in values
                )),
                "rlb_r02_attention_budget_residual_max": float(max(
                    item["budget_residual"].amax().item() for item in values
                )),
                "rlb_r02_attention_pythagorean_residual_max": float(max(
                    item["pythagorean_residual"].amax().item() for item in values
                )),
                "rlb_r02_attention_branch_disagreement_max": float(max(
                    torch.sqrt(
                        (1.0 - item["branch_cosine"].square()).clamp_min(0.0)
                    ).amax().item()
                    for item in values
                )),
                "rlb_r02_qkv_factor_condition_max": float(
                    qkv["factor_condition"].amax().item()
                ),
                "rlb_r02_attention_output_factor_condition_max": float(
                    out["factor_condition"].amax().item()
                ),
                "rlb_r02_attention_u6_exact_limit_count": int(
                    2 * (congruence == 1.0).sum().item()
                ),
                "rlb_r02_attention_u5_exact_limit_count": int(
                    2 * (congruence == 0.0).sum().item()
                ),
            }
        self._capture_telemetry_next_step = False
        return loss
