"""Elapsed-beta2 rank-64 response geometry refreshed every eight steps.

The successful Global-RLB Method-3 trajectory refreshed its expensive global
response geometry periodically while retaining current-gradient decisions on
every transition.  This owner-free realization follows that distinction:

* refresh transitions capture the fixed 32 loss rows, update the rank-64
  posterior with elapsed ``beta2 ** 8``, and cache only scalar-coordinate
  factors/routes;
* ordinary transitions capture no activation-position probes, but rebuild the
  current native rational-group polar directions and solve fresh globally
  coordinated coefficients from current gradient/momentum linear terms; and
* attention consumes the cached response route while rebuilding its current
  head-group direction every transition.

No matrix update is cached or published.  Persistent state is O(LG), never O(N),
and the largest solve remains fixed at rank 64 (posterior) or 96 (consensus).
"""

from __future__ import annotations

import math

import torch

from .rlb_basis_cotangent_trust_muon import _match_rms_adamw_adjustment
from .rlb_compact_four_role_response_homotopy_muon import (
    compact_postpolar_group_response_homotopy,
)
from .rlb_consensus_rank64_response_group_polar_muon import (
    FAMILY_ID as CONSENSUS_PARENT_FAMILY_ID,
    ConsensusRank64HeadGroupPolarAttentionOptimizer,
    ConsensusRank64ResponseGroupPolarRouter,
)
from .rlb_lagged_predictive_response_transaction_muon import (
    MatchedBeta2PredictiveRows,
    _foreach_apply,
    _foreach_nesterov,
)
from .rlb_posterior_rank64_response_group_polar_muon import (
    FAMILY_ID as POSTERIOR_PARENT_FAMILY_ID,
    PERSISTENT_ROWS,
    PosteriorRank64HeadGroupPolarAttentionOptimizer,
    PosteriorRank64ResponseGroupPolarRouter,
    posterior_rank64_response_rows,
    rank64_transaction_from_replicated_rows,
)
from .rlb_temporal_response_group_polar_muon import rational_group_zero_power


REFRESH_INTERVAL = 8
MATCHED_BETA2 = 0.95
EFFECTIVE_REFRESH_BETA2 = MATCHED_BETA2 ** REFRESH_INTERVAL
POSTERIOR_FAMILY_ID = "posterior_rank64_cadence8_group_polar_muon_v1"
CONSENSUS_FAMILY_ID = "consensus_rank64_cadence8_group_polar_muon_v1"


def _elapsed_posterior_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Apply exactly beta2**8 between observed response geometries."""

    if float(beta2) != MATCHED_BETA2:
        raise RuntimeError("cadence8 response metric requires locked beta2=.95")
    if previous_scores is None:
        return posterior_rank64_response_rows(
            current_scores,
            current_decay_action,
            None,
            None,
            beta2=beta2,
        )
    if previous_decay_action is None:
        raise RuntimeError("cadence8 response histories must coinitialize")
    previous_scale = math.sqrt(EFFECTIVE_REFRESH_BETA2 / MATCHED_BETA2)
    current_scale = math.sqrt(
        (1.0 - EFFECTIVE_REFRESH_BETA2) / (1.0 - MATCHED_BETA2)
    )
    return posterior_rank64_response_rows(
        current_scores * current_scale,
        current_decay_action * current_scale,
        previous_scores * previous_scale,
        previous_decay_action * previous_scale,
        beta2=beta2,
    )


def periodic_posterior_rank64_rows(*args, **kwargs) -> MatchedBeta2PredictiveRows:
    return _elapsed_posterior_rows(*args, **kwargs)


def periodic_consensus_rank64_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    posterior = _elapsed_posterior_rows(
        current_scores,
        current_decay_action,
        previous_scores,
        previous_decay_action,
        beta2=beta2,
    )
    scale = 1.0 / math.sqrt(2.0)
    return MatchedBeta2PredictiveRows(
        selection_scores=torch.cat((
            posterior.updated_scores * scale,
            current_scores * scale,
        )),
        selection_decay_action=torch.cat((
            posterior.updated_decay_action * scale,
            current_decay_action * scale,
        )),
        updated_scores=posterior.updated_scores,
        updated_decay_action=posterior.updated_decay_action,
        history_used=posterior.history_used,
        relative_innovation=posterior.relative_innovation,
    )


def cadence8_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
    consensus: bool,
) -> dict[str, int | float]:
    values = tuple(map(int, (
        total_positions,
        total_layers,
        total_groups,
        intermediate_width,
        model_width,
    )))
    if min(values) <= 0 or int(intermediate_width) % int(total_groups):
        raise ValueError("cadence8 scaling dimensions are invalid")
    positions, layers, groups, hidden, model = values
    coordinates = layers * groups
    posterior_factor = PERSISTENT_ROWS * (coordinates + 1)
    parent_state = 10 * coordinates + 2
    route_state = 4 * coordinates + 2 * layers
    selection_state = (96 * (coordinates + 1)) if consensus else 0
    largest = 96 if consensus else 64
    response_summary = 21 * coordinates + 10 * layers
    transaction_summary = largest * largest + 3 * largest + 8 * layers + 8
    refresh_score_summary = 32 * (coordinates + 1)
    return {
        "total_positions": positions,
        "persistent_state_elements": (
            parent_state + posterior_factor + route_state + selection_state
        ),
        "posterior_factor_elements": posterior_factor,
        "cached_response_route_elements": route_state,
        "cached_selection_factor_elements": selection_state,
        "communicated_summary_elements": (
            response_summary + refresh_score_summary
            + transaction_summary + 2 * coordinates
        ),
        "ordinary_communicated_summary_elements": (
            transaction_summary + 2 * coordinates
        ),
        "largest_temporal_dense_dimension": 96,
        "largest_dense_solve_dimension": largest,
        "dense_coordinate_metric_elements": 0,
        "owner_count": 0,
        "selected_update_elements_published": 0,
        "response_refresh_interval": REFRESH_INTERVAL,
        "matched_beta2": MATCHED_BETA2,
        "effective_refresh_beta2": EFFECTIVE_REFRESH_BETA2,
        "local_direction_arithmetic_elements": 4 * layers * hidden * model,
    }


class _Cadence8Rank64RouterMixin:
    metric_rows_fn = None
    base_family_id = ""
    family_id = ""
    base_prefix = ""
    telemetry_prefix = ""
    cache_selection_separately = False

    def __init__(self, pairs, **kwargs):
        self._cadence_transition = 0
        self._capture_response_this_transition = True
        self._cadence_predictive = None
        self._cadence_transaction = None
        self._cached_route = None
        self._cached_selection_scores = None
        self._cached_selection_decay = None
        super().__init__(pairs, **kwargs)
        self.param_groups[0][self.telemetry_prefix + "family_id"] = self.family_id
        self.param_groups[0][self.telemetry_prefix + "refresh_interval"] = (
            REFRESH_INTERVAL
        )

    def _make_input_hook(self, layer):
        parent = super()._make_input_hook(layer)

        @torch.no_grad()
        def capture(module, inputs):
            if self._capture_response_this_transition:
                return parent(module, inputs)
            return None

        return capture

    def _make_feature_hook(self, layer):
        parent = super()._make_feature_hook(layer)

        @torch.no_grad()
        def capture(module, inputs, output):
            if self._capture_response_this_transition:
                return parent(module, inputs, output)
            return None

        return capture

    def _make_cotangent_hook(self, layer):
        parent = super()._make_cotangent_hook(layer)

        def capture(module, inputs, output):
            if self._capture_response_this_transition:
                return parent(module, inputs, output)
            return None

        return capture

    def predictive_rows_fn(self, *args, **kwargs):
        if not callable(self.metric_rows_fn):
            raise RuntimeError("cadence8 metric row function is missing")
        result = self.metric_rows_fn(*args, **kwargs)
        self._cadence_predictive = result
        return result

    def transaction_fn(self, *args, **kwargs):
        result = rank64_transaction_from_replicated_rows(*args, **kwargs)
        self._cadence_transaction = result
        return result

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "elapsed_beta2_cadence8_response_metric_lr_scale": 1.0,
            "current_linear_transaction_every_step_lr_scale": 1.0,
            "cached_scalar_response_route_lr_scale": 1.0,
        })
        return result

    def _cache_refresh_state(self):
        predictive = self._cadence_predictive
        transaction = self._cadence_transaction
        if predictive is None or transaction is None:
            raise RuntimeError("cadence8 refresh omitted metric or transaction")
        route = {
            "group_participation": self._group_participation.detach().clone(),
            "group_congruence": self._group_response_congruence.detach().clone(),
            "intrinsic_participation": self._intrinsic_participation.detach().clone(),
            "attention_congruence": self._attention_response_congruence.detach().clone(),
        }
        self._cached_route = route
        anchor = self.state[self.pairs[0]["in_weight"]]
        for name, value in route.items():
            anchor["cadence8_" + name] = value
        if self.cache_selection_separately:
            scores = predictive.selection_scores.detach().clone()
            decay = predictive.selection_decay_action.detach().clone()
            anchor["cadence8_selection_scores"] = scores
            anchor["cadence8_selection_decay"] = decay
            self._cached_selection_scores = scores
            self._cached_selection_decay = decay
        else:
            self._cached_selection_scores = anchor["predictive_global_score_ema"]
            self._cached_selection_decay = anchor["predictive_global_decay_ema"]
        anchor["cadence8_transition"] = int(self._cadence_transition)

    def _restore_route(self):
        route = self._cached_route
        if route is None or set(route) != {
            "group_participation",
            "group_congruence",
            "intrinsic_participation",
            "attention_congruence",
        }:
            raise RuntimeError("cadence8 cached response route is incomplete")
        self._group_participation = route["group_participation"]
        self._group_response_congruence = route["group_congruence"]
        self._intrinsic_participation = route["intrinsic_participation"]
        self._attention_response_congruence = route["attention_congruence"]

    def _advance_cadence(self, *, refreshed: bool, publish: bool):
        self._cadence_transition += 1
        self.state[self.pairs[0]["in_weight"]]["cadence8_transition"] = (
            self._cadence_transition
        )
        self._capture_response_this_transition = (
            self._cadence_transition % REFRESH_INTERVAL == 0
        )
        if publish:
            prefix = self.telemetry_prefix
            self._last_telemetry.update({
                prefix + "response_refresh_interval": REFRESH_INTERVAL,
                prefix + "response_refreshed": int(refreshed),
                prefix + "response_age": (
                    0 if refreshed else (self._cadence_transition - 1) % REFRESH_INTERVAL
                ),
                prefix + "effective_refresh_beta2": EFFECTIVE_REFRESH_BETA2,
                prefix + "current_gradient_transaction": 1,
                prefix + "cached_parameter_update_elements": 0,
            })

    def _rename_refresh_telemetry(self):
        old = self.base_prefix
        new = self.telemetry_prefix
        self._last_telemetry = {
            key.replace(old, new, 1): (
                self.family_id if value == self.base_family_id else value
            )
            for key, value in self._last_telemetry.items()
        }
        scaling = cadence8_scaling_formula(
            total_positions=1,
            total_layers=len(self.pairs),
            total_groups=self.groups,
            intermediate_width=self.hidden,
            model_width=self.external,
            consensus=self.cache_selection_separately,
        )
        self._last_telemetry.update({
            new + "family_id": self.family_id,
            new + "state_coordinate_count": scaling[
                "persistent_state_elements"
            ],
            new + "cached_response_route_elements": scaling[
                "cached_response_route_elements"
            ],
            new + "cached_selection_factor_elements": scaling[
                "cached_selection_factor_elements"
            ],
        })

    @torch.no_grad()
    def _ordinary_step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("cadence8 router lacks realized clipping")
        if not self._attention_consumed:
            raise RuntimeError("cadence8 router would overwrite attention state")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("cadence8 router refuses nonunit LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        self._restore_route()
        participation = self._group_participation
        congruence = self._group_response_congruence

        role_parameters = {}
        role_selected = {}
        role_adjustment = {}
        role_records = {}
        exact = []
        momentum_descent = []
        for role, index, axis in (
            ("incoming", 0, "rows"),
            ("outgoing", 1, "columns"),
        ):
            key = "in_weight" if role == "incoming" else "out_weight"
            parameters = [pair[key] for pair in self.pairs]
            momenta = _foreach_nesterov(self, parameters)
            gradients = torch.stack([
                parameter.grad.detach() for parameter in parameters
            ]).float()
            parent = rational_group_zero_power(
                momenta,
                self.ns_steps,
                groups=self.groups,
                width=self.width,
            ).float()
            selected, metadata = compact_postpolar_group_response_homotopy(
                parent,
                momenta,
                gradients,
                participation[..., index],
                congruence[..., index],
                groups=self.groups,
                width=self.width,
                grouped_axis=axis,
            )
            adjustment = _match_rms_adamw_adjustment(parameters[0].shape)
            if role == "incoming":
                blocks = selected.view(
                    len(self.pairs), self.groups, self.width, self.external
                )
                gradient_blocks = gradients.view_as(blocks)
                momentum_blocks = momenta.view_as(blocks)
            else:
                blocks = selected.view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
                gradient_blocks = gradients.view_as(selected).view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
                momentum_blocks = momenta.view(
                    len(self.pairs), self.external, self.groups, self.width
                ).permute(0, 2, 3, 1)
            scaled = blocks.float() * adjustment
            exact.append((gradient_blocks * scaled).sum(dim=(-2, -1)))
            momentum_descent.append(
                (momentum_blocks.float() * scaled).sum(dim=(-2, -1))
            )
            role_parameters[role] = parameters
            role_selected[role] = selected
            role_adjustment[role] = adjustment
            role_records[role] = metadata

        incoming = role_selected["incoming"].view(
            len(self.pairs), self.groups, self.width, self.external
        ).float() * role_adjustment["incoming"]
        outgoing = role_selected["outgoing"].view(
            len(self.pairs), self.external, self.groups, self.width
        ).permute(0, 2, 3, 1).float() * role_adjustment["outgoing"]
        weights = (
            incoming.square().sum(dim=(-2, -1))
            + outgoing.square().sum(dim=(-2, -1))
        ).reshape(-1)
        exact_by_role = torch.stack([value.reshape(-1) for value in exact])
        momentum_by_role = torch.stack([
            value.reshape(-1) for value in momentum_descent
        ])
        layer_ids = torch.arange(
            len(self.pairs), device=weights.device, dtype=torch.int64
        ).repeat_interleave(self.groups)
        if self._cached_selection_scores is None or self._cached_selection_decay is None:
            raise RuntimeError("cadence8 selection factor is missing")
        selection = rank64_transaction_from_replicated_rows(
            self._cached_selection_scores,
            self._cached_selection_decay,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            total_layers=len(self.pairs),
            eta=lr,
            gather_rounds=0,
            group=self.loss_probe_group,
        )
        self._cadence_transaction = selection
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)
        incoming_selected = role_selected["incoming"]
        incoming_selected.view(
            len(self.pairs), self.groups, self.width, self.external
        ).mul_(coefficients[..., None, None].to(incoming_selected.dtype))
        outgoing_selected = role_selected["outgoing"]
        outgoing_selected.view(
            len(self.pairs), self.external, self.groups, self.width
        ).permute(0, 2, 3, 1).mul_(
            coefficients[..., None, None].to(outgoing_selected.dtype)
        )
        for role in ("incoming", "outgoing"):
            _foreach_apply(
                role_parameters[role],
                role_selected[role],
                decay=1.0 - lr * weight_decay,
                alpha=-lr * role_adjustment[role],
            )

        anchor = self.state[self.pairs[0]["in_weight"]]
        updates = int(anchor.get("predictive_response_transaction_updates", 0)) + 1
        anchor["predictive_response_transaction_updates"] = updates
        self._attention_update = updates
        self._attention_consumed = False
        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            response_cosine = torch.cat([
                role_records[role]["parent_cosine"].reshape(-1)
                for role in ("incoming", "outgoing")
            ])
            response_safe = torch.cat([
                role_records[role]["safe"].reshape(-1)
                for role in ("incoming", "outgoing")
            ])
            transaction = selection.sharded_result
            prefix = self.telemetry_prefix
            self._last_telemetry.update({
                prefix + "family_id": self.family_id,
                prefix + "transaction_accepted": int(transaction.accepted.item()),
                prefix + "rank": int(transaction.rank.item()),
                prefix + "budget_residual": float(transaction.budget_residual.item()),
                prefix + "coefficient_min": float(flat.amin().item()),
                prefix + "coefficient_median": float(flat.median().item()),
                prefix + "coefficient_max": float(flat.amax().item()),
                prefix + "cross_layer_coupling_ratio": float(
                    selection.cross_layer_coupling_ratio.item()
                ),
                prefix + "response_parent_cosine_median": float(
                    response_cosine.median().item()
                ),
                prefix + "response_safe_fraction": float(
                    response_safe.float().mean().item()
                ),
                prefix + "realized_clip_factor": float(self._clip_factor),
                prefix + "history_used": int(self._cadence_predictive.history_used),
                prefix + "relative_score_innovation": float(
                    self._cadence_predictive.relative_innovation.item()
                ),
                prefix + "realized_factor_rows": int(
                    self._cadence_predictive.updated_scores.shape[0]
                ),
            })
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        refreshed = bool(self._capture_response_this_transition)
        publish = bool(self._capture_telemetry_next_step)
        if refreshed:
            self._cadence_predictive = None
            self._cadence_transaction = None
            loss = super().step(closure)
            self._cache_refresh_state()
            if publish:
                self._rename_refresh_telemetry()
        else:
            loss = self._ordinary_step(closure)
        self._advance_cadence(refreshed=refreshed, publish=publish)
        return loss

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        anchor = self.state[self.pairs[0]["in_weight"]]
        step = anchor.get("cadence8_transition")
        if not isinstance(step, int) or step < 0:
            raise RuntimeError("cadence8 checkpoint transition changed")
        self._cadence_transition = step
        self._capture_response_this_transition = step % REFRESH_INTERVAL == 0
        names = (
            "group_participation",
            "group_congruence",
            "intrinsic_participation",
            "attention_congruence",
        )
        route = {name: anchor.get("cadence8_" + name) for name in names}
        if any(not torch.is_tensor(value) for value in route.values()):
            raise RuntimeError("cadence8 checkpoint route changed")
        self._cached_route = route
        if self.cache_selection_separately:
            self._cached_selection_scores = anchor.get("cadence8_selection_scores")
            self._cached_selection_decay = anchor.get("cadence8_selection_decay")
        else:
            self._cached_selection_scores = anchor.get("predictive_global_score_ema")
            self._cached_selection_decay = anchor.get("predictive_global_decay_ema")
        if (
            not torch.is_tensor(self._cached_selection_scores)
            or not torch.is_tensor(self._cached_selection_decay)
        ):
            raise RuntimeError("cadence8 checkpoint selection factor changed")
        return result


class PosteriorRank64Cadence8GroupPolarRouter(
    _Cadence8Rank64RouterMixin,
    PosteriorRank64ResponseGroupPolarRouter,
):
    metric_rows_fn = staticmethod(periodic_posterior_rank64_rows)
    base_family_id = POSTERIOR_PARENT_FAMILY_ID
    family_id = POSTERIOR_FAMILY_ID
    base_prefix = "posterior_rank64_response_group_polar_"
    telemetry_prefix = "posterior_rank64_cadence8_group_polar_"
    fairness_component = "posterior_rank64_cadence8_group_polar_lr_scale"
    cache_selection_separately = False


class ConsensusRank64Cadence8GroupPolarRouter(
    _Cadence8Rank64RouterMixin,
    ConsensusRank64ResponseGroupPolarRouter,
):
    metric_rows_fn = staticmethod(periodic_consensus_rank64_rows)
    base_family_id = CONSENSUS_PARENT_FAMILY_ID
    family_id = CONSENSUS_FAMILY_ID
    base_prefix = "consensus_rank64_response_group_polar_"
    telemetry_prefix = "consensus_rank64_cadence8_group_polar_"
    fairness_component = "consensus_rank64_cadence8_group_polar_lr_scale"
    cache_selection_separately = True


class PosteriorRank64Cadence8HeadPolarAttentionOptimizer(
    PosteriorRank64HeadGroupPolarAttentionOptimizer
):
    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["posterior_rank64_cadence8_attention_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == POSTERIOR_PARENT_FAMILY_ID:
                    self._last_telemetry[key] = POSTERIOR_FAMILY_ID
            self._last_telemetry.update({
                "posterior_rank64_cadence8_group_polar_attention_family_id": (
                    POSTERIOR_FAMILY_ID
                ),
                "posterior_rank64_cadence8_group_polar_attention_owner_count": 0,
                "posterior_rank64_cadence8_group_polar_attention_selected_update_elements_published": 0,
            })
        return loss


class ConsensusRank64Cadence8HeadPolarAttentionOptimizer(
    ConsensusRank64HeadGroupPolarAttentionOptimizer
):
    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result["consensus_rank64_cadence8_attention_lr_scale"] = 1.0
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value == CONSENSUS_PARENT_FAMILY_ID:
                    self._last_telemetry[key] = CONSENSUS_FAMILY_ID
            self._last_telemetry.update({
                "consensus_rank64_cadence8_group_polar_attention_family_id": (
                    CONSENSUS_FAMILY_ID
                ),
                "consensus_rank64_cadence8_group_polar_attention_owner_count": 0,
                "consensus_rank64_cadence8_group_polar_attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "CONSENSUS_FAMILY_ID",
    "EFFECTIVE_REFRESH_BETA2",
    "MATCHED_BETA2",
    "POSTERIOR_FAMILY_ID",
    "REFRESH_INTERVAL",
    "ConsensusRank64Cadence8GroupPolarRouter",
    "ConsensusRank64Cadence8HeadPolarAttentionOptimizer",
    "PosteriorRank64Cadence8GroupPolarRouter",
    "PosteriorRank64Cadence8HeadPolarAttentionOptimizer",
    "cadence8_scaling_formula",
    "periodic_consensus_rank64_rows",
    "periodic_posterior_rank64_rows",
)
