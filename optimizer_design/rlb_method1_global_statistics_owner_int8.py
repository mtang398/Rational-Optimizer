"""Layer-owned Method1 with all-rank RLB observations.

The existing local-owner approximation reduces both computation and quality
by letting each layer owner see only its own data-parallel rank.  This branch
keeps the same owner-local matrix transaction and block-256 INT8 publication,
but restores the complete four-rank observation set before the transaction:

* aligned ``(x, z, h)`` rows and clipping-corrected loss cotangents are sent
  to the stable owner of each layer;
* response probes and residual-input samples are sent to those owners;
* additive outgoing-feature moments are reduce-scattered to the owners.

Consequently every layer-local RLB statistic is evaluated from the same data
rows as replicated Method1.  On the three outer-reuse transitions, owners
also gather only their small score columns and scalar coordinate summaries to
restore the literal 324-coordinate, 18-layer R01 Fisher/allocation solve.  The
remaining approximation is the higher-order R03/frame decision on the one
outer-refresh transition, which stays partitioned into four 4/5-layer owner
models.  Newton--Schulz, its five iterations, metric and response cadence,
momentum, LR, WD, and the INT8 delta wire are unchanged.  This is a numerical
method and requires a fresh full quality trajectory.
"""

from __future__ import annotations

import threading

import torch
import torch.distributed as dist

from ._archive_r07_frame_878462.rlb_response_capture_core import (
    RLBResponseCaptureCore as _ArchivedResponseCaptureCore,
)
from .rlb_layer_owner_collectives import (
    exchange_functional_row_families,
    owner_layer_lists,
    reduce_scatter_layer_metric_families,
)
from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_statistics_local_owner_int8_v1"
_CONSTRUCTION_LOCK = threading.RLock()
_REAL_ALL_REDUCE = dist.all_reduce


def _gather_owner_coordinate_columns(
    local: torch.Tensor, *, coordinates_per_layer: int
) -> torch.Tensor:
    """Gather padded owner columns while the inner world is patched to one."""

    if local.ndim < 1:
        raise RuntimeError("owner coordinate packet has no coordinate axis")
    rank = int(dist.get_rank())
    inventories = owner_layer_lists()
    owned = inventories[rank]
    coordinates = int(coordinates_per_layer)
    expected = len(owned) * coordinates
    if coordinates < 1 or int(local.shape[-1]) != expected:
        raise RuntimeError("owner coordinate packet width changed")
    maximum = max(len(item) for item in inventories) * coordinates
    send = torch.zeros(
        *local.shape[:-1], maximum, device=local.device, dtype=local.dtype
    )
    send[..., :expected].copy_(local)
    gathered_flat = torch.empty(
        4 * send.numel(), device=send.device, dtype=send.dtype
    )
    dist.all_gather_into_tensor(gathered_flat, send.reshape(-1))
    gathered = gathered_flat.view(4, *send.shape)
    canonical = torch.empty(
        *local.shape[:-1], 18 * coordinates,
        device=local.device,
        dtype=local.dtype,
    )
    for owner, inventory in enumerate(inventories):
        indices = torch.tensor(
            [
                layer * coordinates + coordinate
                for layer in inventory
                for coordinate in range(coordinates)
            ],
            device=local.device,
            dtype=torch.int64,
        )
        canonical.index_copy_(
            -1, indices, gathered[owner, ..., : indices.numel()]
        )
    return canonical


class _GlobalStatisticsOwnerMixin:
    """Consume owner-installed all-rank functional rows.

    Keeping the owner logic as a cooperative mixin lets execution-only
    descendants replace the qualified outer core without silently falling
    back to ``Method1RecursiveInverseRouter``'s archived implementation.
    """

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self._owner_global_functional = None
        self._owner_global_cotangents = None
        self._owner_global_cotangent_scale_range = None
        self._owner_global_r01_used = False
        self._owner_global_r01_dimension = 0
        self._owner_global_r01_sample_count = 0

    # The full capture broker below owns all activation observation.  Keeping
    # a second set of owner hooks would compute and then discard duplicate
    # feature Grams on the expensive metric transitions.
    def _make_probe_hook(self, _index):
        def capture(_module, _inputs):
            return None

        return capture

    def _make_input_hook(self, _index):
        def capture(_module, _inputs):
            return None

        return capture

    def _make_feature_hook(self, _index):
        def capture(_module, _inputs, _output):
            return None

        return capture

    def _make_loss_cotangent_hook(self, _index):
        def capture(_module, _inputs, _output):
            return None

        return capture

    def install_global_functional_rows(
        self,
        functional,
        cotangents,
        *,
        cotangent_scale_range,
    ):
        if self._owner_global_functional is not None:
            raise RuntimeError("global functional rows were installed twice")
        if len(functional) != 3:
            raise RuntimeError("global functional family inventory changed")
        layers = len(self.pairs)
        samples = int(functional[0].shape[1])
        if (
            any(int(value.shape[0]) != layers for value in functional)
            or cotangents.shape[:2] != (layers, samples)
            or any(int(value.shape[1]) != samples for value in functional)
        ):
            raise RuntimeError("global functional row shapes changed")
        self._owner_global_functional = tuple(functional)
        self._owner_global_cotangents = cotangents
        self._owner_global_cotangent_scale_range = tuple(
            float(value) for value in cotangent_scale_range
        )

    def _consume_functional_samples(self):
        functional = self._owner_global_functional
        cotangents = self._owner_global_cotangents
        scale_range = self._owner_global_cotangent_scale_range
        self._owner_global_functional = None
        self._owner_global_cotangents = None
        self._owner_global_cotangent_scale_range = None
        if functional is None or cotangents is None or scale_range is None:
            raise RuntimeError("owner did not receive all-rank functional rows")
        if self._r09_clip_factor is None:
            raise RuntimeError("owner did not receive its clipping certificate")
        self._r09_loss_cotangents = cotangents
        self._r09_cotangent_scale_range = scale_range
        self._last_probe_record_count = self.expected_microbatches
        return functional

    def _select_global_r01_owner_transaction(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_endpoint,
        outgoing_endpoint_transpose,
        lr,
        force_parent,
    ):
        """Solve the literal R01 quadratic over all 18 owner directions."""

        cotangents = self._r09_loss_cotangents
        if cotangents is None:
            raise RuntimeError("global R01 owner lost its cotangents")
        local_layers = len(self.pairs)
        samples = int(functional_inputs.shape[1])
        if force_parent.shape != (local_layers,):
            raise RuntimeError("global R01 owner parent-limit shape changed")
        factors = self._functional_jvp_factors(functional_preactivations)
        images = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_endpoint,
            outgoing_endpoint_transpose,
            factors=factors,
        )
        weight_decay = float(self.param_groups[0]["weight_decay"])
        group_decay = self._group_tangent_images(
            functional_inputs,
            functional_preactivations,
            functional_features,
            torch.stack(self.incoming).float() * weight_decay,
            torch.stack(self.outgoing).float().transpose(-2, -1) * weight_decay,
            factors=factors,
        )
        local_scores = torch.einsum(
            "lngd,lnd->nlg", images, cotangents
        ).reshape(samples, local_layers * self.groups)
        scores = _gather_owner_coordinate_columns(
            local_scores, coordinates_per_layer=self.groups
        )
        local_decay_scores = torch.einsum(
            "lngd,lnd->nlg", group_decay, cotangents
        ).sum(dim=(1, 2))
        _REAL_ALL_REDUCE(local_decay_scores, op=dist.ReduceOp.SUM)
        count = torch.tensor(
            float(samples), device=scores.device, dtype=scores.dtype
        )
        fisher = scores.transpose(0, 1) @ scores
        fisher.div_(count)
        fisher = 0.5 * (fisher + fisher.transpose(-2, -1))
        decay_cross = (
            scores.transpose(0, 1) @ local_decay_scores
        ).div_(count)
        fisher = fisher.unsqueeze(0)
        decay_cross = decay_cross.unsqueeze(0)

        shape = (
            local_layers, self.groups, self.width, self.external_width
        )
        incoming_blocks = incoming_endpoint.view(shape)
        outgoing_blocks = outgoing_endpoint_transpose.view(shape)
        incoming_gradients = torch.stack([
            parameter.grad for parameter in self.incoming
        ]).float().view(shape)
        outgoing_gradients = torch.stack([
            parameter.grad for parameter in self.outgoing
        ]).float().transpose(-2, -1).view(shape)
        incoming_momentum = self._current_nesterov_stack(
            self.incoming
        ).view(shape)
        outgoing_momentum = self._current_nesterov_stack(
            self.outgoing, transpose=True
        ).view(shape)
        incoming_exact = (incoming_gradients * incoming_blocks).sum(
            dim=(-2, -1)
        )
        outgoing_exact = (outgoing_gradients * outgoing_blocks).sum(
            dim=(-2, -1)
        )
        incoming_momentum_linear = (
            incoming_momentum * incoming_blocks
        ).sum(dim=(-2, -1))
        outgoing_momentum_linear = (
            outgoing_momentum * outgoing_blocks
        ).sum(dim=(-2, -1))
        budget_weights = (
            incoming_blocks.square().sum(dim=(-2, -1))
            + outgoing_blocks.square().sum(dim=(-2, -1))
        )
        local_exact = (incoming_exact + outgoing_exact).reshape(1, -1)
        local_momentum = (
            incoming_momentum_linear + outgoing_momentum_linear
        ).reshape(1, -1)
        local_budget = budget_weights.reshape(1, -1)
        exact_linear = _gather_owner_coordinate_columns(
            local_exact, coordinates_per_layer=self.groups
        )
        momentum_linear = _gather_owner_coordinate_columns(
            local_momentum, coordinates_per_layer=self.groups
        )
        flat_budget = _gather_owner_coordinate_columns(
            local_budget, coordinates_per_layer=self.groups
        )
        flat_coefficients, metadata = self._select_group_span_coefficients(
            fisher,
            decay_cross,
            exact_linear,
            momentum_linear,
            flat_budget,
            lr,
        )
        full_coefficients = flat_coefficients.view(18, self.groups)
        indices = torch.tensor(
            owner_layer_lists()[int(dist.get_rank())],
            device=full_coefficients.device,
            dtype=torch.int64,
        )
        candidate_coefficients = full_coefficients.index_select(0, indices)

        candidate_incoming_exact = (
            incoming_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_exact = (
            outgoing_exact * candidate_coefficients
        ).sum(dim=-1)
        candidate_incoming_momentum = (
            incoming_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        candidate_outgoing_momentum = (
            outgoing_momentum_linear * candidate_coefficients
        ).sum(dim=-1)
        local_valid = torch.stack((
            (candidate_incoming_exact > 0.0).all(),
            (candidate_outgoing_exact > 0.0).all(),
            (candidate_incoming_momentum > 0.0).all(),
            (candidate_outgoing_momentum > 0.0).all(),
            (~force_parent).all(),
        )).to(torch.int32)
        _REAL_ALL_REDUCE(local_valid, op=dist.ReduceOp.MIN)
        accepted = metadata["accepted"][0] & local_valid.bool().all()
        coefficients = torch.where(
            accepted,
            candidate_coefficients,
            torch.ones_like(candidate_coefficients),
        )
        incoming_selected = (
            incoming_blocks * coefficients[:, :, None, None]
        ).reshape_as(incoming_endpoint)
        outgoing_selected = (
            outgoing_blocks * coefficients[:, :, None, None]
        ).reshape_as(outgoing_endpoint_transpose)

        parent_exact = exact_linear.sum(dim=-1)
        parent_momentum = momentum_linear.sum(dim=-1)
        metadata["accepted"] = accepted.reshape(1)
        metadata["selected_exact_descent"] = torch.where(
            accepted, metadata["candidate_exact_descent"], parent_exact
        )
        metadata["selected_momentum_descent"] = torch.where(
            accepted, metadata["candidate_momentum_descent"], parent_momentum
        )
        metadata["selected_score"] = torch.where(
            accepted, metadata["candidate_score"], metadata["parent_score"]
        )
        metadata["improvement"] = (
            metadata["parent_score"] - metadata["selected_score"]
        )
        metadata["global_count"] = count
        metadata["clip_factor"] = torch.tensor(
            float(self._r09_clip_factor), device=count.device, dtype=count.dtype
        )
        metadata["cross_layer_coupling_ratio"] = (
            self._cross_layer_coupling_ratio(fisher, 18, self.groups)
        )
        metadata["selected_coefficient_min"] = full_coefficients.amin()
        metadata["selected_coefficient_median"] = full_coefficients.median()
        metadata["selected_coefficient_max"] = full_coefficients.amax()
        local_role = torch.stack((
            candidate_incoming_exact.amin(),
            candidate_outgoing_exact.amin(),
            candidate_incoming_momentum.amin(),
            candidate_outgoing_momentum.amin(),
        ))
        _REAL_ALL_REDUCE(local_role, op=dist.ReduceOp.MIN)
        parent_local_role = torch.stack((
            incoming_exact.sum(dim=-1).amin(),
            outgoing_exact.sum(dim=-1).amin(),
            incoming_momentum_linear.sum(dim=-1).amin(),
            outgoing_momentum_linear.sum(dim=-1).amin(),
        ))
        _REAL_ALL_REDUCE(parent_local_role, op=dist.ReduceOp.MIN)
        selected_role = torch.where(accepted, local_role, parent_local_role)
        (
            metadata["selected_incoming_exact_descent_min"],
            metadata["selected_outgoing_exact_descent_min"],
            metadata["selected_incoming_momentum_descent_min"],
            metadata["selected_outgoing_momentum_descent_min"],
        ) = selected_role.unbind(0)
        self._r01_global_metadata = metadata
        repeated = lambda value: value.reshape(1).expand(local_layers)
        self._r09_span_metadata = {
            "accepted": repeated(accepted),
            "rank": repeated(metadata["rank"][0]),
            "eigenvalue_max": repeated(metadata["eigenvalue_max"][0]),
            "coefficient_min": repeated(metadata["coefficient_min"][0]),
            "coefficient_median": repeated(metadata["coefficient_median"][0]),
            "coefficient_max": repeated(metadata["coefficient_max"][0]),
            "selected_exact_descent": repeated(
                metadata["selected_exact_descent"][0]
            ),
            "selected_momentum_descent": repeated(
                metadata["selected_momentum_descent"][0]
            ),
            "budget_residual": repeated(metadata["budget_residual"][0]),
            "improvement": repeated(metadata["improvement"][0]),
            "global_count": count,
            "clip_factor": metadata["clip_factor"],
        }
        choices = torch.where(
            accepted,
            torch.full(
                (local_layers,), 3, device=count.device, dtype=torch.int64
            ),
            torch.zeros(
                (local_layers,), device=count.device, dtype=torch.int64
            ),
        )
        parent_score = metadata["parent_score"][0]
        candidate_score = metadata["candidate_score"][0]
        scores4 = torch.stack((
            parent_score, candidate_score, parent_score, candidate_score
        )).reshape(1, 4).expand(local_layers, 4)
        self._owner_global_r01_used = True
        self._owner_global_r01_dimension = int(18 * self.groups)
        self._owner_global_r01_sample_count = samples
        return incoming_selected, outgoing_selected, {
            "choices": choices,
            "scores": scores4,
            "score_margin": (
                parent_score - candidate_score
            ).abs().expand(local_layers),
            "energies": torch.zeros_like(scores4),
            "global_count": count,
        }

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
        if not bool(self._method1_outer_active):
            del (
                incoming_parent,
                outgoing_parent_transpose,
                incoming_parent_descent,
                incoming_endpoint_descent,
                outgoing_parent_descent,
                outgoing_endpoint_descent,
            )
            return self._select_global_r01_owner_transaction(
                functional_inputs,
                functional_preactivations,
                functional_features,
                incoming_endpoint,
                outgoing_endpoint_transpose,
                lr,
                force_parent,
            )
        return super()._select_functional_corner(
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
            force_parent=force_parent,
        )


class _GlobalStatisticsOwnerRouter(
    _GlobalStatisticsOwnerMixin,
    Method1RecursiveInverseRouter,
):
    """Qualified outer core plus owner-installed all-rank observations."""

    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalStatisticsOwnerInt8Composite(
    Method1LocalLayerOwnerInt8DirectComposite
):
    """Run layer-local Method1 from complete four-rank observations."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        router_kwargs = {
            key: kwargs[key]
            for key in ("lr", "weight_decay", "momentum", "ns_steps", "beta2", "eps")
        }
        # The parent constructor has a deliberately fixed router symbol.  Swap
        # only that constructor dependency under a private lock, and restore
        # it before exposing the composite to training.
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = _GlobalStatisticsOwnerRouter
            try:
                super().__init__(blocks, adamw, **kwargs)
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(self.router, _GlobalStatisticsOwnerRouter):
            raise RuntimeError("global-statistics owner router was not installed")

        # This optimizer never steps parameters.  It is a capture broker for
        # all 18 layers, with the exact same cadence-aware hook hierarchy as
        # Method1.  The owner router retains all actual optimizer state.
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

    def _sync_capture_plan(self):
        self.capture_broker._capture_full_metric_this_step = bool(
            self.router._capture_full_metric_this_step
        )
        self.capture_broker._capture_full_response_this_step = bool(
            self.router._capture_full_response_this_step
        )

    def record_realized_clipping(self, preclip_norm, max_norm):
        self.capture_broker.record_realized_clipping(preclip_norm, max_norm)
        return self.router.record_realized_clipping(preclip_norm, max_norm)

    def _assert_owned(self, owned):
        expected = torch.tensor(
            self.owned_layers, device=owned.device, dtype=owned.dtype
        )
        torch._assert_async((owned == expected).all())

    @staticmethod
    def _owner_rows(value):
        """Convert ``[origin, owner_layer, row, ...]`` to owner row order."""
        if value.ndim < 4 or int(value.shape[0]) != 4:
            raise RuntimeError("all-rank owner row packet changed")
        return value.permute(1, 0, 2, *range(3, value.ndim)).reshape(
            int(value.shape[1]),
            int(value.shape[0]) * int(value.shape[2]),
            *value.shape[3:],
        )

    def _prepare_functional_rows(self):
        local = self.capture_broker._consume_functional_samples()
        local_cotangents = self.capture_broker._r09_loss_cotangents
        scale_range = self.capture_broker._r09_cotangent_scale_range
        if local_cotangents is None or scale_range is None:
            raise RuntimeError("capture broker lost clipping-aware cotangents")
        exchanged, owned = exchange_functional_row_families(
            (*local, local_cotangents)
        )
        self._assert_owned(owned)
        owner_values = tuple(self._owner_rows(value) for value in exchanged)
        self.router.install_global_functional_rows(
            owner_values[:3],
            owner_values[3],
            cotangent_scale_range=scale_range,
        )
        self._last_global_functional_rows = int(owner_values[0].shape[1])

        # The broker does not execute its selector, which is the normal owner
        # of these transient resets.
        self.capture_broker._r09_loss_cotangents = None
        self.capture_broker._r09_clip_factor = None
        self.capture_broker._functional_pending_inputs = [
            None for _ in self.capture_broker.pairs
        ]

        # The local router's duplicate hook records are intentionally replaced
        # by the all-rank cache above.
        self.router._functional_pending_inputs = [None for _ in self.router.pairs]
        self.router._functional_records = [[] for _ in self.router.pairs]
        self.router._functional_packets = [None for _ in self.router.pairs]
        self.router._r09_cotangent_records = [[] for _ in self.router.pairs]

    def _prepare_response_rows(self):
        local = torch.stack([
            _ArchivedResponseCaptureCore._consume_probe(
                self.capture_broker, layer
            )
            for layer in range(len(self.capture_broker.pairs))
        ])
        if not bool(self.router._capture_full_response_this_step):
            for local_layer, global_layer in enumerate(self.owned_layers):
                self.router._probe_records[local_layer] = list(
                    torch.tensor_split(
                        local[global_layer],
                        self.router.expected_microbatches,
                        dim=0,
                    )
                )
            return
        exchanged, owned = exchange_functional_row_families((local,))
        self._assert_owned(owned)
        global_rows = exchanged[0]
        rows_per_origin = int(global_rows.shape[2])
        self.router.probe_count = 4 * rows_per_origin
        for local_layer in range(len(self.owned_layers)):
            self.router._probe_records[local_layer] = [
                global_rows[origin, local_layer]
                for origin in range(4)
            ]
        self._last_global_response_rows = int(self.router.probe_count)

    def _prepare_metric_rows(self):
        broker = self.capture_broker
        if not bool(self.router._capture_full_metric_this_step):
            if any(broker._input_records):
                raise RuntimeError("stale broker unexpectedly retained input rows")
            if any(value is not None for value in broker._feature_moment_sums):
                raise RuntimeError("stale broker unexpectedly formed feature moments")
            if any(
                count != broker.expected_microbatches
                for count in broker._stale_input_record_counts
            ) or any(
                count != broker.expected_microbatches
                for count in broker._stale_feature_record_counts
            ):
                raise RuntimeError("stale broker capture count changed")
            broker._stale_input_record_counts = [0 for _ in broker.pairs]
            broker._stale_feature_record_counts = [0 for _ in broker.pairs]
            self.router._stale_input_record_counts = [
                self.router.expected_microbatches for _ in self.router.pairs
            ]
            self.router._stale_feature_record_counts = [
                self.router.expected_microbatches for _ in self.router.pairs
            ]
            return

        local_input_rows = []
        for layer, records in enumerate(broker._input_records):
            if len(records) != broker.expected_microbatches:
                raise RuntimeError(
                    f"global input layer {layer} lost a microbatch"
                )
            local_input_rows.append(torch.cat(records, dim=0).float())
            broker._input_records[layer] = []
        local_inputs = torch.stack(local_input_rows)
        exchanged_inputs, owned = exchange_functional_row_families(
            (local_inputs,)
        )
        self._assert_owned(owned)
        owner_inputs = exchanged_inputs[0]
        rows_per_origin = int(owner_inputs.shape[2])
        self.router.input_capture_count = rows_per_origin
        for local_layer in range(len(self.owned_layers)):
            self.router._input_records[local_layer] = [
                owner_inputs[origin, local_layer]
                for origin in range(4)
            ]
        self._last_global_input_rows = 4 * rows_per_origin

        if any(value is None for value in broker._feature_moment_sums):
            raise RuntimeError("global feature moment inventory is incomplete")
        if any(
            count != broker.expected_microbatches
            for count in broker._feature_record_counts
        ):
            raise RuntimeError("global feature record count changed")
        moments = torch.stack(broker._feature_moment_sums)
        counts = torch.tensor(
            broker._feature_sample_counts,
            device=moments.device,
            dtype=moments.dtype,
        ).unsqueeze(-1)
        (owned_moments, owned_counts), metric_owned = (
            reduce_scatter_layer_metric_families((moments, counts))
        )
        self._assert_owned(metric_owned)
        global_feature_count = (
            int(broker.feature_capture_count)
            * int(broker.expected_microbatches)
            * 4
        )
        torch._assert_async(
            (owned_counts[:, 0] == float(global_feature_count)).all()
        )
        for local_layer in range(len(self.owned_layers)):
            self.router._feature_moment_sums[local_layer] = owned_moments[
                local_layer
            ]
            self.router._feature_sample_counts[local_layer] = global_feature_count
            # The value is an additive global statistic; the consumer's four
            # record gate remains a cadence certificate, not a rank count.
            self.router._feature_record_counts[local_layer] = (
                self.router.expected_microbatches
            )
        self._last_global_feature_samples = global_feature_count
        broker._feature_moment_sums = [None for _ in broker.pairs]
        broker._feature_sample_counts = [0 for _ in broker.pairs]
        broker._feature_record_counts = [0 for _ in broker.pairs]

    @torch.no_grad()
    def step(self):
        self._prepare_functional_rows()
        self._prepare_response_rows()
        self._prepare_metric_rows()
        try:
            return super().step()
        finally:
            self.router.probe_count = self._owner_original_probe_count
            self.router.input_capture_count = (
                self._owner_original_input_capture_count
            )
            self._sync_capture_plan()

    def telemetry(self):
        result = super().telemetry()
        result.update({
            "rlb_owner_global_functional_rows": self._last_global_functional_rows,
            "rlb_owner_global_response_rows": self._last_global_response_rows,
            "rlb_owner_global_input_rows": self._last_global_input_rows,
            "rlb_owner_global_feature_samples": self._last_global_feature_samples,
            "rlb_owner_global_r01_used": int(
                self.router._owner_global_r01_used
            ),
            "rlb_owner_global_r01_dimension": (
                self.router._owner_global_r01_dimension
            ),
            "rlb_owner_global_r01_sample_count": (
                self.router._owner_global_r01_sample_count
            ),
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "inner_statistics": "all_four_data_ranks_per_owned_layer",
            "global_functional_rows_restored": True,
            "global_clipping_aware_cotangents_restored": True,
            "global_response_statistics_restored": True,
            "global_input_metric_restored": True,
            "global_feature_metric_restored": True,
            "duplicate_owner_activation_capture_removed": True,
            "global_cross_layer_model": (
                "exact_18_layer_r01_on_three_outer_reuse_transitions; "
                "four_owner_local_higher_order_model_on_outer_refresh"
            ),
            "global_r01_coordinate_solve_restored": True,
            "global_r01_coordinate_dimension": int(18 * self.router.groups),
            "global_r01_outer_reuse_fraction": "3/4",
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "lr_or_wd_multiplier_changed": False,
            "scientific_equations_changed_vs_int8_parent": True,
            "floating_point_update_changed_vs_int8_parent": True,
            "fresh_quality_required": True,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "Method1GlobalStatisticsOwnerInt8Composite",
    "_GlobalStatisticsOwnerMixin",
    "_GlobalStatisticsOwnerRouter",
)
