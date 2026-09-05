"""Transaction-local common-subexpression reuse for compiled Method 1.

The compiled-functional owner evaluates one nested R03/R05/R07 transaction
from a single immutable set of functional rows and model parameters.  The
literal implementation nevertheless reconstructs the same rational JVP
factors four times and stacks the same outgoing weights for every tangent
image.  This module computes each exact tensor once per outer transaction and
reuses that tensor object for the remaining nested calls.

The cache is cleared unconditionally at the transaction boundary and rejects
any change in tensor identity, layout, or in-place version.  It therefore
changes no equation, floating-point association, cadence, collective, owner
assignment, INT8 wire, LR, WD, or Newton--Schulz iteration.
"""

from __future__ import annotations

import torch

from . import rlb_method1_local_layer_owner as _owner_module
from .rlb_method1_global_owner_compiled_functional_int8 import (
    Method1GlobalOwnerCompiledFunctionalInt8Composite,
    _CompiledFunctionalGlobalOwnerRouter,
    _compiled_group_tangent_images,
    _group_tangent_image_program,
)
from .rlb_method1_global_statistics_owner_int8 import (
    Method1GlobalStatisticsOwnerInt8Composite,
    _CONSTRUCTION_LOCK,
)
from .rlb_method1_local_layer_owner_int8_direct import (
    Method1LocalLayerOwnerInt8DirectComposite,
)
from .rlb_r01_9150_local_owner_int8_ragged import (
    Method1LocalLayerOwnerInt8RaggedComposite,
    _initialize_ragged_owner_transport,
)
from .rlb_recursive_inverse_numerics import Method1RecursiveInverseRouter


FAMILY_ID = "method1_global_owner_transaction_cached_functional_int8_v1"
RAGGED_FAMILY_ID = (
    "method1_global_owner_transaction_cached_functional_ragged_int8_v1"
)
SELECTED_RAGGED_FAMILY_ID = "method1_global_statistics_owner_ragged_int8_v1"


_PADDED_TRANSPORT_TENSORS = (
    "_int8_before",
    "_int8_delta",
    "_int8_scratch",
    "_int8_send_values",
    "_int8_send_scales",
    "_int8_gathered_values",
    "_int8_gathered_scales",
    "_int8_decode_row",
)


def _tensor_signature(value: torch.Tensor):
    return (
        int(value.data_ptr()),
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
        value.device,
        int(value._version),
    )


class _TransactionCachedFunctionalMixin:
    """Reuse immutable functional tensors only inside one nested transaction."""

    def __init__(self, pairs, **kwargs):
        self._transaction_cache_active = False
        self._transaction_factor_key = None
        self._transaction_factor_parameter_key = None
        self._transaction_factors = None
        self._transaction_outgoing_key = None
        self._transaction_outgoing_weights = None
        self._transaction_outgoing_storage = None
        self._transaction_outgoing_storage_views = None
        super().__init__(pairs, **kwargs)

    def _clear_transaction_functional_cache(self):
        self._transaction_factor_key = None
        self._transaction_factor_parameter_key = None
        self._transaction_factors = None
        self._transaction_outgoing_key = None
        self._transaction_outgoing_weights = None

    def _factor_parameter_signature(self):
        return tuple(
            (
                _tensor_signature(pair["numerator"]),
                _tensor_signature(pair["denominator"]),
            )
            for pair in self.pairs
        )

    def _outgoing_parameter_signature(self):
        return tuple(_tensor_signature(value) for value in self.outgoing)

    def _functional_jvp_factors(self, preactivations):
        row_key = _tensor_signature(preactivations)
        if self._transaction_factors is None:
            parameter_key = self._factor_parameter_signature()
            factors = super()._functional_jvp_factors(preactivations)
            self._transaction_factor_key = row_key
            self._transaction_factor_parameter_key = parameter_key
            self._transaction_factors = factors
            return factors
        if row_key != self._transaction_factor_key:
            raise RuntimeError(
                "Method1 functional factors changed inside one transaction"
            )
        if (
            not self._transaction_cache_active
            and self._factor_parameter_signature()
            != self._transaction_factor_parameter_key
        ):
            raise RuntimeError(
                "Method1 functional factors changed inside one transaction"
            )
        return self._transaction_factors

    def _transaction_outgoing_weight_blocks(self):
        if self._transaction_outgoing_weights is None:
            parameter_key = self._outgoing_parameter_signature()
            layers = len(self.pairs)
            storage_shape = (layers, self.external_width, self.hidden)
            first = self.outgoing[0]
            if self._transaction_outgoing_storage is None:
                storage = torch.empty(
                    storage_shape,
                    dtype=torch.float32,
                    device=first.device,
                )
                self._transaction_outgoing_storage = storage
                self._transaction_outgoing_storage_views = tuple(
                    storage.unbind(0)
                )
            else:
                storage = self._transaction_outgoing_storage
                if (
                    storage.shape != storage_shape
                    or storage.dtype is not torch.float32
                    or storage.device != first.device
                ):
                    raise RuntimeError(
                        "Method1 outgoing FP32 transaction storage changed"
                    )
            torch._foreach_copy_(
                self._transaction_outgoing_storage_views,
                self.outgoing,
            )
            weights = storage.view(
                layers, self.external_width, self.groups, self.width
            ).permute(0, 2, 3, 1)
            self._transaction_outgoing_key = parameter_key
            self._transaction_outgoing_weights = weights
            return weights
        if (
            not self._transaction_cache_active
            and self._outgoing_parameter_signature()
            != self._transaction_outgoing_key
        ):
            raise RuntimeError(
                "Method1 outgoing weights changed inside one transaction"
            )
        return self._transaction_outgoing_weights

    def _validate_transaction_functional_cache(self):
        """Validate immutable parameter snapshots once before returning."""
        if (
            self._transaction_factors is not None
            and self._factor_parameter_signature()
            != self._transaction_factor_parameter_key
        ):
            raise RuntimeError(
                "Method1 functional factors changed inside one transaction"
            )
        if (
            self._transaction_outgoing_weights is not None
            and self._outgoing_parameter_signature()
            != self._transaction_outgoing_key
        ):
            raise RuntimeError(
                "Method1 outgoing weights changed inside one transaction"
            )

    def _group_tangent_images(
        self,
        inputs,
        preactivations,
        features,
        incoming_direction,
        outgoing_direction_transpose,
        *,
        factors,
    ):
        layers = len(self.pairs)
        samples = inputs.shape[1]
        expected_matrix = (layers, self.hidden, self.external_width)
        if (
            inputs.shape != (layers, samples, self.external_width)
            or preactivations.shape != (layers, samples, self.hidden)
            or features.shape != (layers, samples, self.hidden)
            or incoming_direction.shape != expected_matrix
            or outgoing_direction_transpose.shape != expected_matrix
        ):
            raise RuntimeError(
                "transaction-cached functional group-span inventory changed"
            )
        outgoing_weights = self._transaction_outgoing_weight_blocks()
        program = self._transaction_group_tangent_program(inputs)
        return program(
            inputs,
            features,
            incoming_direction,
            outgoing_direction_transpose,
            outgoing_weights,
            *factors,
            int(self.groups),
            int(self.width),
        )

    @staticmethod
    def _transaction_group_tangent_program(inputs):
        return (
            _compiled_group_tangent_images
            if inputs.is_cuda
            else _group_tangent_image_program
        )

    def _select_functional_corner(self, *args, **kwargs):
        if any(
            value is not None
            for value in (
                self._transaction_factor_key,
                self._transaction_factor_parameter_key,
                self._transaction_factors,
                self._transaction_outgoing_key,
                self._transaction_outgoing_weights,
            )
        ):
            raise RuntimeError("Method1 functional cache leaked across transactions")
        if self._transaction_cache_active:
            raise RuntimeError("Method1 functional transaction was already active")
        self._transaction_cache_active = True
        try:
            result = super()._select_functional_corner(*args, **kwargs)
            self._validate_transaction_functional_cache()
            return result
        finally:
            self._transaction_cache_active = False
            self._clear_transaction_functional_cache()


class _EagerTransactionCachedFunctionalMixin(
    _TransactionCachedFunctionalMixin
):
    """Use the parent's eager operation order while caching exact inputs."""

    @staticmethod
    def _transaction_group_tangent_program(inputs):
        del inputs
        return _group_tangent_image_program


class _TransactionCachedFunctionalGlobalOwnerRouter(
    _TransactionCachedFunctionalMixin,
    _CompiledFunctionalGlobalOwnerRouter,
):
    checkpoint_schema = FAMILY_ID + "_router"


class Method1GlobalOwnerTransactionCachedFunctionalInt8Composite(
    Method1GlobalOwnerCompiledFunctionalInt8Composite
):
    """Compiled-functional Method 1 with exact transaction-local reuse."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        router_kwargs = {
            key: kwargs[key]
            for key in ("lr", "weight_decay", "momentum", "ns_steps", "beta2", "eps")
        }
        with _CONSTRUCTION_LOCK:
            original = _owner_module.Method1RecursiveInverseRouter
            if original is not Method1RecursiveInverseRouter:
                raise RuntimeError("Method1 owner router constructor was already patched")
            _owner_module.Method1RecursiveInverseRouter = (
                _TransactionCachedFunctionalGlobalOwnerRouter
            )
            try:
                Method1LocalLayerOwnerInt8DirectComposite.__init__(
                    self, blocks, adamw, **kwargs
                )
            finally:
                _owner_module.Method1RecursiveInverseRouter = original
        if not isinstance(
            self.router, _TransactionCachedFunctionalGlobalOwnerRouter
        ):
            raise RuntimeError(
                "transaction-cached functional Method1 owner was not installed"
            )

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

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": FAMILY_ID,
            "functional_factor_recomputations_per_transaction": "four_to_one",
            "outgoing_weight_stacks_per_transaction": "up_to_eight_to_one",
            "cache_scope": "one_select_functional_corner_transaction",
            "cache_fail_closed_on_tensor_change": True,
            "parameter_signature_validation": "once_at_transaction_boundary",
            "outgoing_stack_fp32_path": "one_pass_foreach_copy_to_reused_storage",
            "outgoing_bf16_stack_intermediate_removed": True,
            "functional_equations_changed": False,
            "floating_point_association_changed_vs_compiled_parent": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": False,
        })
        return result


class Method1GlobalStatisticsOwnerRaggedInt8Composite(
    Method1GlobalStatisticsOwnerInt8Composite
):
    """Selected Method 1 global owner with exact ragged publication."""

    _SCHEMA = SELECTED_RAGGED_FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        for name in _PADDED_TRANSPORT_TENSORS:
            delattr(self, name)
        _initialize_ragged_owner_transport(self)

    @torch.no_grad()
    def step(self):
        self._prepare_functional_rows()
        self._prepare_response_rows()
        self._prepare_metric_rows()
        try:
            return Method1LocalLayerOwnerInt8RaggedComposite.step(self)
        finally:
            self.router.probe_count = self._owner_original_probe_count
            self.router.input_capture_count = (
                self._owner_original_input_capture_count
            )
            self._sync_capture_plan()

    def telemetry(self):
        result = dict(super().telemetry())
        result.update({
            "rlb_layer_owner_int8_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_int8_value_bytes": self._ragged_wire_value_bytes,
            "rlb_layer_owner_int8_scale_bytes": self._ragged_wire_scale_bytes,
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": SELECTED_RAGGED_FAMILY_ID,
            "scientific_parent": "selected_method1_global_statistics_owner",
            "ragged_owner_transport_only": True,
            "ragged_owner_counts": self._ragged_owner_counts,
            "padded_owner_slots_removed": True,
            "method1_equations_changed": False,
            "collective_payload_bits_changed_vs_int8_parent": False,
            "floating_point_update_changed_vs_int8_parent": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": False,
        })
        return result


class Method1GlobalOwnerTransactionCachedFunctionalRaggedInt8Composite(
    Method1GlobalOwnerTransactionCachedFunctionalInt8Composite
):
    """Compose cached Method 1 with exact unequal-row publication."""

    _SCHEMA = RAGGED_FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        for name in _PADDED_TRANSPORT_TENSORS:
            delattr(self, name)
        _initialize_ragged_owner_transport(self)

    @torch.no_grad()
    def step(self):
        self._prepare_functional_rows()
        self._prepare_response_rows()
        self._prepare_metric_rows()
        try:
            return Method1LocalLayerOwnerInt8RaggedComposite.step(self)
        finally:
            self.router.probe_count = self._owner_original_probe_count
            self.router.input_capture_count = (
                self._owner_original_input_capture_count
            )
            self._sync_capture_plan()

    def telemetry(self):
        result = dict(super().telemetry())
        result.update({
            "rlb_layer_owner_int8_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_int8_value_bytes": self._ragged_wire_value_bytes,
            "rlb_layer_owner_int8_scale_bytes": self._ragged_wire_scale_bytes,
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "family_id": RAGGED_FAMILY_ID,
            "scientific_parent_family_id": FAMILY_ID,
            "ragged_owner_transport_only": True,
            "ragged_owner_counts": self._ragged_owner_counts,
            "padded_owner_slots_removed": True,
            "transaction_cache_changed": False,
            "functional_equations_changed": False,
            "collective_payload_bits_changed_vs_int8_parent": False,
            "floating_point_update_changed_vs_int8_parent": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "fresh_quality_required_if_faster": False,
        })
        return result


__all__ = (
    "FAMILY_ID",
    "RAGGED_FAMILY_ID",
    "SELECTED_RAGGED_FAMILY_ID",
    "Method1GlobalStatisticsOwnerRaggedInt8Composite",
    "Method1GlobalOwnerTransactionCachedFunctionalInt8Composite",
    "Method1GlobalOwnerTransactionCachedFunctionalRaggedInt8Composite",
    "_TransactionCachedFunctionalGlobalOwnerRouter",
    "_TransactionCachedFunctionalMixin",
    "_EagerTransactionCachedFunctionalMixin",
    "_tensor_signature",
)
