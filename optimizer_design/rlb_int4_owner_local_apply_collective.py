"""Exact execution alternatives for packed-INT4 owner publication.

The current packed-INT4 executor restores every owner-local parameter, copies
its packet through the local receive slot, decodes that packet, and then adds
the decoded update back.  The local update is already present in rounded FP32
code form before publication.  This module consumes that exact code first and
writes ``original + decoded`` directly to the owner-local parameter.  Remote
owners retain the literal decode and apply path.

Two independently timed execution arms are provided:

* local apply with the existing unequal peer-to-peer publication;
* local apply with one padded ``all_gather_into_tensor`` publication.

Neither arm changes an INT4 code, scale, decoded update, optimizer equation,
owner assignment, cadence, LR/WD, or Newton--Schulz iteration.  Promotion is
permitted only after complete multi-rank transitions prove bitwise equality
to the current packed-INT4 parent.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_layer_owner_collectives import owner_layer_lists
from .rlb_r01_9150_batched_response_inverse_compiled_span_int4 import (
    R019150BatchedResponseInverseCompiledSpanInt4Composite,
)
from .rlb_r01_9150_local_owner_int4_direct import (
    _packed_int4_transport_views,
)
from .rlb_method1_global_owner_transaction_cache_int4 import (
    Method1GlobalOwnerTransactionCachedFunctionalInt4Composite,
)
from .rlb_method1_local_layer_owner import _owner_local_reductions
from .rlb_layer_owner_int8_delta import BLOCK_ELEMENTS


RAGGED_FAMILY_ID = "int4_owner_local_apply_ragged_publication_v1"
PADDED_FAMILY_ID = "int4_owner_local_apply_padded_allgather_v1"


def _publish_ragged_remote_rows(
    send: torch.Tensor,
    outputs: list[torch.Tensor],
    *,
    rank: int,
) -> None:
    """Publish only remote rows; the local arm consumes prepacked codes."""

    rank = int(rank)
    if len(outputs) != int(dist.get_world_size()) or not 0 <= rank < len(outputs):
        raise RuntimeError("local-apply ragged publication world changed")
    if outputs[rank].shape != send.shape or outputs[rank].dtype != send.dtype:
        raise RuntimeError("local-apply local packet shape changed")
    operations = []
    for peer, output in enumerate(outputs):
        if peer == rank:
            continue
        if output.dtype != send.dtype or output.device != send.device:
            raise RuntimeError("local-apply remote packet changed")
        operations.append(dist.P2POp(dist.isend, send, peer))
        operations.append(dist.P2POp(dist.irecv, output, peer))
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _apply_decoded_owner_rows(
    composite,
    *,
    owner: int,
    inventory: tuple[int, ...],
    decoded_rows: torch.Tensor,
    local_owner: bool,
) -> torch.Tensor:
    """Apply one owner's decoded rows and return its maximum magnitude."""

    parameters = []
    deltas = []
    originals = []
    for local_index, layer in enumerate(inventory):
        offset = 0
        for family, count, shape in zip(
            composite._structural_families,
            composite._int4_family_counts,
            composite._int4_family_shapes,
        ):
            parameters.append(family[layer])
            deltas.append(
                decoded_rows[local_index, offset : offset + count].view(shape)
            )
            if local_owner:
                originals.append(
                    composite._int4_before[
                        local_index, offset : offset + count
                    ].view(shape)
                )
            offset += count
    if local_owner:
        if owner != int(composite.rank) or len(originals) != len(parameters):
            raise RuntimeError("local-apply owner inventory changed")
        for parameter, original, delta in zip(parameters, originals, deltas):
            torch.add(original, delta, out=parameter)
    else:
        torch._foreach_add_(parameters, deltas)
    return decoded_rows[:, : composite._int4_elements_per_layer].abs().amax()


class _Int4OwnerLocalApplyMixin:
    """Share the exact local-restore/decode elision between RLB methods."""

    def _int4_transport_send_views(self):
        return self._int4_send_packed, self._int4_send_scales

    def _publish_int4_owner_rows(self) -> None:
        _publish_ragged_remote_rows(
            self._int4_send_packet,
            self._int4_packets,
            rank=self.rank,
        )

    def _int4_remote_views(self, owner: int):
        return (
            self._int4_gathered_packed[owner],
            self._int4_gathered_scales[owner],
        )

    def _int4_actual_wire_bytes(self) -> tuple[int, int]:
        return (
            sum(value.numel() for value in self._int4_gathered_packed),
            4 * sum(value.numel() for value in self._int4_gathered_scales),
        )

    @torch.no_grad()
    def _step_int4_owner_local_apply(self):
        local_families = tuple(
            tuple(family[layer] for layer in self.owned_layers)
            for family in self._structural_families
        )
        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int4_family_counts,
            self._int4_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                self._int4_before[
                    local_index, offset : offset + count
                ].view(shape).copy_(parameter)
            offset += count

        with _owner_local_reductions():
            self.router.step()
            self.attention.step()

        # Form the same FP32 delta as the parent, but defer replacement of the
        # owner-local parameter until its decoded update is available.
        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int4_family_counts,
            self._int4_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                original = self._int4_before[
                    local_index, offset : offset + count
                ].view(shape)
                target = self._int4_delta[
                    local_index, offset : offset + count
                ].view(shape)
                torch.sub(parameter, original, out=target)
            offset += count

        delta_blocks = self._int4_delta.view(
            self._int4_delta.shape[0],
            self._int4_blocks_per_layer,
            BLOCK_ELEMENTS,
        )
        scratch_blocks = self._int4_scratch[
            : self._int4_delta.shape[0]
        ].view_as(delta_blocks)
        send_packed, send_scales = self._int4_transport_send_views()
        torch.abs(delta_blocks, out=scratch_blocks)
        torch.amax(scratch_blocks, dim=-1, out=send_scales)
        send_scales.div_(7.0)
        send_scales.clamp_(min=torch.finfo(torch.float32).tiny)
        torch.div(delta_blocks, send_scales.unsqueeze(-1), out=scratch_blocks)
        scratch_blocks.round_().clamp_(-7, 7)
        self._int4_nibbles.copy_(
            self._int4_scratch[: self._int4_delta.shape[0]]
        ).add_(8)
        nibble_pairs = self._int4_nibbles.view(
            self._int4_nibbles.shape[0], -1, 2
        )
        torch.bitwise_left_shift(nibble_pairs[..., 1], 4, out=send_packed)
        torch.bitwise_or(send_packed, nibble_pairs[..., 0], out=send_packed)

        self._publish_int4_owner_rows()

        inventories = owner_layer_lists()
        maxima = [None] * len(inventories)
        local_count = len(inventories[self.rank])
        local_rows = self._int4_scratch[:local_count]
        local_blocks = local_rows.view(
            local_count, self._int4_blocks_per_layer, BLOCK_ELEMENTS
        )
        torch.mul(local_blocks, send_scales.unsqueeze(-1), out=local_blocks)
        maxima[self.rank] = _apply_decoded_owner_rows(
            self,
            owner=self.rank,
            inventory=inventories[self.rank],
            decoded_rows=local_rows,
            local_owner=True,
        )

        for owner, inventory in enumerate(inventories):
            if owner == self.rank:
                continue
            owner_count = len(inventory)
            decode_codes = self._int4_decode_codes[:owner_count]
            decode_pairs = decode_codes.view(owner_count, -1, 2)
            packed, scales = self._int4_remote_views(owner)
            torch.bitwise_and(packed, 15, out=decode_pairs[..., 0])
            torch.bitwise_right_shift(packed, 4, out=decode_pairs[..., 1])
            decoded_rows = self._int4_scratch[:owner_count]
            decoded_rows.copy_(decode_codes).sub_(8.0)
            decode_blocks = decoded_rows.view(
                owner_count, self._int4_blocks_per_layer, BLOCK_ELEMENTS
            )
            torch.mul(decode_blocks, scales.unsqueeze(-1), out=decode_blocks)
            maxima[owner] = _apply_decoded_owner_rows(
                self,
                owner=owner,
                inventory=inventory,
                decoded_rows=decoded_rows,
                local_owner=False,
            )

        self.adamw.step()
        if any(value is None for value in maxima):
            raise RuntimeError("local-apply maximum inventory changed")
        self._last_delta_max_abs = float(torch.stack(maxima).amax().item())
        value_bytes, scale_bytes = self._int4_actual_wire_bytes()
        self._int4_wire_value_bytes = int(value_bytes)
        self._int4_wire_scale_bytes = int(scale_bytes)
        self._last_wire_elements = self._int4_wire_value_bytes
        self._last_wire_bytes = value_bytes + scale_bytes

    def _local_apply_execution_report(self):
        return {
            "local_packet_copy_removed": True,
            "owner_local_nibble_decode_removed": True,
            "owner_local_restore_and_add_fused": True,
            "decoded_updates_changed_vs_parent": False,
            "floating_point_update_changed_vs_parent": False,
            "optimizer_equations_changed": False,
            "newton_schulz_changed": False,
            "ns_steps": int(self.router.ns_steps),
            "quality_inheritance_requires_complete_bitwise_preflight": True,
        }


class _PaddedInt4AllGatherMixin(_Int4OwnerLocalApplyMixin):
    """Replace unequal P2P publication with one padded NCCL collective."""

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        world = len(self._int4_owner_counts)
        maximum = max(self._int4_owner_counts)
        packed_elements = self._int4_padded_elements // 2
        packet_bytes = maximum * packed_elements + (
            4 * maximum * self._int4_blocks_per_layer
        )
        first = self._structural_families[0][0]
        self._padded_int4_send_packet = torch.zeros(
            packet_bytes, device=first.device, dtype=torch.uint8
        )
        self._padded_int4_packets = torch.empty(
            world, packet_bytes, device=first.device, dtype=torch.uint8
        )
        self._padded_int4_send_packed, self._padded_int4_send_scales = (
            _packed_int4_transport_views(
                self._padded_int4_send_packet,
                rows=maximum,
                packed_elements=packed_elements,
                blocks_per_layer=self._int4_blocks_per_layer,
            )
        )
        packet_views = [
            _packed_int4_transport_views(
                packet,
                rows=maximum,
                packed_elements=packed_elements,
                blocks_per_layer=self._int4_blocks_per_layer,
            )
            for packet in self._padded_int4_packets.unbind(0)
        ]
        self._padded_int4_gathered_packed = [item[0] for item in packet_views]
        self._padded_int4_gathered_scales = [item[1] for item in packet_views]
        self._padded_int4_maximum_owner_count = maximum

    def _int4_transport_send_views(self):
        local_count = len(self.owned_layers)
        return (
            self._padded_int4_send_packed[:local_count],
            self._padded_int4_send_scales[:local_count],
        )

    def _publish_int4_owner_rows(self) -> None:
        dist.all_gather_into_tensor(
            self._padded_int4_packets.reshape(-1),
            self._padded_int4_send_packet,
        )

    def _int4_remote_views(self, owner: int):
        count = self._int4_owner_counts[owner]
        return (
            self._padded_int4_gathered_packed[owner][:count],
            self._padded_int4_gathered_scales[owner][:count],
        )

    def _int4_actual_wire_bytes(self) -> tuple[int, int]:
        world = len(self._int4_owner_counts)
        maximum = self._padded_int4_maximum_owner_count
        return (
            world * maximum * (self._int4_padded_elements // 2),
            4 * world * maximum * self._int4_blocks_per_layer,
        )

    def _padded_execution_report(self):
        return {
            "owner_publication": "one_padded_all_gather_into_tensor",
            "owner_rows_on_wire": (
                len(self._int4_owner_counts)
                * self._padded_int4_maximum_owner_count
            ),
            "mathematical_owner_rows": sum(self._int4_owner_counts),
            "collective_payload_bits_changed": False,
            "decoded_updates_changed_vs_parent": False,
        }


class Method1GlobalOwnerTransactionCachedFunctionalInt4LocalApplyComposite(
    _Int4OwnerLocalApplyMixin,
    Method1GlobalOwnerTransactionCachedFunctionalInt4Composite,
):
    _SCHEMA = RAGGED_FAMILY_ID + "_method1_composite"

    @torch.no_grad()
    def step(self):
        self._prepare_functional_rows()
        self._prepare_response_rows()
        self._prepare_metric_rows()
        try:
            return self._step_int4_owner_local_apply()
        finally:
            self.router.probe_count = self._owner_original_probe_count
            self.router.input_capture_count = self._owner_original_input_capture_count
            self._sync_capture_plan()

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._local_apply_execution_report())
        result["family_id"] = RAGGED_FAMILY_ID + "_method1"
        return result


class Method1GlobalOwnerTransactionCachedFunctionalInt4PaddedComposite(
    _PaddedInt4AllGatherMixin,
    Method1GlobalOwnerTransactionCachedFunctionalInt4LocalApplyComposite,
):
    _SCHEMA = PADDED_FAMILY_ID + "_method1_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._padded_execution_report())
        result["family_id"] = PADDED_FAMILY_ID + "_method1"
        return result


class R019150BatchedResponseInverseCompiledSpanInt4LocalApplyComposite(
    _Int4OwnerLocalApplyMixin,
    R019150BatchedResponseInverseCompiledSpanInt4Composite,
):
    _SCHEMA = RAGGED_FAMILY_ID + "_r01_composite"

    @torch.no_grad()
    def step(self):
        return self._step_int4_owner_local_apply()

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._local_apply_execution_report())
        result["family_id"] = RAGGED_FAMILY_ID + "_r01"
        return result


class R019150BatchedResponseInverseCompiledSpanInt4PaddedComposite(
    _PaddedInt4AllGatherMixin,
    R019150BatchedResponseInverseCompiledSpanInt4LocalApplyComposite,
):
    _SCHEMA = PADDED_FAMILY_ID + "_r01_composite"

    def execution_report(self):
        result = dict(super().execution_report())
        result.update(self._padded_execution_report())
        result["family_id"] = PADDED_FAMILY_ID + "_r01"
        return result


__all__ = (
    "PADDED_FAMILY_ID",
    "RAGGED_FAMILY_ID",
    "Method1GlobalOwnerTransactionCachedFunctionalInt4LocalApplyComposite",
    "Method1GlobalOwnerTransactionCachedFunctionalInt4PaddedComposite",
    "R019150BatchedResponseInverseCompiledSpanInt4LocalApplyComposite",
    "R019150BatchedResponseInverseCompiledSpanInt4PaddedComposite",
)
