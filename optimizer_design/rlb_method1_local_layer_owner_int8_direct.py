"""Allocation-light execution of Method1's blockwise-INT8 owner wire.

This preserves the block-256 symmetric INT8 values and FP32 scales of
``rlb_method1_local_layer_owner_int8``.  It changes only execution: original
parameters, FP32 deltas, quantization scratch, collective packets, and one-row
decode storage are persistent buffers.  A gathered layer is decoded directly
into the reusable row and immediately applied, rather than materializing a
complete FP32 model-sized update.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_layer_owner_collectives import owner_layer_lists
from .rlb_layer_owner_int8_delta import BLOCK_ELEMENTS
from .rlb_method1_local_layer_owner import (
    Method1LocalLayerOwnerComposite,
    _owner_local_reductions,
)


FAMILY_ID = "method1_local_owner_block256_int8_direct_apply_v1"


class Method1LocalLayerOwnerInt8DirectComposite(
    Method1LocalLayerOwnerComposite
):
    """Execute the existing INT8 approximation without full-model decoding."""

    _SCHEMA = FAMILY_ID + "_composite"

    def __init__(self, blocks, adamw, **kwargs):
        super().__init__(blocks, adamw, **kwargs)
        self._int8_family_counts = tuple(
            int(family[0].numel()) for family in self._structural_families
        )
        self._int8_family_shapes = tuple(
            tuple(int(item) for item in family[0].shape)
            for family in self._structural_families
        )
        for family, count, shape in zip(
            self._structural_families,
            self._int8_family_counts,
            self._int8_family_shapes,
        ):
            if any(
                int(parameter.numel()) != count
                or tuple(int(item) for item in parameter.shape) != shape
                for parameter in family
            ):
                raise RuntimeError("direct INT8 owner family shape changed")

        inventories = owner_layer_lists()
        maximum = max(map(len, inventories))
        elements = sum(self._int8_family_counts)
        padded = (
            (elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
        ) * BLOCK_ELEMENTS
        blocks_per_layer = padded // BLOCK_ELEMENTS
        first = self._structural_families[0][0]
        device = first.device
        self._int8_elements_per_layer = elements
        self._int8_padded_elements = padded
        self._int8_blocks_per_layer = blocks_per_layer
        self._int8_before = torch.empty(
            maximum, padded, device=device, dtype=torch.float32
        )
        self._int8_delta = torch.zeros_like(self._int8_before)
        self._int8_scratch = torch.empty_like(self._int8_before)
        self._int8_send_values = torch.empty(
            maximum, padded, device=device, dtype=torch.int8
        )
        self._int8_send_scales = torch.empty(
            maximum, blocks_per_layer, device=device, dtype=torch.float32
        )
        self._int8_gathered_values = torch.empty(
            len(inventories), maximum, padded, device=device, dtype=torch.int8
        )
        self._int8_gathered_scales = torch.empty(
            len(inventories),
            maximum,
            blocks_per_layer,
            device=device,
            dtype=torch.float32,
        )
        self._int8_decode_row = torch.empty(
            padded, device=device, dtype=torch.float32
        )
        self._int8_wire_value_bytes = 0
        self._int8_wire_scale_bytes = 0

    def _publish_int8_rows(self):
        """Publish values and scales with the parent's two collectives."""
        dist.all_gather_into_tensor(
            self._int8_gathered_values.reshape(-1),
            self._int8_send_values.reshape(-1),
        )
        dist.all_gather_into_tensor(
            self._int8_gathered_scales.reshape(-1),
            self._int8_send_scales.reshape(-1),
        )

    @torch.no_grad()
    def step(self):
        local_families = tuple(
            tuple(family[layer] for layer in self.owned_layers)
            for family in self._structural_families
        )
        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int8_family_counts,
            self._int8_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                self._int8_before[
                    local_index, offset : offset + count
                ].view(shape).copy_(parameter)
            offset += count

        with _owner_local_reductions():
            self.router.step()
            self.attention.step()

        offset = 0
        for family, count, shape in zip(
            local_families,
            self._int8_family_counts,
            self._int8_family_shapes,
        ):
            for local_index, parameter in enumerate(family):
                original = self._int8_before[
                    local_index, offset : offset + count
                ].view(shape)
                target = self._int8_delta[
                    local_index, offset : offset + count
                ].view(shape)
                torch.sub(parameter, original, out=target)
                parameter.copy_(original)
            offset += count

        delta_blocks = self._int8_delta.view(
            self._int8_delta.shape[0],
            self._int8_blocks_per_layer,
            BLOCK_ELEMENTS,
        )
        scratch_blocks = self._int8_scratch.view_as(delta_blocks)
        torch.abs(delta_blocks, out=scratch_blocks)
        torch.amax(
            scratch_blocks, dim=-1, out=self._int8_send_scales
        )
        self._int8_send_scales.div_(127.0)
        self._int8_send_scales.clamp_(
            min=torch.finfo(torch.float32).tiny
        )
        torch.div(
            delta_blocks,
            self._int8_send_scales.unsqueeze(-1),
            out=scratch_blocks,
        )
        scratch_blocks.round_().clamp_(-127, 127)
        self._int8_send_values.copy_(self._int8_scratch)

        self._publish_int8_rows()

        maxima = []
        inventories = owner_layer_lists()
        decode_blocks = self._int8_decode_row.view(
            self._int8_blocks_per_layer, BLOCK_ELEMENTS
        )
        for owner, inventory in enumerate(inventories):
            for local_index, layer in enumerate(inventory):
                torch.mul(
                    self._int8_gathered_values[
                        owner, local_index
                    ].view(self._int8_blocks_per_layer, BLOCK_ELEMENTS),
                    self._int8_gathered_scales[
                        owner, local_index
                    ].unsqueeze(-1),
                    out=decode_blocks,
                )
                maxima.append(
                    self._int8_decode_row[
                        : self._int8_elements_per_layer
                    ].abs().amax()
                )
                offset = 0
                for family, count, shape in zip(
                    self._structural_families,
                    self._int8_family_counts,
                    self._int8_family_shapes,
                ):
                    family[layer].add_(
                        self._int8_decode_row[
                            offset : offset + count
                        ].view(shape)
                    )
                    offset += count

        self.adamw.step()
        self._last_delta_max_abs = float(torch.stack(maxima).amax().item())
        world = len(inventories)
        self._int8_wire_value_bytes = (
            world * self._int8_send_values.numel()
        )
        self._int8_wire_scale_bytes = (
            4 * world * self._int8_send_scales.numel()
        )
        self._last_wire_elements = world * self._int8_send_values.numel()
        self._last_wire_bytes = (
            self._int8_wire_value_bytes + self._int8_wire_scale_bytes
        )

    def telemetry(self):
        result = super().telemetry()
        result.update({
            "rlb_layer_owner_int8_block_elements": BLOCK_ELEMENTS,
            "rlb_layer_owner_int8_wire_bytes": self._last_wire_bytes,
            "rlb_layer_owner_int8_value_bytes": self._int8_wire_value_bytes,
            "rlb_layer_owner_int8_scale_bytes": self._int8_wire_scale_bytes,
        })
        return result

    def execution_report(self):
        result = dict(super().execution_report())
        result.update({
            "transport_execution_family_id": FAMILY_ID,
            "direction_wire": "block256_symmetric_int8_plus_fp32_scales",
            "bf16_complete_parameter_delta": False,
            "persistent_collective_buffers": True,
            "full_model_fp32_decode_removed": True,
            "one_layer_decode_buffer": True,
            "scientific_equations_changed_vs_int8_parent": False,
            "floating_point_update_changed_vs_int8_parent": False,
            "fresh_quality_required": True,
        })
        return result


__all__ = ("FAMILY_ID", "Method1LocalLayerOwnerInt8DirectComposite")
