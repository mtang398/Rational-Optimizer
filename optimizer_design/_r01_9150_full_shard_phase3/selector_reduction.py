"""Owner-local five-output selector reduction with a fused Triton backend."""

from __future__ import annotations

import torch


try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installations retain the reference fallback.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _selector_strided_partial_kernel(
        incoming,
        outgoing,
        incoming_gradient,
        outgoing_gradient,
        incoming_buffer,
        outgoing_buffer,
        incoming_stride_group: tl.constexpr,
        incoming_stride_width: tl.constexpr,
        incoming_stride_external: tl.constexpr,
        outgoing_stride_group: tl.constexpr,
        outgoing_stride_width: tl.constexpr,
        outgoing_stride_external: tl.constexpr,
        incoming_gradient_stride_group: tl.constexpr,
        incoming_gradient_stride_width: tl.constexpr,
        incoming_gradient_stride_external: tl.constexpr,
        outgoing_gradient_stride_group: tl.constexpr,
        outgoing_gradient_stride_width: tl.constexpr,
        outgoing_gradient_stride_external: tl.constexpr,
        incoming_buffer_stride_group: tl.constexpr,
        incoming_buffer_stride_width: tl.constexpr,
        incoming_buffer_stride_external: tl.constexpr,
        outgoing_buffer_stride_group: tl.constexpr,
        outgoing_buffer_stride_width: tl.constexpr,
        outgoing_buffer_stride_external: tl.constexpr,
        partial_incoming_exact,
        partial_outgoing_exact,
        partial_incoming_momentum,
        partial_outgoing_momentum,
        partial_budget,
        momentum,
        first_group: tl.constexpr,
        external: tl.constexpr,
        elements: tl.constexpr,
        chunks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Read original layer views; never materialize 81-block tensors."""

        local_block = tl.program_id(0)
        chunk = tl.program_id(1)
        group = first_group + 4 * local_block
        offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < elements
        width_offsets = offsets // external
        external_offsets = offsets - width_offsets * external

        incoming_value = tl.load(
            incoming
            + group * incoming_stride_group
            + width_offsets * incoming_stride_width
            + external_offsets * incoming_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        outgoing_value = tl.load(
            outgoing
            + group * outgoing_stride_group
            + width_offsets * outgoing_stride_width
            + external_offsets * outgoing_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        incoming_gradient_value = tl.load(
            incoming_gradient
            + group * incoming_gradient_stride_group
            + width_offsets * incoming_gradient_stride_width
            + external_offsets * incoming_gradient_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        outgoing_gradient_value = tl.load(
            outgoing_gradient
            + group * outgoing_gradient_stride_group
            + width_offsets * outgoing_gradient_stride_width
            + external_offsets * outgoing_gradient_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        incoming_buffer_value = tl.load(
            incoming_buffer
            + group * incoming_buffer_stride_group
            + width_offsets * incoming_buffer_stride_width
            + external_offsets * incoming_buffer_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        outgoing_buffer_value = tl.load(
            outgoing_buffer
            + group * outgoing_buffer_stride_group
            + width_offsets * outgoing_buffer_stride_width
            + external_offsets * outgoing_buffer_stride_external,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Mirror ATen/native/Lerp.h, including its |weight|=0.5 branch.
        # R01 uses momentum=0.95 and therefore takes the stable second form.
        momentum_is_small = tl.abs(momentum) < 0.5
        incoming_difference = (
            incoming_buffer_value - incoming_gradient_value
        )
        outgoing_difference = (
            outgoing_buffer_value - outgoing_gradient_value
        )
        incoming_momentum_value = tl.where(
            momentum_is_small,
            incoming_gradient_value + momentum * incoming_difference,
            incoming_buffer_value
            - incoming_difference * (1.0 - momentum),
        )
        outgoing_momentum_value = tl.where(
            momentum_is_small,
            outgoing_gradient_value + momentum * outgoing_difference,
            outgoing_buffer_value
            - outgoing_difference * (1.0 - momentum),
        )
        partial_offset = local_block * chunks + chunk
        tl.store(
            partial_incoming_exact + partial_offset,
            tl.sum(incoming_gradient_value * incoming_value, axis=0),
        )
        tl.store(
            partial_outgoing_exact + partial_offset,
            tl.sum(outgoing_gradient_value * outgoing_value, axis=0),
        )
        tl.store(
            partial_incoming_momentum + partial_offset,
            tl.sum(incoming_momentum_value * incoming_value, axis=0),
        )
        tl.store(
            partial_outgoing_momentum + partial_offset,
            tl.sum(outgoing_momentum_value * outgoing_value, axis=0),
        )
        tl.store(
            partial_budget + partial_offset,
            tl.sum(
                incoming_value * incoming_value
                + outgoing_value * outgoing_value,
                axis=0,
            ),
        )

    @triton.jit
    def _selector_finish_kernel(
        partial_incoming_exact,
        partial_outgoing_exact,
        partial_incoming_momentum,
        partial_outgoing_momentum,
        partial_budget,
        output,
        blocks: tl.constexpr,
        chunks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < chunks
        partial_offsets = block * chunks + offsets
        incoming_exact = tl.sum(
            tl.load(
                partial_incoming_exact + partial_offsets,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        outgoing_exact = tl.sum(
            tl.load(
                partial_outgoing_exact + partial_offsets,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        incoming_momentum = tl.sum(
            tl.load(
                partial_incoming_momentum + partial_offsets,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        outgoing_momentum = tl.sum(
            tl.load(
                partial_outgoing_momentum + partial_offsets,
                mask=mask,
                other=0.0,
            ),
            axis=0,
        )
        budget = tl.sum(
            tl.load(
                partial_budget + partial_offsets, mask=mask, other=0.0
            ),
            axis=0,
        )
        tl.store(output + block * 5 + 0, incoming_exact)
        tl.store(output + block * 5 + 1, outgoing_exact)
        tl.store(output + block * 5 + 2, incoming_momentum)
        tl.store(output + block * 5 + 3, outgoing_momentum)
        tl.store(output + block * 5 + 4, budget)


def _check_layer_inputs(values) -> tuple[int, int, int]:
    first = values[0]
    if first.ndim != 3 or first.dtype != torch.float32:
        raise RuntimeError("Phase-3 layer inputs must be FP32 [group,width,external]")
    for value in values[1:]:
        if (
            value.shape != first.shape
            or value.dtype != first.dtype
            or value.device != first.device
        ):
            raise RuntimeError("Phase-3 strided layer inventories differ")
    return tuple(int(value) for value in first.shape)


def fused_selector_layer_reductions(
    incoming,
    outgoing,
    incoming_gradient,
    outgoing_gradient,
    incoming_buffer,
    outgoing_buffer,
    *,
    first_group: int,
    momentum: float,
):
    """Reduce one layer's owner groups directly from original strided views.

    The CUDA path forms ``grad.lerp(buffer, momentum)`` in registers.  It
    therefore performs the archived Nesterov equation without allocating the
    four owner-local gradient/momentum tensors or the two endpoint tensors.
    """

    values = (
        incoming,
        outgoing,
        incoming_gradient,
        outgoing_gradient,
        incoming_buffer,
        outgoing_buffer,
    )
    groups, width, external = _check_layer_inputs(values)
    first_group = int(first_group)
    if not 0 <= first_group < 4 or first_group >= groups:
        raise RuntimeError("Phase-3 first owned group is invalid")
    owned_blocks = (groups - first_group + 3) // 4
    elements = width * external

    if incoming.is_cuda:
        if triton is None:
            raise RuntimeError("Phase-3 CUDA execution requires Triton")
        block_size = 1024
        chunks = triton.cdiv(elements, block_size)
        partials = [
            torch.empty(
                (owned_blocks, chunks),
                device=incoming.device,
                dtype=torch.float32,
            )
            for _ in range(5)
        ]
        strides = tuple(
            component
            for value in values
            for component in value.stride()
        )
        _selector_strided_partial_kernel[(owned_blocks, chunks)](
            *values,
            *strides,
            *partials,
            float(momentum),
            first_group=first_group,
            external=external,
            elements=elements,
            chunks=chunks,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        output = torch.empty(
            (owned_blocks, 5), device=incoming.device, dtype=torch.float32
        )
        finish_size = triton.next_power_of_2(chunks)
        _selector_finish_kernel[(owned_blocks,)](
            *partials,
            output,
            blocks=owned_blocks,
            chunks=chunks,
            BLOCK_SIZE=finish_size,
            num_warps=4,
        )
        return output, "triton_fused_layer_strided_two_stage"

    indices = torch.arange(
        first_group, groups, 4, device=incoming.device, dtype=torch.long
    )
    owned = tuple(value.index_select(0, indices) for value in values)
    (
        incoming_owned,
        outgoing_owned,
        incoming_gradient_owned,
        outgoing_gradient_owned,
        incoming_buffer_owned,
        outgoing_buffer_owned,
    ) = owned
    incoming_momentum = incoming_gradient_owned.lerp(
        incoming_buffer_owned, float(momentum)
    )
    outgoing_momentum = outgoing_gradient_owned.lerp(
        outgoing_buffer_owned, float(momentum)
    )
    packet = torch.stack((
        (incoming_gradient_owned * incoming_owned).sum(dim=(-2, -1)),
        (outgoing_gradient_owned * outgoing_owned).sum(dim=(-2, -1)),
        (incoming_momentum * incoming_owned).sum(dim=(-2, -1)),
        (outgoing_momentum * outgoing_owned).sum(dim=(-2, -1)),
        (
            incoming_owned.square().sum(dim=(-2, -1))
            + outgoing_owned.square().sum(dim=(-2, -1))
        ),
    ), dim=-1)
    return packet, "torch_layer_strided_reference"


__all__ = ("fused_selector_layer_reductions",)
