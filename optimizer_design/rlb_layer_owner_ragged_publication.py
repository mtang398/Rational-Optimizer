"""Exact grouped publication for unequal layer-owner row inventories."""

from __future__ import annotations

import torch
import torch.distributed as dist


def publish_ragged_owner_rows(
    send: torch.Tensor,
    outputs: list[torch.Tensor],
    *,
    rank: int,
) -> None:
    """Publish unequal owner rows with one grouped point-to-point launch."""
    rank = int(rank)
    if len(outputs) != int(dist.get_world_size()) or not 0 <= rank < len(outputs):
        raise RuntimeError("ragged owner publication world changed")
    if outputs[rank].shape != send.shape or outputs[rank].dtype != send.dtype:
        raise RuntimeError("ragged owner local publication shape changed")
    outputs[rank].copy_(send)
    operations = []
    for peer, output in enumerate(outputs):
        if peer == rank:
            continue
        if output.dtype != send.dtype or output.device != send.device:
            raise RuntimeError("ragged owner remote publication changed")
        operations.append(dist.P2POp(dist.isend, send, peer))
        operations.append(dist.P2POp(dist.irecv, output, peer))
    for request in dist.batch_isend_irecv(operations):
        request.wait()


__all__ = ("publish_ragged_owner_rows",)
