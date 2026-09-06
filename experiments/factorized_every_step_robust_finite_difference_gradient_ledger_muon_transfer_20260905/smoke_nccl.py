#!/usr/bin/env python3
"""Exercise large four-rank NCCL collectives before endpoint admission."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=45))
    rank = dist.get_rank()
    value = torch.full(
        (64 * 1024 * 1024,), float(rank + 1),
        dtype=torch.float32, device=local_rank,
    )
    dist.broadcast(value, src=0)
    if value[0].item() != 1.0:
        raise RuntimeError("large NCCL broadcast returned the wrong value")
    value.fill_(float(rank + 1))
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)
    if value[0].item() != 10.0:
        raise RuntimeError("large NCCL all-reduce returned the wrong value")
    dist.barrier()
    if rank == 0:
        print(json.dumps({
            "bytes_per_collective": value.numel() * value.element_size(),
            "nccl_algo": os.environ.get("NCCL_ALGO"),
            "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"),
            "passed": True,
            "world_size": dist.get_world_size(),
        }, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
