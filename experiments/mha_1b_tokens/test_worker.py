"""CPU-only deterministic reduction control for the distributed resume test.

All-gather + rank-order summation eliminates Gloo bucket-layout rounding changes
at DDP reconstruction. Optimizer code and all TILLER collectives stay unchanged.
This helper is never used by production launchers.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from . import train


def rank_order_mean(state, bucket):
    if bucket.buffer().device.type != 'cpu': raise RuntimeError('test reduction is CPU-only')
    world = dist.get_world_size()
    local = bucket.buffer().detach().clone().div_(world)
    values = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(values, local)
    total = values[0].clone()
    for value in values[1:]: total.add_(value)
    future = torch.futures.Future(); future.set_result(total)
    return future


class ControlledDDP(DistributedDataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_comm_hook(None, rank_order_mean)


if __name__ == '__main__':
    train.DDP = ControlledDDP
    train.main()
