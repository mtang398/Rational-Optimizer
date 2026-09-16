"""Pack/unpack native equal-QKV parameters for the unchanged reference optimizer."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class AttentionLayout:
    residual_width: int
    q_heads: int
    kv_heads: int
    head_dim: int
    def __post_init__(self):
        if self.q_heads != 16 or self.kv_heads != 16 or self.head_dim * 16 != self.residual_width:
            raise ValueError('original TILLER requires 16 equal heads and square output')
    @property
    def widths(self): return (self.residual_width,) * 3

class QKVBundle:
    """Nontrainable packing workspace; native Q/K/V remain the only model owners."""
    def __init__(self,parameters,layout):
        self.parameters=tuple(parameters);self.layout=layout
        if len(self.parameters)!=3 or any(tuple(p.shape)!=(w,layout.residual_width) for p,w in zip(self.parameters,layout.widths)):
            raise ValueError('native Q/K/V packing inventory mismatch')
        if len({id(p) for p in self.parameters})!=3:raise ValueError('duplicated Q/K/V owner')
        if any(p.dtype!=torch.float32 or p.device!=self.parameters[0].device for p in self.parameters):raise ValueError('native attention parameters must share one device and FP32 master precision')
        self.logical=torch.cat([p.detach() for p in self.parameters],dim=0)
        if self.logical.requires_grad:raise RuntimeError('packing workspace must not be trainable')
    @torch.no_grad()
    def gather(self):
        if any(p.grad is None for p in self.parameters):raise RuntimeError('missing native attention gradient')
        self.logical.copy_(torch.cat([p.detach() for p in self.parameters],dim=0))
        self.logical.grad=torch.cat([p.grad.detach() for p in self.parameters],dim=0)
    @torch.no_grad()
    def scatter(self):
        for p,value in zip(self.parameters,self.logical.split(self.layout.widths,dim=0)):p.copy_(value)
