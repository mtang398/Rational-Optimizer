"""Native Q/K/V packing only; no GQA optimizer is exported or executed."""
from dataclasses import asdict, dataclass
import math
import torch
VERSION = "native_qkv_packing"

@dataclass(frozen=True)
class AttentionLayout:
    residual_width:int
    q_heads:int
    kv_heads:int
    head_dim:int
    head_group_size:int=4

    def __post_init__(self):
        values=asdict(self)
        if any(type(v) is not int or v<=0 for v in values.values()):raise ValueError('positive integer attention metadata required')
        if self.head_group_size!=4 or self.q_heads!=16 or self.kv_heads not in (8,16):raise ValueError('unsupported attention head inventory for bounded v1')
        if self.q_heads%self.kv_heads or self.kv_heads%4:raise ValueError('nonintegral GQA/block ratio')
        legacy=self.kv_heads==self.q_heads and self.q_heads*self.head_dim==self.residual_width
        native=self.kv_heads==8 and self.q_heads*self.head_dim==2*self.residual_width
        if not (legacy or native):raise ValueError('only legacy MHA or Qwen3-ratio attention supported')

    @property
    def widths(self):return (self.q_heads*self.head_dim,self.kv_heads*self.head_dim,self.kv_heads*self.head_dim)
    @property
    def offsets(self):
        q,k,v=self.widths;return (0,q,q+k,q+k+v)
    def shape(self,role):
        if role=='qkv':return (sum(self.widths),self.residual_width)
        if role=='attn_out':return (self.residual_width,self.widths[0])
        raise ValueError('explicit QKV/output role required')
    def blocks(self,role):
        width=4*self.head_dim
        if role=='qkv':
            return [('rows',start,start+width) for low,high in zip(self.offsets[:-1],self.offsets[1:]) for start in range(low,high,width)]
        if role=='attn_out':return [('columns',start,start+width) for start in range(0,self.widths[0],width)]
        raise ValueError('explicit QKV/output role required')
    def nominal_scale(self,role):
        rows,columns=self.shape(role);width=4*self.head_dim
        block_rank=min(width,columns) if role=='qkv' else min(rows,width)
        # Retain published operation order/constant on its exact legacy domain.
        if self.kv_heads==self.q_heads and self.widths[0]==self.residual_width:
            return 1.0/math.sqrt(3.0) if role=='qkv' else 1.0
        return math.sqrt(min(rows,columns)/(len(self.blocks(role))*block_rank))
    def manifest(self):
        return dict(version=VERSION,**asdict(self),qkv_offsets=list(self.offsets),qkv_shape=list(self.shape('qkv')),output_shape=list(self.shape('attn_out')),qkv_nominal_scale=self.nominal_scale('qkv'),output_nominal_scale=self.nominal_scale('attn_out'))


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
