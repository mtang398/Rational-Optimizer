"""Random-initialized modern MHA decoder; never loads weights."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps
    def forward(self, x):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return self.weight*normalized.to(x.dtype)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.heads=c['num_attention_heads'];self.kv_heads=c['num_key_value_heads'];self.head_dim=c['head_dim']
        if self.heads != self.kv_heads or self.heads*self.head_dim != c['hidden_size']:
            raise ValueError('this package requires equal QKV heads and square attention output')
        self.q_width=self.heads*self.head_dim;self.kv_width=self.kv_heads*self.head_dim
        self.q_proj=nn.Linear(c['hidden_size'],self.q_width,bias=c['attention_bias'])
        self.k_proj=nn.Linear(c['hidden_size'],self.kv_width,bias=c['attention_bias'])
        self.v_proj=nn.Linear(c['hidden_size'],self.kv_width,bias=c['attention_bias'])
        self.out=nn.Linear(self.q_width,c['hidden_size'],bias=c['attention_bias'])
        self.q_norm=RMSNorm(self.head_dim,c['rms_norm_eps']);self.k_norm=RMSNorm(self.head_dim,c['rms_norm_eps'])
        inv=1./(c['rope_theta']**(torch.arange(0,self.head_dim,2,dtype=torch.float32)/self.head_dim))
        self.register_buffer('inv_freq',inv,persistent=False)
    def forward(self,x):
        batch,length,_=x.shape
        q,k,v=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        q=self.q_norm(q.view(batch,length,self.heads,self.head_dim)).transpose(1,2)
        k=self.k_norm(k.view(batch,length,self.kv_heads,self.head_dim)).transpose(1,2)
        v=v.view(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        # Split-half RoPE convention; frequencies repeat in halves, not interleaved pairs.
        with torch.autocast(device_type=x.device.type,enabled=False):
            phases=torch.outer(torch.arange(length,device=x.device,dtype=torch.float32),self.inv_freq.float())
            phases=torch.cat((phases,phases),-1)[None,None]
            cos,sin=phases.cos().to(q.dtype),phases.sin().to(q.dtype)
        def rotate(t): return torch.cat((-t[...,self.head_dim//2:],t[...,:self.head_dim//2]),-1)
        q=q*cos+rotate(q)*sin;k=k*cos+rotate(k)*sin
        k=k.repeat_interleave(self.heads//self.kv_heads,dim=1)
        v=v.repeat_interleave(self.heads//self.kv_heads,dim=1)
        attended=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True)
        return self.out(attended.transpose(1,2).reshape(batch,length,self.q_width))

class SwiGLU(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['hidden_size'];h=c['intermediate_size']
        self.gate=nn.Linear(d,h,bias=False);self.value=nn.Linear(d,h,bias=False);self.down=nn.Linear(h,d,bias=False)
    def forward(self,x): return self.down(F.silu(self.gate(x))*self.value(x))

class GrainMLP(nn.Module):
    def __init__(self,c,groups):
        super().__init__()
        from rational_opt import GRAIN
        d=c['hidden_size'];h=3*c['intermediate_size']//2
        if 2*h!=3*c['intermediate_size'] or h%groups: raise ValueError('GRAIN width/group mismatch')
        self.hidden_dim=h;self.groups=groups
        self.in_proj=nn.Linear(d,h,bias=False)
        self.rlb_activation=GRAIN(h,groups,init='silu',fit_range=5.,eps=c['rms_norm_eps'])
        self.out_proj=nn.Linear(h,d,bias=False)
    def forward(self,x):
        preactivation=self.in_proj(x)
        # The published fused operator and TILLER response statistics require FP32.
        # Matmul remains under the caller's BF16 autocast; rational algebra stays FP32.
        with torch.autocast(device_type=x.device.type,enabled=False):
            activated=self.rlb_activation(preactivation.float())
        return self.out_proj(activated)

class Block(nn.Module):
    def __init__(self,c,grain,groups):
        super().__init__();d=c['hidden_size'];eps=c['rms_norm_eps']
        self.input_layernorm=RMSNorm(d,eps);self.post_attention_layernorm=RMSNorm(d,eps)
        self.attn=Attention(c);self.mlp=GrainMLP(c,groups) if grain else SwiGLU(c)
    def forward(self,x):
        x=x+self.attn(self.input_layernorm(x))
        return x+self.mlp(self.post_attention_layernorm(x))

class ModernMHADecoder(nn.Module):
    def __init__(self,config,*,grain=False,groups=18):
        super().__init__();self.config=dict(config);self.grain=grain;self.groups=groups
        if config.get('rope_scaling') is not None or config.get('use_sliding_window') or not config['tie_word_embeddings']:
            raise ValueError('unsupported pinned config change')
        self.embed_tokens=nn.Embedding(config['vocab_size'],config['hidden_size'])
        self.layers=nn.ModuleList([Block(config,grain,groups) for _ in range(config['num_hidden_layers'])])
        self.norm=RMSNorm(config['hidden_size'],config['rms_norm_eps'])
        self.lm_head=nn.Linear(config['hidden_size'],config['vocab_size'],bias=False)
        self.lm_head.weight=self.embed_tokens.weight
        # initialize each unique tensor once, including tied embeddings.
        with torch.no_grad():
            for name,p in self.named_parameters():
                if 'rlb_activation' in name: continue  # preserve canonical rational coefficients
                if p.ndim>=2: nn.init.normal_(p,mean=0.,std=config['initializer_range'])
                elif name.endswith('weight'): p.fill_(1.)
                else: p.zero_()
    def forward(self,ids):
        x=self.embed_tokens(ids)
        for layer in self.layers: x=layer(x)
        return self.lm_head(self.norm(x))

@torch.no_grad()
def copy_shared(source,target):
    """Copy and explicitly compare every common attention/norm/embedding tensor."""
    original=dict(source.named_parameters());copied=[]
    for name,p in target.named_parameters():
        if '.mlp.' in name: continue
        if name not in original or p.shape!=original[name].shape: raise RuntimeError('shared parameter inventory mismatch: '+name)
        p.copy_(original[name])
        if not torch.equal(p,original[name]): raise RuntimeError('initial parameter mismatch: '+name)
        copied.append(name)
    expected={name for name in original if '.mlp.' not in name}
    if set(copied)!=expected: raise RuntimeError('incomplete initial shared parameter copy')
    return copied

@torch.no_grad()
def copy_to_hf(source,target):
    """Validation mapping for the SwiGLU reference only; not a GRAIN eval adapter."""
    if source.grain: raise ValueError('cannot load GRAIN into SwiGLU')
    target.model.embed_tokens.weight.copy_(source.embed_tokens.weight)
    target.model.norm.weight.copy_(source.norm.weight)
    for left,right in zip(source.layers,target.model.layers,strict=True):
        right.input_layernorm.weight.copy_(left.input_layernorm.weight)
        right.post_attention_layernorm.weight.copy_(left.post_attention_layernorm.weight)
        q,k,v=left.attn.q_proj.weight,left.attn.k_proj.weight,left.attn.v_proj.weight
        right.self_attn.q_proj.weight.copy_(q);right.self_attn.k_proj.weight.copy_(k);right.self_attn.v_proj.weight.copy_(v)
        right.self_attn.o_proj.weight.copy_(left.attn.out.weight)
        right.self_attn.q_norm.weight.copy_(left.attn.q_norm.weight);right.self_attn.k_norm.weight.copy_(left.attn.k_norm.weight)
        right.mlp.gate_proj.weight.copy_(left.mlp.gate.weight);right.mlp.up_proj.weight.copy_(left.mlp.value.weight);right.mlp.down_proj.weight.copy_(left.mlp.down.weight)
