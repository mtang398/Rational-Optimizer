"""Portable optimizer composition using native Muon and unchanged reference MHA TILLER attention.

All structural matrices decay at .1. Tied embeddings, normalization vectors,
biases and GRAIN rational coefficients use AdamW without weight decay. AdamW
uses (.9,.95), eps=1e-8; Muon/TILLER use momentum=.95, NS5 and the repository's
match_rms_adamw adjustment. Clipping is a single global norm at 1.0 after DDP
gradient synchronization; TILLER receives that exact realized clipping norm.
No local gradient rescaling or replacement of TILLER reductions is performed.
"""
import math
import torch
from optimizer_design.tiller import TILLERRouter, TILLERAttentionOptimizer, configure_microbatch_count
from .packing import AttentionLayout, QKVBundle

from .conditions import CONDITIONS, SWITCH_CONDITIONS, MHA_ORDER, MHA_TILLER, MHA_MUON


def collect_blocks(model):
    """Collect canonical GRAIN hooks and explicit native attention bundles."""
    config=model.config
    layout=AttentionLayout(config['hidden_size'],config['num_attention_heads'],config['num_key_value_heads'],config['head_dim'])
    blocks = []
    for index, block in enumerate(model.layers):
        mlp = block.mlp
        activation = mlp.rlb_activation
        hidden = config['hidden_size']
        bundle=QKVBundle((block.attn.q_proj.weight,block.attn.k_proj.weight,block.attn.v_proj.weight),layout)
        blocks.append(dict(
            layer_index=index, num_layers=len(model.layers), block=block,
            mlp=mlp, module=activation, in_weight=mlp.in_proj.weight,
            out_weight=mlp.out_proj.weight, qkv_weight=bundle.logical, qkv_bundle=bundle, attention_layout=layout,
            attn_out_weight=block.attn.out.weight, numerator=activation.numerator,
            denominator=activation.denominator, coeff_logits=None, centers=None,
            beta=None, coeff_limit=0.0, groups=activation.groups,
            hidden_dim=activation.hidden_dim, eps=activation.eps))
    return blocks


class CampaignOptimizer:
    schema = 'portable_mha_optimizer_v1'

    def __init__(self, model, condition, accumulation, children, roles, inventory, router=None, attention=None, bundles=()):
        self.condition = condition
        self.accumulation = accumulation
        self.children = tuple(children)
        self.roles = tuple(roles)
        self.parameter_inventory = inventory
        self.parameters = tuple(p for p in model.parameters() if p.requires_grad)
        self.router = router
        self.attention = attention
        self.bundles = tuple(bundles)
        self.completed_steps = 0
        self.last_clipping = None
        self._clipping_recorded = False

    @property
    def param_groups(self):
        # Child load_state_dict can replace dictionaries; never cache this list.
        return [group for child in self.children for group in child.param_groups]

    def zero_grad(self, set_to_none=True):
        if self._clipping_recorded:
            raise RuntimeError('cannot discard gradients after clipping before step')
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:parameter.grad=None
                else:parameter.grad.detach_();parameter.grad.zero_()
        for child in self.children:
            child.zero_grad(set_to_none=set_to_none)

    def record_realized_clipping(self, preclip_norm, max_norm=1.0):
        norm = float(preclip_norm)
        if self._clipping_recorded or max_norm != 1.0 or not math.isfinite(norm) or norm < 0:
            raise ValueError('one finite realized clipping norm at max_norm=1 is required')
        if self.router is not None:
            self.router.record_realized_clipping(norm, max_norm)
        self.last_clipping = dict(preclip_norm=norm, max_norm=float(max_norm),
                                  factor=min(1., max_norm/(norm+1e-6)))
        self._clipping_recorded = True

    def clip_grad_norm(self, max_norm=1.0):
        if max_norm != 1.0 or self._clipping_recorded:
            raise ValueError('campaign requires exactly one clipping operation at max_norm=1')
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, max_norm, error_if_nonfinite=True)
        self.record_realized_clipping(norm, max_norm)
        return norm

    def step(self):
        if not self._clipping_recorded:
            raise RuntimeError('record clipping after synchronized backward and before optimizer step')
        for child in self.children:
            if child is self.attention:
                for bundle in self.bundles:bundle.gather()
            child.step()
            if child is self.attention:
                for bundle in self.bundles:bundle.scatter()
        self.completed_steps += 1
        self._clipping_recorded = False

    def state_dict(self):
        if self._clipping_recorded:
            raise RuntimeError('checkpoint only at completed optimizer boundaries')
        if self.router is not None and self.completed_steps == 0:
            raise RuntimeError('TILLER checkpoint requires at least one completed optimizer step')
        state=dict(schema=self.schema, condition=self.condition, accumulation=self.accumulation,
                    roles=self.roles, inventory=self.parameter_inventory, completed_steps=self.completed_steps,
                    last_clipping=self.last_clipping, children=[c.state_dict() for c in self.children])
        return state

    def load_state_dict(self, state):
        if (state.get('schema') != self.schema or state.get('condition') != self.condition
                or state.get('accumulation') != self.accumulation or tuple(state.get('roles', ())) != self.roles
                or state.get('inventory') != self.parameter_inventory or len(state.get('children', ())) != len(self.children)):
            raise ValueError('campaign optimizer checkpoint configuration/inventory mismatch')
        if type(state.get('completed_steps')) is not int or state['completed_steps'] < 0:
            raise ValueError('invalid optimizer step counter')
        if self.router is not None and state['completed_steps'] == 0:
            raise ValueError('TILLER checkpoint requires at least one completed optimizer step')
        if 'token_clock' in state: raise ValueError('token-time checkpoints are not original TILLER')
        for child, saved in zip(self.children, state['children']):
            child.load_state_dict(saved)
        self.completed_steps = state['completed_steps']
        self.last_clipping = state['last_clipping']
        self._clipping_recorded = False


def build_optimizer(model, condition, lr=3e-4, accumulation=4):
    if condition not in CONDITIONS or type(accumulation) is not int or accumulation < 1:
        raise ValueError('invalid campaign condition or accumulation')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('optimizer LR must be finite and positive')
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    AttentionLayout(model.config['hidden_size'], model.config['num_attention_heads'], model.config['num_key_value_heads'], model.config['head_dim'])
    if condition in SWITCH_CONDITIONS:
        if not model.grain: raise ValueError('switch experiment requires GRAIN throughout')
        from .tiller_then_muon import TillerThenMuon
        return TillerThenMuon(model, lr, accumulation, condition=condition)
    if bool(model.grain) != (condition == MHA_TILLER):
        raise ValueError('condition does not match model activation')
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    structural = []
    router = attention = None
    children, roles = [], [];bundles=[]
    if condition == MHA_TILLER:
        configure_microbatch_count(accumulation)
        blocks = collect_blocks(model)
        bundles=[b['qkv_bundle'] for b in blocks]
        structural = [b[key] for b in blocks for key in ('in_weight', 'out_weight', 'attn_out_weight')]+[p for bundle in bundles for p in bundle.parameters]
        # Public attention constructor locks the reference cell to 3e-4. The
        # scheduler operates on group LRs after construction, as in the trainer.
        kwargs = dict(lr=3e-4, weight_decay=.1, momentum=.95, ns_steps=5, beta2=.95, eps=1e-8)
        router = TILLERRouter(blocks, **kwargs)
        attention = TILLERAttentionOptimizer(blocks, router, adjust_lr_fn='match_rms_adamw', **kwargs)
        roles.extend(('tiller_router', 'tiller_attention_mha_reference'))
        children.extend((router, attention))
    elif condition == MHA_MUON:
        structural = [p for n, p in named if p.ndim == 2 and n != 'embed_tokens.weight']
        children.append(torch.optim.Muon(structural, lr=lr, weight_decay=.1,
                        momentum=.95, ns_steps=5, adjust_lr_fn='match_rms_adamw'))
        roles.append('muon')
    structural_ids = {id(p) for p in structural}
    if len(structural_ids) != len(structural):
        raise RuntimeError('structural parameter ownership overlaps')
    decay, no_decay = [], []
    for name, parameter in named:
        if id(parameter) not in structural_ids:
            no_wd = parameter.ndim < 2 or '.rlb_activation.' in name or name == 'embed_tokens.weight'
            (no_decay if no_wd else decay).append(parameter)
    adam_groups = []
    if decay:
        adam_groups.append(dict(params=decay, weight_decay=.1))
    if no_decay:
        adam_groups.append(dict(params=no_decay, weight_decay=0.))
    children.append(torch.optim.AdamW(adam_groups, lr=lr, betas=(.9, .95), eps=1e-8))
    roles.append('adamw')
    ownership = {}
    for role, child in zip(roles, children):
        for group in child.param_groups:
            group['lr'] = lr
            for parameter in group['params']:
                bundle=next((bundle for bundle in bundles if bundle.logical is parameter),None)
                real=bundle.parameters if bundle else (parameter,)
                for owner in real:
                    if id(owner) in ownership:raise RuntimeError('optimizer parameter ownership overlaps')
                    ownership[id(owner)] = (role, group['weight_decay'])
    if set(ownership) != {id(p) for _, p in named}:
        raise RuntimeError('optimizer parameter coverage is incomplete')
    inventory = [dict(name=n, shape=list(p.shape), role=ownership[id(p)][0],
                      weight_decay=ownership[id(p)][1]) for n, p in named]
    return CampaignOptimizer(model, condition, accumulation, children, roles, inventory, router, attention, bundles)
