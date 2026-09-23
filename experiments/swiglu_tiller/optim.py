"""Native whole-matrix Muon baseline; fixed-SwiGLU TILLER composition."""
import math
import torch
from .tiller_swiglu import SwiGLUTILLER
CONDITIONS = ('swiglu_muon', 'swiglu_tiller')

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


def build_optimizer(model, condition, lr=3e-4, accumulation=16, group_width=256):
    if condition not in CONDITIONS:
        raise ValueError('only SwiGLU Muon and fixed-SwiGLU TILLER are supported')
    if condition == 'swiglu_tiller':
        optimizer = SwiGLUTILLER(model, group_width=group_width)
        optimizer.condition = condition
        optimizer.roles = ('tiller_swiglu_fixed_v1', 'tiller_attention_mha_fixed_congruence', 'adamw')
        for group in optimizer.param_groups: group['lr'] = lr
        return optimizer
    raw = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    named = list(raw.named_parameters())
    matrices = [p for n,p in named if p.ndim == 2 and n != 'embed_tokens.weight']
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for _,p in named if id(p) not in matrix_ids]
    children = (
        torch.optim.Muon(matrices, lr=lr, weight_decay=.1, momentum=.95, ns_steps=5, adjust_lr_fn='match_rms_adamw'),
        torch.optim.AdamW(auxiliary, lr=lr, weight_decay=0., betas=(.9,.95), eps=1e-8),
    )
    inventory = [dict(name=n, shape=list(p.shape), role='muon' if id(p) in matrix_ids else 'adamw',
                      weight_decay=.1 if id(p) in matrix_ids else 0.) for n,p in named]
    return CampaignOptimizer(raw, condition, accumulation, children, ('muon','adamw'), inventory)
