"""Original TILLER for 100 updates, then native Muon on the same GRAIN model.

This is an explicit optimizer-switch experiment, not a modification of TILLER.
No response/history state is used after the switch. Momentum and auxiliary AdamW
continue without resetting. Checkpoint phase is authoritative and validated.
"""
import copy
import torch

from .switch_contract import CONDITION, SWITCH_UPDATE, POLICY
from .conditions import SWITCH_CONDITIONS, MHA_SWITCH, MHA_TILLER


class TillerThenMuon:
    schema = 'portable_mha_tiller100_muon_v1'

    def __init__(self, model, lr, accumulation, *, condition=CONDITION):
        if condition not in SWITCH_CONDITIONS: raise ValueError('unsupported switch identity')
        self._condition = condition
        self.model = model
        self.lr = lr
        self.accumulation = accumulation
        self.phase = 'tiller'
        self.handoff = None
        self.inner = self._new_tiller()

    def _new_tiller(self):
        from .optim import build_optimizer
        return build_optimizer(self.model, MHA_TILLER, self.lr, self.accumulation)

    @property
    def condition(self): return self._condition

    def __getattr__(self, name):
        if name == 'inner': raise AttributeError(name)
        return getattr(self.inner, name)

    def _remove_hooks(self):
        if self.inner.router is not None:
            for handle in self.inner.router._hook_handles: handle.remove()
            self.inner.router._hook_handles.clear()

    @torch.no_grad()
    def _to_muon(self, *, transfer):
        from .optim import CampaignOptimizer
        old = self.inner
        if transfer and (old.completed_steps != SWITCH_UPDATE or old._clipping_recorded):
            raise RuntimeError('switch requires the exact completed update100 boundary')
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        structural = [(n, p) for n, p in named
                      if p.ndim == 2 and n != 'embed_tokens.weight' and '.rlb_activation.' not in n]
        muon = torch.optim.Muon([p for _, p in structural], lr=self.lr, weight_decay=.1,
                               momentum=.95, ns_steps=5, adjust_lr_fn='match_rms_adamw')
        if transfer:
            buffers = {}
            for child in (old.router, old.attention):
                for group in child.param_groups:
                    for p in group['params']:
                        buffer = child.state[p]['momentum_buffer']
                        if buffer.shape != p.shape or not torch.isfinite(buffer).all():
                            raise RuntimeError('invalid handoff momentum')
                        bundle = next((b for b in old.bundles if b.logical is p), None)
                        owners = bundle.parameters if bundle else (p,)
                        parts = buffer.split(bundle.layout.widths, dim=0) if bundle else (buffer,)
                        for owner, value in zip(owners, parts):
                            if id(owner) in buffers: raise RuntimeError('duplicate handoff owner')
                            buffers[id(owner)] = value
            if set(buffers) != {id(p) for _, p in structural}:
                raise RuntimeError('incomplete handoff momentum coverage')
            for _, p in structural:
                muon.state[p]['momentum_buffer'] = buffers[id(p)].clone()
        auxiliary = old.children[-1]
        if old.roles[-1] != 'adamw': raise RuntimeError('unexpected auxiliary child')
        ids = {id(p) for _, p in structural}
        aux_ids = {id(p) for g in auxiliary.param_groups for p in g['params']}
        if ids & aux_ids or ids | aux_ids != {id(p) for _, p in named}:
            raise RuntimeError('switch ownership mismatch')
        self._remove_hooks()
        inventory = [dict(name=n, shape=list(p.shape), role='muon' if id(p) in ids else 'adamw',
                          weight_decay=.1 if id(p) in ids else 0.) for n, p in named]
        self.inner = CampaignOptimizer(self.model, self.condition, self.accumulation,
                                       (muon, auxiliary), ('muon', 'adamw'), inventory)
        self.inner.completed_steps = old.completed_steps
        self.inner.last_clipping = old.last_clipping
        self.phase = 'muon'
        if transfer:
            self.handoff = dict(completed_update=SWITCH_UPDATE, next_update=SWITCH_UPDATE+1,
                                structural_matrices=len(structural), auxiliary_preserved=True,
                                momentum_transferred=True, tiller_hooks_removed=True, policy=POLICY)

    def zero_grad(self, set_to_none=True):
        if self.phase == 'tiller' and self.completed_steps == SWITCH_UPDATE:
            self._to_muon(transfer=True)
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.phase == 'tiller' and self.completed_steps >= SWITCH_UPDATE:
            raise RuntimeError('switch must occur before forward at update101')
        self.inner.step()

    def state_dict(self):
        return dict(schema=self.schema, condition=self.condition, policy=POLICY, phase=self.phase,
                    handoff=self.handoff, completed_steps=self.completed_steps, inner=self.inner.state_dict())

    def load_state_dict(self, state):
        if (state.get('schema') != self.schema or state.get('condition') != self.condition
                or state.get('policy') != POLICY):
            raise ValueError('switch checkpoint identity mismatch')
        phase = state.get('phase'); step = state['inner']['completed_steps']
        if state.get('completed_steps') != step: raise ValueError('switch/child counter mismatch')
        if (type(step) is not int or phase not in ('tiller', 'muon')
                or (phase == 'tiller' and not 1 <= step <= SWITCH_UPDATE)
                or (phase == 'muon' and step < SWITCH_UPDATE)):
            raise ValueError('invalid switch checkpoint phase/counter')
        handoff = state.get('handoff')
        if phase == 'tiller' and handoff is not None:
            raise ValueError('premature handoff receipt')
        if phase == 'muon' and (not isinstance(handoff, dict) or handoff.get('policy') != POLICY
                               or handoff.get('completed_update') != SWITCH_UPDATE
                               or handoff.get('next_update') != SWITCH_UPDATE+1
                               or handoff.get('momentum_transferred') is not True
                               or handoff.get('auxiliary_preserved') is not True
                               or handoff.get('tiller_hooks_removed') is not True):
            raise ValueError('missing verified handoff receipt')
        if self.phase != phase:
            if phase == 'muon': self._to_muon(transfer=False)
            else:
                self._remove_hooks(); self.inner = self._new_tiller(); self.phase = 'tiller'
        self.inner.load_state_dict(state['inner'])
        self.handoff = copy.deepcopy(handoff)
