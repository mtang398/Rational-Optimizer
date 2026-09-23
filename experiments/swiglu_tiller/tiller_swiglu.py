"""Experimental fixed-SwiGLU specialization of TILLER, without GRAIN.

See TILLER_SWIGLU.md before use: fixed activation congruence is exactly one,
so the reference response-drift rotation is identically zero. The functional
loss ledger and equal-budget coordination remain active. This is a separately
named extension, not the published GRAIN construction or a campaign condition.
"""
from __future__ import annotations

import math
import torch
import torch.distributed as dist
from torch.nn import functional as F

from . import core
from .packing import AttentionLayout, QKVBundle

VERSION = "tiller_swiglu_fixed_v1"


def swiglu_factors(gate, value):
    """Exact features and diagonal partial derivatives; supports float64 tests."""
    sigmoid = gate.sigmoid()
    silu = F.silu(gate)
    return silu * value, value * sigmoid * (1 + gate * (1 - sigmoid)), silu


def group_scores(inputs, gate, value, features, cotangent, down_weight,
                 gate_direction, value_direction, down_direction, group_width):
    """JVP contracted with loss cotangent, one column per SwiGLU channel group."""
    _, dg, dv = swiglu_factors(gate, value)
    pullback = cotangent @ down_weight
    incoming = ((inputs @ gate_direction.T) * dg
                + (inputs @ value_direction.T) * dv) * pullback
    outgoing = features * (cotangent @ down_direction)
    return (incoming + outgoing).reshape(len(inputs), -1, group_width).sum(-1)


def participation_statistics(gate, value, features, cotangent, down_weight, width):
    """Trace-ratio participation for J=[diag(dg),diag(dv)] and h h^T.

    Returns weighted numerators/denominators, to reduce before taking ratios.
    """
    _, dg, dv = swiglu_factors(gate, value)
    energy = (dg.square() + dv.square()).reshape(len(gate), -1, width)
    h2 = features.square().reshape_as(energy)
    tiny = torch.finfo(gate.dtype).tiny
    incoming = energy.sum(-1).square() / (width * energy.square().sum(-1)).clamp_min(tiny)
    outgoing = h2.sum(-1).square() / (width * h2.square().sum(-1)).clamp_min(tiny)
    pulled = (cotangent @ down_weight).reshape_as(energy)
    in_weight = (energy * pulled.square()).mean(-1)
    out_weight = cotangent.square().mean(-1, keepdim=True).expand_as(in_weight)
    return torch.stack(((incoming * in_weight).sum(0), in_weight.sum(0),
                        (outgoing * out_weight).sum(0), out_weight.sum(0)), -1)


class SwiGLURouter(torch.optim.Optimizer):
    """Three native matrices, two logical roles (joint gate/value, down).

    Reuses the actual public Robust-FD ledger and transaction, including their
    two logical-role descent guards. No core monkeypatch or rational activation.
    """
    _advance_rows = core.TILLERRouter._advance_rows
    _transaction = core.TILLERRouter._transaction
    momentum = .95
    ns_steps = 5

    def __init__(self, blocks, group_width, process_group=None):
        self.pairs = blocks
        self.loss_probe_group = process_group
        self.width = group_width
        self.hidden, self.external = blocks[0]['in_weight'].shape
        self.groups = self.hidden // self.width
        self.roles = {
            'gate': [b['in_weight'] for b in blocks],
            'value': [b['value_weight'] for b in blocks],
            'down': [b['out_weight'] for b in blocks],
        }
        super().__init__([{'params': [p for ps in self.roles.values() for p in ps],
                          'lr': 3e-4, 'weight_decay': .1, 'version': VERSION}], {})
        self._attention_consumed = True
        self._attention_update = 0
        self._clip_factor = None
        self._capture_telemetry_next_step = True
        self._rfd_rows = self._rfd_selection = None
        self._rfd_gradient_scale = self._rfd_decay_derivative = None
        self.active = False
        self.last_metrics = {}
        self.handles = []
        for index, block in enumerate(blocks):
            self.handles.append(block['mlp'].register_forward_pre_hook(self._input_hook(index)))
            for role in ('gate', 'value'):
                self.handles.append(block[role].register_forward_hook(self._projection_hook(index, role)))
            self.handles.append(block['down'].register_forward_pre_hook(self._feature_hook(index)))
            self.handles.append(block['mlp'].register_forward_hook(self._output_hook(index)))

    @property
    def anchor(self):
        return self.state[self.pairs[0]['in_weight']]

    @property
    def completed_steps(self):
        return int(self.anchor.get('factorized_rfd_step', 0))

    def begin(self, masks, indices, loss_denominator):
        if self.active or not self._attention_consumed:
            raise RuntimeError('finish the previous optimizer/attention update first')
        self.active = True
        self.refresh = self.completed_steps % 8 == 0
        self.masks = masks
        self.indices = indices
        self.loss_denominator = loss_denominator
        self.calls = [0] * len(self.pairs)
        self.records = [[] for _ in self.pairs]
        self.pending = [None] * len(self.pairs)
        self._clip_factor = None

    def _input_hook(self, layer):
        @torch.no_grad()
        def capture(module, args):
            if not module.training:
                return
            if not self.active:
                raise RuntimeError('call begin_step before a training forward')
            micro = self.calls[layer]
            if micro >= len(self.masks) or self.pending[layer] is not None:
                raise RuntimeError('reentrant/checkpointed MLP forwards are not supported')
            x = args[0].detach().reshape(-1, self.external)
            if len(x) != self.masks[micro].numel():
                raise RuntimeError('MLP rows differ from the declared loss mask')
            self.calls[layer] += 1
            if self.refresh:
                ids = self.indices[micro]
                self.pending[layer] = {'indices': ids, 'x': x[ids].clone()}
        return capture

    def _projection_hook(self, layer, role):
        @torch.no_grad()
        def capture(module, args, output):
            if self.active and self.refresh:
                record = self.pending[layer]
                if record is None or role in record:
                    raise RuntimeError('SwiGLU projection capture is unaligned')
                record[role] = output.detach().reshape(-1, self.hidden)[record['indices']].clone()
        return capture

    def _feature_hook(self, layer):
        @torch.no_grad()
        def capture(module, args):
            if self.active and self.refresh:
                record = self.pending[layer]
                h = args[0].detach().reshape(-1, self.hidden)[record['indices']].clone()
                expected = F.silu(record['gate']) * record['value']
                if not torch.equal(h, expected):
                    raise RuntimeError('adapter requires exactly silu(gate) * value')
                record['h'] = h
        return capture

    def _output_hook(self, layer):
        def capture(module, args, output):
            if self.active and self.refresh:
                record = self.pending[layer]
                if record is None or 'h' not in record or not output.requires_grad:
                    raise RuntimeError('missing differentiable SwiGLU output')
                self.records[layer].append(record)
                self.pending[layer] = None

                def backward(gradient):
                    if 'cotangent' in record:
                        raise RuntimeError('a probe was backpropagated twice')
                    record['cotangent'] = gradient.detach().reshape(-1, self.external)[record['indices']].clone()
                    return gradient
                output.register_hook(backward)
        return capture

    def _packets(self):
        packets = []
        for records in self.records:
            if len(records) != len(self.masks) or any('cotangent' not in r for r in records):
                raise RuntimeError('incomplete functional probe backward')
            packet = [torch.cat([r[key] for r in records]).float()
                      for key in ('x', 'gate', 'value', 'h', 'cotangent')]
            packet[-1].mul_(self.loss_denominator * self._clip_factor)
            packets.append(packet)
        return packets

    def consume_response_homotopy(self):
        if self._attention_consumed:
            raise RuntimeError('attention route consumed twice')
        self._attention_consumed = True
        route = self.anchor['swiglu_participation']
        intrinsic = (route[..., 0].mean(-1) * route[..., 1].mean(-1)).clamp(0, 1).sqrt()
        return intrinsic, torch.ones_like(intrinsic), self._attention_update

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closures are incompatible with the explicit probe batch')
        if not self.active or not self._attention_consumed or self._clip_factor is None:
            raise RuntimeError('missing batch, synchronized clipping, or attention update')
        if any(n != len(self.masks) for n in self.calls):
            raise RuntimeError('missing MLP microbatch forwards')
        step = self.completed_steps + 1
        lr, wd = (float(self.param_groups[0][k]) for k in ('lr', 'weight_decay'))
        packets = self._packets() if self.refresh else None
        if self.refresh:
            stats = torch.stack([participation_statistics(g, v, h, c, b['out_weight'], self.width)
                                 for b, (x, g, v, h, c) in zip(self.pairs, packets)])
            if dist.is_initialized():
                dist.all_reduce(stats, group=self.loss_probe_group)
            nums, dens = stats[..., (0, 2)], stats[..., (1, 3)]
            self.anchor['swiglu_participation'] = torch.where(dens > 0, nums / dens.clamp_min(1e-38), 1.).clamp(0, 1)
        route = self.anchor['swiglu_participation']
        directions, contractions, momentum_contractions, energies = {}, {}, {}, []
        for role, parameters in self.roles.items():
            axis = 'columns' if role == 'down' else 'rows'
            momentum = core._batched_homotopy___foreach_nesterov(self, parameters)
            gradient = torch.stack([p.grad for p in parameters]).float()
            parent = core._temporal_geometry__rational_group_zero_power(
                momentum, 5, groups=self.groups, width=self.width).float()
            _, rows, columns = core._factorized_chord___state_for_direction(
                self, parent=parent, groups=self.groups, width=self.width,
                grouped_axis=axis, key_prefix='swiglu_' + role)
            selected, _ = core._compiled_chord__compiled_factorized_adaptive_tangent_chord_direction(
                parent, momentum, gradient, rows, columns,
                route[..., 1 if role == 'down' else 0], torch.ones_like(route[..., 0]),
                groups=self.groups, width=self.width, grouped_axis=axis, beta2=.95, step=step)
            scale = core._basis_trust___match_rms_adamw_adjustment(parameters[0].shape)
            selected = selected * scale
            directions[role] = selected
            grouped = lambda t: core._factorized_chord___group_view(
                t, groups=self.groups, width=self.width, grouped_axis=axis)[0]
            d = grouped(selected)
            contractions[role] = (d * grouped(gradient)).sum((-2, -1)).flatten()
            momentum_contractions[role] = (d * grouped(momentum)).sum((-2, -1)).flatten()
            energies.append(d.square().sum((-2, -1)).flatten())
        # The incoming logical role is the concatenated gate/value parameter block.
        exact = torch.stack((contractions['gate'] + contractions['value'], contractions['down']))
        moment = torch.stack((momentum_contractions['gate'] + momentum_contractions['value'], momentum_contractions['down']))
        weights = torch.stack(energies).sum(0)
        self._rfd_rows = None
        self._rfd_gradient_scale = None
        self._rfd_decay_derivative = sum((p.grad * p).sum() * wd for ps in self.roles.values() for p in ps)
        if self.refresh:
            scores, decay = [], []
            for layer, (b, packet) in enumerate(zip(self.pairs, packets)):
                args = (*packet, b['out_weight'])
                scores.append(group_scores(*args, *(directions[r][layer] for r in self.roles), self.width))
                decay.append(group_scores(*args, b['in_weight'] * wd, b['value_weight'] * wd,
                                          b['out_weight'] * wd, self.width).sum(-1))
            local = torch.cat((torch.stack(scores, 1).flatten(1), torch.stack(decay).sum(0)[:, None]), 1)
            global_packet, _ = core._fixed_probe_transaction___gather_variable_probe_rows(
                local, expected_global_rows=32, group=self.loss_probe_group)
            self._advance_rows(global_packet[:, :-1], global_packet[:, -1], functional=True)
        layer_ids = torch.arange(len(self.pairs), device=weights.device).repeat_interleave(self.groups)
        self._transaction(exact, moment, weights, layer_ids, total_layers=len(self.pairs), eta=lr)
        coefficients = self._rfd_selection.coefficients.reshape(len(self.pairs), self.groups)
        for role, parameters in self.roles.items():
            axis = 'columns' if role == 'down' else 'rows'
            blocks, restore = core._factorized_chord___group_view(
                directions[role], groups=self.groups, width=self.width, grouped_axis=axis)
            selected = restore(blocks * coefficients[..., None, None])
            core._batched_homotopy___foreach_apply(parameters, selected, decay=1-lr*wd, alpha=-lr)
        self._attention_update = step
        self._attention_consumed = False
        self.last_metrics = dict(version=VERSION, step=step, functional_refresh=self.refresh,
                                 ledger_step=self.completed_steps, beta2=.95,
                                 fixed_activation_congruence=1., response_departure_energy=0.,
                                 proposal_accepted=bool(self._rfd_selection.accepted.item()),
                                 coefficient_abs_deviation_mean=float((coefficients-1).abs().mean()),
                                 coefficient_abs_deviation_max=float((coefficients-1).abs().max()),
                                 budget_residual=float(self._rfd_selection.budget_residual.item()),
                                 ledger_measurement='functional' if self.refresh else 'gradient_surrogate')
        self.active = False
        self.records = []
        self.pending = []
        self._clip_factor = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SwiGLUTILLER:
    """Composition for the existing SwiGLU MHA decoder, with explicit loss masks.

    Parameters remain FP32; autocast may be used for forward/backward. DDP is
    supported, with synchronized gradients before step. No AMP GradScaler,
    activation checkpointing, FSDP, or tensor/pipeline parallelism in v1.
    """
    def __init__(self, model, *, group_width=256):
        ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self.process_group = model.process_group if ddp else None
        self.world = dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        if self.world > 1 and not ddp:
            raise ValueError('distributed construction requires a DDP-wrapped model')
        model = model.module if ddp else model
        c = model.config
        if getattr(model, 'grain', True) or not isinstance(c, dict):
            raise ValueError('requires the existing fixed-SwiGLU decoder')
        if c['num_attention_heads'] != 16 or c['num_key_value_heads'] != 16 or c['head_dim'] * 16 != c['hidden_size']:
            raise ValueError('v1 requires original sixteen-head MHA, with square output')
        if type(group_width) is not int or group_width < 1 or c['intermediate_size'] % group_width:
            raise ValueError('group width must divide the SwiGLU intermediate width')
        if c['intermediate_size'] <= c['hidden_size']:
            raise ValueError('v1 requires an expanded SwiGLU hidden width')
        if len(model.layers) * (c['intermediate_size'] // group_width) < 2:
            raise ValueError('TILLER needs at least two coordination coordinates')
        self.parameters = list(model.parameters())
        self.model = model
        device = self.parameters[0].device
        if any(p.dtype != torch.float32 or p.device != device or not p.requires_grad for p in self.parameters):
            raise ValueError('all model parameters must be trainable FP32 on one device')
        self.device = device
        self.inventory = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
        self.group_width = group_width
        self.blocks, self.bundles = [], []
        structural = []
        layout = AttentionLayout(c['hidden_size'], 16, 16, c['head_dim'])
        for i, layer in enumerate(model.layers):
            mlp, attn = layer.mlp, layer.attn
            modules = [mlp.gate, mlp.value, mlp.down, attn.q_proj, attn.k_proj, attn.v_proj, attn.out]
            if any(not isinstance(m, torch.nn.Linear) or m.bias is not None for m in modules):
                raise ValueError('only bias-free native SwiGLU/MHA linear projections supported')
            d, h = c['hidden_size'], c['intermediate_size']
            expected = [(h, d), (h, d), (d, h)] + [(d, d)] * 4
            if [tuple(m.weight.shape) for m in modules] != expected:
                raise ValueError('projection shapes differ from model configuration')
            structural.extend(m.weight for m in modules)
            bundle = QKVBundle((attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), layout)
            self.bundles.append(bundle)
            self.blocks.append(dict(layer_index=i, mlp=mlp, gate=mlp.gate, value=mlp.value,
                                    down=mlp.down, in_weight=mlp.gate.weight,
                                    value_weight=mlp.value.weight, out_weight=mlp.down.weight,
                                    qkv_weight=bundle.logical, attn_out_weight=attn.out.weight))
        ids = {id(p) for p in structural}
        if len(ids) != len(structural):
            raise ValueError('structural parameter ownership overlaps')
        auxiliary = [p for p in self.parameters if id(p) not in ids]
        embeddings = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
        if any(p.ndim > 1 and id(p) not in embeddings for p in auxiliary):
            raise ValueError('unrecognized auxiliary matrix; explicit routing is required')
        self.router = SwiGLURouter(self.blocks, group_width, self.process_group)
        kwargs = dict(lr=3e-4, weight_decay=.1, momentum=.95, ns_steps=5, beta2=.95, eps=1e-8)
        self.attention = core.TILLERAttentionOptimizer(self.blocks, self.router, adjust_lr_fn='match_rms_adamw', **kwargs)
        self.auxiliary = torch.optim.AdamW(auxiliary, lr=3e-4, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        self.children = (self.router, self.attention, self.auxiliary)
        self.total_loss_tokens = 0
        self.batch = None

    @property
    def param_groups(self):
        return [g for child in self.children for g in child.param_groups]

    @property
    def completed_steps(self):
        return self.router.completed_steps

    def zero_grad(self):
        if self.batch is not None:
            raise RuntimeError('cannot discard an unfinished update')
        for p in self.parameters:
            p.grad = None
        for bundle in self.bundles:
            bundle.logical.grad = None

    def begin_step(self, loss_masks, *, probe_indices=None):
        """Declare every local microbatch's actual loss mask before any forward.

        Optional probe_indices are local flattened update positions, sorted,
        unique, and loss-bearing. They allow controlled decomposition tests.
        """
        if self.batch is not None:
            raise RuntimeError('optimizer update already active')
        if not self.model.training:
            raise RuntimeError('call model.train() before begin_step')
        if any(p.grad is not None for p in self.parameters):
            raise RuntimeError('call zero_grad before begin_step')
        masks = [m.detach().clone() for m in loss_masks]
        if not masks or any(m.dtype != torch.bool or m.device != self.device for m in masks):
            raise ValueError('nonempty list of boolean masks on the model device required')
        valid = torch.cat([m.flatten() for m in masks]).nonzero().flatten()
        layout = core._probe_loss_image__import_fixed_global_probe_layout(32, self.rank, self.world)
        count = layout.local_probe_count
        if self.world > 32:
            raise ValueError('v1 supports at most 32 data-parallel ranks')
        counts = torch.tensor([len(valid), int(len(valid) < count)], dtype=torch.int64, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(counts, group=self.process_group)
        if counts[1].item():
            raise ValueError('each rank needs enough loss-bearing positions for its probes')
        total = int(counts[0].item())
        if probe_indices is None:
            slots = core._probe_loss_image__import_evenly_spaced_indices(len(valid), count, device=self.device)
            probe_indices = valid[slots]
        valid_probes = not (probe_indices.dtype != torch.int64 or probe_indices.device != self.device
                or tuple(probe_indices.shape) != (count,) or not bool(torch.isin(probe_indices, valid).all())
                or (count > 1 and not bool((probe_indices[1:] > probe_indices[:-1]).all())))
        bad_probes = torch.tensor(int(not valid_probes), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(bad_probes, group=self.process_group)
        if bad_probes.item():
            raise ValueError('probe identities must be sorted, unique, local loss-bearing positions')
        indices, offset = [], 0
        for mask in masks:
            indices.append(probe_indices[(probe_indices >= offset) & (probe_indices < offset + mask.numel())] - offset)
            offset += mask.numel()
        self.batch = dict(masks=masks, next_backward=0, global_tokens=total)
        self.router.begin(masks, indices, total / self.world)

    def backward(self, per_token_losses):
        if self.batch is None or self.batch['next_backward'] >= len(self.batch['masks']):
            raise RuntimeError('begin_step and exactly one backward per microbatch required')
        mask = self.batch['masks'][self.batch['next_backward']]
        if per_token_losses.shape != mask.shape or not per_token_losses.requires_grad:
            raise ValueError('pass differentiable unreduced per-token losses matching the mask')
        (per_token_losses[mask].sum() * (self.world / self.batch['global_tokens'])).backward()
        self.batch['next_backward'] += 1

    def step(self):
        if torch.is_autocast_enabled(self.device.type):
            raise RuntimeError('optimizer.step must run outside autocast')
        if self.batch is None or self.batch['next_backward'] != len(self.batch['masks']):
            raise RuntimeError('optimizer update has incomplete backwards')
        if any(p.grad is None for p in self.parameters):
            raise RuntimeError('missing parameter gradient')
        self._validate_hyperparameters()
        lrs = [float(g['lr']) for g in self.param_groups]
        if any(not math.isfinite(lr) or lr <= 0 for lr in lrs) or len(set(lrs)) != 1:
            raise ValueError('all optimizer children require the same positive scheduled LR')
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        self.router._clip_factor = min(1., 1. / (float(norm) + 1e-6))
        self.router.step()
        for bundle in self.bundles:
            bundle.gather()
        self.attention.step()
        for bundle in self.bundles:
            bundle.scatter()
        self.auxiliary.step()
        self.total_loss_tokens += self.batch['global_tokens']
        metrics = dict(self.router.last_metrics, total_loss_tokens=self.total_loss_tokens,
                       loss_tokens_this_update=self.batch['global_tokens'], lr=lrs[0], preclip_norm=float(norm))
        self.batch = None
        return metrics

    def state_dict(self):
        if self.batch is not None or not self.router._attention_consumed:
            raise RuntimeError('checkpoint only at a completed optimizer boundary')
        return dict(version=VERSION, group_width=self.group_width, world_size=self.world, inventory=self.inventory,
                    completed_steps=self.completed_steps, total_loss_tokens=self.total_loss_tokens,
                    children=[child.state_dict() for child in self.children])

    def load_state_dict(self, state):
        if self.batch is not None:
            raise RuntimeError('cannot restore in the middle of an update')
        if (state.get('version') != VERSION or state.get('group_width') != self.group_width or state.get('world_size') != self.world
                or state.get('inventory') != self.inventory or len(state.get('children', ())) != 3):
            raise ValueError('SwiGLU TILLER checkpoint identity/inventory mismatch')
        steps = state.get('completed_steps')
        tokens = state.get('total_loss_tokens')
        if type(steps) is not int or steps < 0 or type(tokens) is not int or tokens < 0 or ((steps == 0) != (tokens == 0)):
            raise ValueError('invalid persisted optimizer counters')
        for child, saved in zip(self.children, state['children']):
            child.load_state_dict(saved)
        if steps != self.completed_steps:
            raise ValueError('ledger and optimizer counters disagree')
        if steps:
            anchor = self.router.anchor
            core._previous(anchor, anchor['factorized_rfd_persistent_scores'])
            if 'factorized_rfd_reference_row_norm' not in anchor or 'swiglu_participation' not in anchor:
                raise ValueError('missing functional calibration or cached response route')
            route = anchor['swiglu_participation']
            if (tuple(route.shape) != (len(self.blocks), self.router.groups, 2)
                    or not bool(torch.isfinite(route).all() & (route >= 0).all() & (route <= 1).all())):
                raise ValueError('invalid cached response route')
            reference = anchor['factorized_rfd_reference_row_norm']
            if reference.numel() != 1 or not bool(torch.isfinite(reference).all() & (reference >= 0).all()):
                raise ValueError('invalid functional calibration')
            def tensor_ok(value, shape):
                return (torch.is_tensor(value) and tuple(value.shape) == tuple(shape)
                        and value.dtype == torch.float32 and bool(torch.isfinite(value).all()))
            for role, parameters in self.router.roles.items():
                for p in parameters:
                    if not tensor_ok(self.router.state[p].get('momentum_buffer'), p.shape):
                        raise ValueError('missing/invalid parent momentum')
                for suffix, size in [('row', self.router.width), ('column', self.router.external)]:
                    value = anchor.get('swiglu_' + role + '_' + suffix + '_second_moment')
                    if not tensor_ok(value, (len(self.blocks), self.router.groups, size)) or not bool((value >= 0).all()):
                        raise ValueError('missing/invalid factorized adaptive history')
            attn = self.attention.state[self.attention.role_parameters['qkv'][0]]
            if (attn.get('compact_response_homotopy_router_update') != steps
                    or attn.get('factorized_adaptive_tangent_updates') != steps
                    or any(int(self.auxiliary.state[p]['step']) != steps for g in self.auxiliary.param_groups for p in g['params'])):
                raise ValueError('optimizer-child counters disagree')
            for role, parameters in self.attention.role_parameters.items():
                for p in parameters:
                    if not tensor_ok(self.attention.state[p].get('momentum_buffer'), p.shape):
                        raise ValueError('missing/invalid attention momentum')
                prefix = 'qkv' if role == 'qkv' else 'attention_output'
                for suffix, size in [('row', parameters[0].shape[0]), ('column', parameters[0].shape[1])]:
                    value = attn.get('factorized_adaptive_tangent_' + prefix + '_' + suffix + '_second_moment')
                    if not tensor_ok(value, (len(self.blocks), 1, size)) or not bool((value >= 0).all()):
                        raise ValueError('missing/invalid attention adaptive history')
            for g in self.auxiliary.param_groups:
                for p in g['params']:
                    for key in ('exp_avg', 'exp_avg_sq'):
                        if not tensor_ok(self.auxiliary.state[p].get(key), p.shape):
                            raise ValueError('missing/invalid auxiliary history')
        self.router._attention_update = steps
        self.router._attention_consumed = True
        self.total_loss_tokens = tokens
        self._validate_hyperparameters()

    def _validate_hyperparameters(self):
        if (self.router.param_groups[0].get('version') != VERSION
                or self.router.param_groups[0]['weight_decay'] != .1
                or self.attention.param_groups[0]['weight_decay'] != .1
                or any(g['weight_decay'] != 0. or g['betas'] != (.9, .95) or g['eps'] != 1e-8
                       for g in self.auxiliary.param_groups)):
            raise ValueError('fixed SwiGLU TILLER optimizer cell changed')

    def close(self):
        self.router.close()
