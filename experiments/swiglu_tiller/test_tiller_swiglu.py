"""CPU references and execution contracts for the separate SwiGLU extension."""
import copy
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from . import core
from .model import NativeQwen3Decoder
from .tiller_swiglu import (SwiGLUTILLER, VERSION, group_scores,
                            participation_statistics, swiglu_factors)


def config():
    return dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=16, num_key_value_heads=16, head_dim=2,
                attention_bias=False, rms_norm_eps=1e-6, rope_theta=10000.,
                vocab_size=97, initializer_range=.02, rope_scaling=None,
                use_sliding_window=False, tie_word_embeddings=True)


def build():
    torch.manual_seed(1337)
    model = NativeQwen3Decoder(config())
    return model, SwiGLUTILLER(model, group_width=16)


def data(step=1):
    rng = torch.Generator().manual_seed(1000 + step)
    ids = torch.randint(0, 97, (8, 8), generator=rng)
    labels = torch.randint(0, 97, (8, 8), generator=rng)
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, -1] = False
    return ids, labels, mask


def update(model, optimizer, step=1, pieces=2):
    ids, labels, mask = data(step)
    optimizer.zero_grad()
    optimizer.begin_step(list(mask.chunk(pieces)))
    for x, y in zip(ids.chunk(pieces), labels.chunk(pieces)):
        logits = model(x)
        losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)
        optimizer.backward(losses)
    return optimizer.step()


def assert_tree(test, left, right, *, atol=0., rtol=0.):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree(test, left[key], right[key], atol=atol, rtol=rtol)
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for x, y in zip(left, right):
            assert_tree(test, x, y, atol=atol, rtol=rtol)
    else:
        test.assertEqual(left, right)


def distributed_worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + path, world_size=4, rank=rank)
    try:
        torch.manual_seed(1337)
        model = NativeQwen3Decoder(config())
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = SwiGLUTILLER(ddp, group_width=16)
        for step in range(1, 10):
            ids, labels, mask = data(step + rank * 30)
            optimizer.zero_grad()
            optimizer.begin_step(list(mask.chunk(2)))
            for i, (x, y) in enumerate(zip(ids.chunk(2), labels.chunk(2))):
                from contextlib import nullcontext
                with ddp.no_sync() if i == 0 else nullcontext():
                    losses = F.cross_entropy(ddp(x).flatten(0, 1), y.flatten(), reduction='none').reshape_as(y)
                    optimizer.backward(losses)
            metrics = optimizer.step()
            assert metrics['loss_tokens_this_update'] == 224
            # Every parameter and every persistent optimizer tensor must agree.
            def check(tree):
                if torch.is_tensor(tree):
                    reference = tree.detach().clone()
                    dist.broadcast(reference, 0)
                    torch.testing.assert_close(tree, reference, atol=0, rtol=0)
                elif isinstance(tree, dict):
                    for value in tree.values(): check(value)
                elif isinstance(tree, (list, tuple)):
                    for value in tree: check(value)
            check(model.state_dict())
            check(optimizer.state_dict())
            if step == 8:
                saved = copy.deepcopy(optimizer.state_dict())
                optimizer.close()
                optimizer = SwiGLUTILLER(ddp, group_width=16)
                optimizer.load_state_dict(saved)
        assert optimizer.completed_steps == 9
        assert metrics['functional_refresh'] is True
        optimizer.close()
    finally:
        dist.destroy_process_group()


class SwiGLUTILLERTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_swiglu_jacobian_and_three_projection_jvp(self):
        torch.manual_seed(7)
        x = torch.randn(5, 4, dtype=torch.float64)
        weights = [torch.randn(6, 4, dtype=torch.float64),
                   torch.randn(6, 4, dtype=torch.float64), torch.randn(4, 6, dtype=torch.float64)]
        directions = [torch.randn_like(w) for w in weights]
        cotangent = torch.randn(5, 4, dtype=torch.float64)
        gate, value = (x @ w.T for w in weights[:2])
        h, dg, dv = swiglu_factors(gate, value)
        independent = lambda g, v: g * torch.sigmoid(g) * v
        jac = torch.autograd.functional.jacobian(independent, (gate, value))
        for role, derivative in enumerate((dg, dv)):
            torch.testing.assert_close(jac[role].reshape(30, 30), torch.diag(derivative.flatten()))
        score = group_scores(x, gate, value, h, cotangent, weights[2], *directions, 2)
        for group in range(3):
            masked = [torch.zeros_like(w) for w in weights]
            sl = slice(group*2, (group+1)*2)
            masked[0][sl] = directions[0][sl]
            masked[1][sl] = directions[1][sl]
            masked[2][:, sl] = directions[2][:, sl]
            def f(a, b, c): return independent(x @ a.T, x @ b.T) @ c.T
            _, jvp = torch.autograd.functional.jvp(f, tuple(weights), tuple(masked))
            torch.testing.assert_close(score[:, group], (jvp * cotangent).sum(-1))
            eps = 1e-6
            diff = (f(*(w+eps*d for w, d in zip(weights, masked)))
                    - f(*(w-eps*d for w, d in zip(weights, masked)))) / (2*eps)
            torch.testing.assert_close(diff, jvp, atol=1e-8, rtol=1e-8)

    def test_participation_matches_dense_joint_jacobian(self):
        torch.manual_seed(8)
        g, v, c, out = [torch.randn(*s, dtype=torch.float64) for s in ((5, 6), (5, 6), (5, 4), (4, 6))]
        h, dg, dv = swiglu_factors(g, v)
        expected = torch.zeros(3, 4, dtype=torch.float64)
        for k in range(5):
            for group in range(3):
                sl = slice(group*2, (group+1)*2)
                j = torch.cat((torch.diag(dg[k, sl]), torch.diag(dv[k, sl])), 1)
                gram = j @ j.T
                rho = gram.trace().square() / (2 * gram.square().sum())
                q = (c[k] @ out)[sl]
                weight = (j.T @ q).square().sum() / 2
                z = h[k, sl]
                rho_out = z.square().sum().square() / (2*z.pow(4).sum())
                w_out = c[k].square().mean()
                expected[group] += torch.stack((rho*weight, weight, rho_out*w_out, w_out))
        torch.testing.assert_close(participation_statistics(g, v, h, c, out, 2), expected)

    def test_refresh_surrogate_history_and_auxiliary_recurrence(self):
        model, optimizer = build()
        try:
            measurements, deviations = [], []
            for step in range(1, 10):
                before = copy.deepcopy(optimizer.router.state_dict())
                old = [copy.deepcopy(optimizer.router.state[p]) for p in optimizer.router.roles['gate']]
                aux_old = {p: copy.deepcopy(optimizer.auxiliary.state[p]) for g in optimizer.auxiliary.param_groups for p in g['params']}
                original = optimizer.router.step
                original_advance = optimizer.router._advance_rows
                captured = {}
                def record():
                    captured['grads'] = [p.grad.clone() for p in optimizer.router.roles['gate']]
                    captured['all_grads'] = {role: torch.stack([p.grad.clone() for p in ps])
                                             for role, ps in optimizer.router.roles.items()}
                    captured['aux'] = {p: p.grad.clone() for p in aux_old}
                    return original()
                def record_advance(scores, decay, *, functional):
                    captured['ledger_cross'] = (scores.double().T @ decay.double()) / 32
                    return original_advance(scores, decay, functional=functional)
                chord = core._compiled_chord__compiled_factorized_adaptive_tangent_chord_direction
                with patch.object(optimizer.router, 'step', side_effect=record), patch.object(optimizer.router, '_advance_rows', side_effect=record_advance), patch.object(core, '_compiled_chord__compiled_factorized_adaptive_tangent_chord_direction', wraps=chord) as calls:
                    metrics = update(model, optimizer, step)
                self.assertEqual([call.kwargs['step'] for call in calls.call_args_list], [step]*5)
                self.assertEqual([call.kwargs['beta2'] for call in calls.call_args_list], [.95]*5)
                measurements.append(metrics['ledger_measurement'])
                deviations.append(metrics['coefficient_abs_deviation_max'])
                self.assertEqual(metrics['ledger_step'], step)
                self.assertEqual(metrics['total_loss_tokens'], 56*step)
                self.assertEqual(metrics['response_departure_energy'], 0.)
                self.assertLess(metrics['budget_residual'], 1e-5)
                anchor = optimizer.router.anchor
                old_anchor = next(iter(before['state'].values()), {})
                current_cross = captured['ledger_cross']
                expected_cross = current_cross if step == 1 else .95*old_anchor['factorized_rfd_persistent_decay_cross'].double() + .05*current_cross
                torch.testing.assert_close(anchor['factorized_rfd_persistent_decay_cross'], expected_cross.float())
                for role, gradient in captured['all_grads'].items():
                    grouped = (gradient.reshape(2, 32, 4, 16).permute(0, 2, 3, 1)
                               if role == 'down' else gradient.reshape(2, 4, 16, 32))
                    for suffix, summed in [('row', grouped.square().sum(-1)), ('column', grouped.square().sum(-2))]:
                        key = 'swiglu_' + role + '_' + suffix + '_second_moment'
                        prev = old_anchor.get(key, torch.zeros_like(summed))
                        torch.testing.assert_close(anchor[key], .95*prev + .05*summed)
                for p, grad, state in zip(optimizer.router.roles['gate'], captured['grads'], old):
                    torch.testing.assert_close(optimizer.router.state[p]['momentum_buffer'],
                                              .95*state.get('momentum_buffer', torch.zeros_like(p)) + .05*grad)
                for p, prior in aux_old.items():
                    state = optimizer.auxiliary.state[p]
                    torch.testing.assert_close(state['exp_avg_sq'], .95*prior.get('exp_avg_sq', torch.zeros_like(p)) + .05*captured['aux'][p].square())
                    self.assertEqual(int(state['step']), step)
                self.assertEqual(optimizer.auxiliary.param_groups[0]['betas'], (.9, .95))
                self.assertEqual(optimizer.auxiliary.param_groups[0]['weight_decay'], 0.)
            self.assertEqual(measurements, ['functional'] + ['gradient_surrogate']*7 + ['functional'])
            self.assertGreater(max(deviations), 1e-5)  # coordination really operated
        finally:
            optimizer.close()

    def test_saved_loaded_all_child_states_and_next_update(self):
        model, optimizer = build()
        try:
            for step in range(1, 9): update(model, optimizer, step)
            for g in optimizer.param_groups: g['lr'] = 1.23e-4
            stream = io.BytesIO()
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state()), stream)
            metrics = update(model, optimizer, 9)
            expected_model, expected_opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
            other, resumed = build()
            try:
                stream.seek(0)
                saved = torch.load(stream, weights_only=True)
                other.load_state_dict(saved['model'])
                resumed.load_state_dict(saved['optimizer'])
                torch.set_rng_state(saved['rng'])
                self.assertEqual(update(other, resumed, 9), metrics)
                assert_tree(self, expected_model, other.state_dict())
                assert_tree(self, expected_opt, resumed.state_dict())
            finally: resumed.close()
        finally: optimizer.close()

    def test_controlled_probes_and_loss_scaling_across_microbatch_decompositions(self):
        packets, gradients = [], []
        for pieces in (1, 2, 4):
            model, optimizer = build()
            original = optimizer.router._packets
            def record():
                result = original()
                packets.append(copy.deepcopy(result))
                gradients.append([p.grad.clone() for p in model.parameters()])
                return result
            try:
                with patch.object(optimizer.router, '_packets', side_effect=record):
                    update(model, optimizer, pieces=pieces)
            finally: optimizer.close()
        for candidate in packets[1:]:
            assert_tree(self, packets[0], candidate, atol=2e-7, rtol=2e-5)
        for candidate in gradients[1:]:
            assert_tree(self, gradients[0], candidate, atol=2e-7, rtol=2e-5)

    def test_original_attention_class_and_fixed_congruence_limit(self):
        model, optimizer = build()
        try:
            self.assertIs(type(optimizer.attention), core.TILLERAttentionOptimizer)
            update(model, optimizer)
            route = optimizer.router.anchor['swiglu_participation']
            self.assertEqual(tuple(route.shape), (2, 4, 2))
            # With chi=1, the full public factorized chord returns its parent,
            # while updating the actual history tensors (not skipping the path).
            x = torch.randn(2, 64, 32)
            g = torch.randn_like(x)
            rows, cols = torch.zeros(2, 4, 16), torch.zeros(2, 4, 32)
            result, meta = core._compiled_chord__compiled_factorized_adaptive_tangent_chord_direction(
                x, g, g, rows, cols, route[..., 0], torch.ones(2, 4), groups=4,
                width=16, grouped_axis='rows', beta2=.95, step=1)
            torch.testing.assert_close(result, x, atol=0, rtol=0)
            self.assertTrue(bool(rows.sum() > 0))
            self.assertEqual(float(meta['departure_energy'].abs().max()), 0.)
        finally: optimizer.close()

    def test_attention_packing_polar_normalization_and_history_against_literal_reference(self):
        model, optimizer = build()
        try:
            for step in range(1, 4):
                original = optimizer.attention.step
                expected, history = [], []
                def reference_then_step():
                    anchor = optimizer.attention.state[optimizer.attention.role_parameters['qkv'][0]]
                    for role in ('qkv', 'attn_out'):
                        gs = []
                        for p in optimizer.attention.role_parameters[role]:
                            grad = p.grad.clone(); gs.append(grad)
                            buffer = optimizer.attention.state[p].get('momentum_buffer', torch.zeros_like(p))
                            new_buffer = buffer.lerp(grad, .05)
                            nesterov = grad.lerp(new_buffer, .95)
                            # Explicit native head blocks; no adapter/head-map helper.
                            if role == 'qkv':
                                polar = torch.cat([core._basis_trust___zeropower_via_newton_schulz(t, 5)
                                                   for t in nesterov.split(8, dim=0)], dim=0)
                                polar = polar * (1/math.sqrt(3))  # BF16, as public reference
                            else:
                                polar = torch.cat([core._basis_trust___zeropower_via_newton_schulz(t, 5)
                                                   for t in nesterov.split(8, dim=1)], dim=1)
                            after = p.detach().clone() * (1-3e-4*.1)
                            after.add_(polar.float(), alpha=-3e-4*.2*math.sqrt(max(p.shape)))
                            expected.append((p, after, new_buffer))
                        gradient = torch.stack(gs)[:, None]
                        prefix = 'factorized_adaptive_tangent_' + ('qkv' if role == 'qkv' else 'attention_output')
                        for suffix, summed in [('row', gradient.square().sum(-1)), ('column', gradient.square().sum(-2))]:
                            key = prefix + '_' + suffix + '_second_moment'
                            history.append((key, .95*anchor.get(key, torch.zeros_like(summed)) + .05*summed))
                    return original()
                with patch.object(optimizer.attention, 'step', side_effect=reference_then_step):
                    update(model, optimizer, step)
                for p, after, momentum in expected:
                    torch.testing.assert_close(p, after, atol=2e-8, rtol=1e-5)
                    torch.testing.assert_close(optimizer.attention.state[p]['momentum_buffer'], momentum)
                anchor = optimizer.attention.state[optimizer.attention.role_parameters['qkv'][0]]
                for key, value in history: torch.testing.assert_close(anchor[key], value)
                for bundle in optimizer.bundles:
                    torch.testing.assert_close(torch.cat(list(bundle.parameters)), bundle.logical, atol=0, rtol=0)
        finally: optimizer.close()

    def test_bfloat16_forward_and_changing_actual_loss_counts(self):
        model, optimizer = build()
        try:
            tokens = 0
            for step in range(1, 4):
                ids, labels, mask = data(step)
                mask[:, -step:] = False
                optimizer.zero_grad()
                optimizer.begin_step([mask])
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(ids)
                    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels)
                optimizer.backward(losses)
                metrics = optimizer.step()
                tokens += int(mask.sum())
                self.assertEqual(metrics['loss_tokens_this_update'], int(mask.sum()))
                self.assertEqual(metrics['total_loss_tokens'], tokens)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        finally: optimizer.close()

    def test_missing_histories_are_rejected_instead_of_reinitialized(self):
        model, optimizer = build()
        try:
            update(model, optimizer)
            saved = copy.deepcopy(optimizer.state_dict())
            for key in ('swiglu_gate_row_second_moment', 'swiglu_value_column_second_moment',
                        'swiglu_down_row_second_moment', 'momentum_buffer'):
                bad = copy.deepcopy(saved)
                first = next(iter(bad['children'][0]['state'].values()))
                del first[key]
                with self.assertRaises(ValueError): optimizer.load_state_dict(bad)
                optimizer.load_state_dict(saved)
            bad = copy.deepcopy(saved)
            bad['children'][2]['param_groups'][0]['betas'] = (.9, .6634204312890623)
            with self.assertRaises(ValueError): optimizer.load_state_dict(bad)
            optimizer.load_state_dict(saved)
        finally: optimizer.close()

    def test_wrong_activation_and_missing_batch_are_rejected(self):
        model, optimizer = build()
        try:
            with self.assertRaises(RuntimeError): model(data()[0])
            mlp = model.layers[0].mlp
            mlp.forward = lambda x: mlp.down(torch.tanh(mlp.gate(x)) * mlp.value(x))
            with self.assertRaisesRegex(RuntimeError, 'exactly silu'):
                update(model, optimizer)
            self.assertEqual(optimizer.completed_steps, 0)
        finally: optimizer.close()

    def test_step_rejects_autocast_without_mutating_state(self):
        model, optimizer = build()
        try:
            model.eval()
            with self.assertRaisesRegex(RuntimeError, 'model.train'):
                optimizer.begin_step([data()[2]])
            model.train()
            ids, labels, mask = data()
            optimizer.begin_step([mask])
            loss = F.cross_entropy(model(ids).flatten(0, 1), labels.flatten(), reduction='none').reshape_as(labels)
            optimizer.backward(loss)
            before = copy.deepcopy(model.state_dict())
            with torch.autocast('cpu', dtype=torch.bfloat16), self.assertRaisesRegex(RuntimeError, 'outside autocast'):
                optimizer.step()
            assert_tree(self, before, model.state_dict())
            self.assertEqual(optimizer.completed_steps, 0)
            optimizer.step()
        finally: optimizer.close()

    def test_adapter_does_not_modify_backbone_or_auxiliary_inventory(self):
        torch.manual_seed(1337)
        model = NativeQwen3Decoder(config())
        before = copy.deepcopy(model.state_dict())
        model.eval()
        with torch.no_grad(): logits = model(data()[0])
        optimizer = SwiGLUTILLER(model, group_width=16)
        try:
            assert_tree(self, before, model.state_dict())
            with torch.no_grad(): torch.testing.assert_close(model(data()[0]), logits, atol=0, rtol=0)
            bundles = {id(b.logical): b.parameters for b in optimizer.bundles}
            owners = [owner for child in optimizer.children for g in child.param_groups
                      for p in g['params'] for owner in bundles.get(id(p), (p,))]
            self.assertEqual(len(owners), len({id(p) for p in owners}))
            self.assertEqual({id(p) for p in owners}, {id(p) for p in model.parameters()})
            aux = {id(p) for g in optimizer.auxiliary.param_groups for p in g['params']}
            for name, p in model.named_parameters():
                self.assertEqual(id(p) in aux, p.ndim == 1 or name == 'embed_tokens.weight')
        finally: optimizer.close()

    def test_checkpoint_configuration_and_counter_rejection(self):
        model, optimizer = build()
        try:
            update(model, optimizer)
            state = copy.deepcopy(optimizer.state_dict())
            for key, bad in [('version', 'grain_tiller'), ('group_width', 32), ('completed_steps', 3)]:
                invalid = copy.deepcopy(state); invalid[key] = bad
                with self.assertRaises(ValueError): optimizer.load_state_dict(invalid)
                optimizer.load_state_dict(state)
            optimizer.zero_grad()
            optimizer.begin_step([data()[2]])
            with self.assertRaises(RuntimeError): optimizer.state_dict()
            with self.assertRaises(RuntimeError): optimizer.step()
        finally: optimizer.close()

    def test_scope_rejects_gqa_and_invalid_grouping(self):
        c = config(); c['num_key_value_heads'] = 8
        with self.assertRaises(ValueError): SwiGLUTILLER(NativeQwen3Decoder(c), group_width=16)
        with self.assertRaises(ValueError): SwiGLUTILLER(NativeQwen3Decoder(config()), group_width=17)

    def test_import_and_update_without_any_grain_module(self):
        code = '''
import importlib.abc, sys
class RejectGrain(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('rational_opt', 'activation')):
            raise RuntimeError('GRAIN import attempted: '+fullname)
sys.meta_path.insert(0, RejectGrain())
from experiments.swiglu_tiller.test_tiller_swiglu import build, update
import torch
torch.set_num_threads(1)
m, o = build()
print(update(m, o)['version'])
o.close()
'''
        result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(VERSION, result.stdout)

    def test_four_cpu_ranks_and_resume_across_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker, args=(str(Path(tmp)/'gloo'),), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
