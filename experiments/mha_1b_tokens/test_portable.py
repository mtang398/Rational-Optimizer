"""CPU tests run explicitly with the rational reference path; never a GPU job."""
import copy
import json
import math
import os
os.environ.setdefault('RATIONAL_OPT_TORCH_FALLBACK', '1')
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from optimizer_design._tiller import core
from . import checkpoint
from .conditions import CONDITIONS
from .data import PackedBatches, write_cache, split_for
from .io import digest
from .model import ModernMHADecoder, copy_to_hf
from .optim import build_optimizer
from .train import initial_model, schedule, validate_config, validation


def tiny_config():
    cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    cfg['model'].update(hidden_size=32, intermediate_size=48, head_dim=2, num_hidden_layers=2, vocab_size=128)
    cfg['grain'].update(intermediate_size=72, groups=3, group_width=24)
    cfg.update(context=16, microbatch=1, accumulation=1, world_size=4,
               loss_tokens_per_update=64, updates=103, loss_tokens_per_condition=6592,
               small_validation_tokens=64)
    return cfg


def equal(test, a, b):
    if torch.is_tensor(a): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: equal(test, a[k], b[k])
    elif isinstance(a, (list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): equal(test,x,y)
    else: test.assertEqual(a,b)


def update(model, opt, batches):
    opt.zero_grad()
    for ids in batches:
        logits = model(ids)
        torch.nn.functional.cross_entropy(logits.flatten(0,1), ids.roll(-1,1).flatten()).div(len(batches)).backward()
    opt.clip_grad_norm(); opt.step()


class PortableTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(1337)

    def test_metrics_recovery_preserves_abandoned_work(self):
        from .train import reconcile_metrics
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metrics.jsonl'
            p.write_text(''.join(json.dumps({'step':n})+'\n' for n in range(1,5))+'{"step":')
            reconcile_metrics(p,2)
            self.assertEqual([json.loads(line)['step'] for line in p.read_text().splitlines()],[1,2])
            saved=list(Path(folder).glob('metrics-abandoned-*.jsonl'));self.assertEqual(len(saved),1)
            self.assertIn('"step": 4',saved[0].read_text());self.assertTrue(saved[0].read_text().endswith('{"step":'))
            reconcile_metrics(p,2);self.assertEqual(len(list(Path(folder).glob('metrics-abandoned-*'))),1)

    def test_import_requires_explicit_tokenizer_identity(self):
        from .prepare import import_manifest
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.json';p.write_text(json.dumps({'splits':{}}))
            with self.assertRaisesRegex(ValueError,'tokenizer'): import_manifest(p,Path(folder)/'dest',tiny_config())

    def test_fd_comparison_checks_the_represented_history(self):
        from .test_distributed import close_tree
        a={'factorized_rfd_persistent_scores':torch.tensor([[1.,2.],[3.,4.]])}
        b={'factorized_rfd_persistent_scores':a['factorized_rfd_persistent_scores']*torch.tensor([[-1.],[1.]])}
        close_tree(self,a,b,{'max_absolute_difference':0.})
        b['factorized_rfd_persistent_scores'][0,0]+=.1
        with self.assertRaises(AssertionError):close_tree(self,a,b,{'max_absolute_difference':0.})

    def test_default_counts_shapes_and_schedule(self):
        cfg = json.loads(Path(__file__).with_name('config.json').read_text()); validate_config(cfg)
        # Real module inventories on the meta device; numel is exact, no estimate.
        with torch.device('meta'):
            swiglu=ModernMHADecoder(cfg['model']); grain=ModernMHADecoder(cfg['model'],grain=True,groups=18)
        self.assertEqual(sum(p.numel() for p in swiglu.parameters()),537326080)
        self.assertEqual(sum(p.numel() for p in grain.parameters()),537331120)
        self.assertEqual(cfg['updates']*cfg['loss_tokens_per_update'],1000079360)
        self.assertEqual(schedule(0,cfg,.0003),.0003/200)
        self.assertGreater(schedule(3814,cfg,.0003),.00029)
        wrong=copy.deepcopy(cfg);wrong['model']['num_key_value_heads']=8
        with self.assertRaises(ValueError):validate_config(wrong)

    def test_upstream_mha_forward_loss_gradients(self):
        from transformers import Qwen3Config,Qwen3ForCausalLM
        c=tiny_config()['model'];a=ModernMHADecoder(c);b=Qwen3ForCausalLM(Qwen3Config(**c));b.config._attn_implementation='sdpa';copy_to_hf(a,b)
        ids=torch.randint(0,128,(2,16));x=a(ids);y=b(ids,use_cache=False).logits
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        la=torch.nn.functional.cross_entropy(x[:,:-1].reshape(-1,128),ids[:,1:].reshape(-1))
        lb=b(ids,labels=ids,use_cache=False).loss
        torch.testing.assert_close(la,lb,atol=2e-6,rtol=2e-5);la.backward();lb.backward()
        torch.testing.assert_close(a.embed_tokens.weight.grad,b.model.embed_tokens.weight.grad,atol=2e-5,rtol=2e-4)
        for left,right in zip(a.layers,b.model.layers):
            for name in ('q_proj','k_proj','v_proj'):
                torch.testing.assert_close(getattr(left.attn,name).weight.grad,getattr(right.self_attn,name).weight.grad,atol=2e-5,rtol=2e-4)

    def test_shared_initialization_and_auxiliary_routing(self):
        cfg=tiny_config();base,sha=initial_model(cfg,'swiglu_muon');other,other_sha=initial_model(cfg,'grain_tiller')
        self.assertEqual(sha,other_sha)
        grain_switch,switch_sha=initial_model(cfg,'grain_tiller100_muon');equal(self,other.state_dict(),grain_switch.state_dict())
        for condition in CONDITIONS:
            model,_=initial_model(cfg,condition);opt=build_optimizer(model,condition,accumulation=1)
            for record in opt.parameter_inventory:
                if record['name']=='embed_tokens.weight' or '.rlb_activation.' in record['name'] or len(record['shape'])==1:
                    self.assertEqual(record['role'],'adamw');self.assertEqual(record['weight_decay'],0)
            self.assertEqual(opt.children[-1].param_groups[0]['betas'],(.9,.95))

    def test_original_attention_math_and_complete_routing(self):
        model,_=initial_model(tiny_config(),'grain_tiller');opt=build_optimizer(model,'grain_tiller',accumulation=1)
        self.assertIs(type(opt.router),core.TILLERRouter);self.assertIs(type(opt.attention),core.TILLERAttentionOptimizer)
        qkv=torch.randn(2,96,32);out=torch.randn(2,32,32)
        expected=core._group_numerics___batched_zero_power(qkv.reshape(24,8,32),5).reshape_as(qkv)/math.sqrt(3)
        torch.testing.assert_close(core._temporal_geometry__attention_head_group_zero_power(qkv,5),expected,rtol=0,atol=0)
        expected=core._group_numerics___batched_zero_power(out.reshape(2,32,4,8).permute(0,2,1,3).reshape(8,32,8),5).reshape(2,4,32,8).permute(0,2,1,3).reshape_as(out)
        torch.testing.assert_close(core._temporal_geometry__attention_head_group_zero_power(out,5),expected,rtol=0,atol=0)
        for step in range(10):
            update(model,opt,[torch.randint(0,128,(2,16))]);self.assertEqual(opt.router._rfd_functional_refresh,step in (0,8))
        self.assertEqual(opt.attention.state[opt.attention.role_parameters['qkv'][0]]['factorized_adaptive_tangent_updates'],10)

    def test_switch_exact_first100_and_resume99_100_101(self):
        cfg=tiny_config();a,_=initial_model(cfg,'grain_tiller100_muon');b=copy.deepcopy(a)
        switch=build_optimizer(a,'grain_tiller100_muon',accumulation=1);full=build_optimizer(b,'grain_tiller',accumulation=1)
        batches=[[torch.randint(0,128,(2,16))] for _ in range(103)];saved={}
        for i,batch in enumerate(batches):
            update(a,switch,batch)
            if i<100:
                update(b,full,batch);equal(self,a.state_dict(),b.state_dict());equal(self,switch.inner.state_dict(),full.state_dict())
            if i+1 in (99,100,101):saved[i+1]=(copy.deepcopy(a.state_dict()),copy.deepcopy(switch.state_dict()))
        for position,(state,opt_state) in saved.items():
            m,_=initial_model(cfg,'grain_tiller100_muon');m.load_state_dict(state);o=build_optimizer(m,'grain_tiller100_muon',accumulation=1);o.load_state_dict(opt_state)
            for batch in batches[position:]:update(m,o,batch)
            equal(self,m.state_dict(),a.state_dict());equal(self,o.state_dict(),switch.state_dict())
        self.assertEqual(switch.phase,'muon');self.assertTrue(switch.handoff['momentum_transferred'])
        self.assertEqual({int(s['step']) for s in switch.children[-1].state.values()},{103})

    def test_baseline_and_full_tiller_atomic_resume(self):
        for condition in ('swiglu_muon','swiglu_adamw','grain_tiller'):
            model,_=initial_model(tiny_config(),condition);opt=build_optimizer(model,condition,accumulation=1)
            batch=[torch.randint(0,128,(2,16))];update(model,opt,batch)
            with tempfile.TemporaryDirectory() as directory:
                manifest=checkpoint.save(Path(directory)/'one',model,opt,1,{'test':condition},'batch-1')
                other,_=initial_model(tiny_config(),condition);other_opt=build_optimizer(other,condition,accumulation=1)
                checkpoint.load(manifest,other,other_opt,{'test':condition},lambda u:f'batch-{u}')
                update(model,opt,batch);update(other,other_opt,batch)
                equal(self,model.state_dict(),other.state_dict());equal(self,opt.state_dict(),other_opt.state_dict())
                with (manifest.parent/'model.pt').open('ab') as stream:stream.write(b'x')
                with self.assertRaisesRegex(RuntimeError,'checksum'):checkpoint.load(manifest,other,other_opt,{'test':condition},lambda u:f'batch-{u}')

    def test_uint32_disjoint_batch_positions_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.u32';ids=np.arange(1000,dtype=np.int64)+70000
            write_cache(path,ids,vocab_size=151936);data=PackedBatches(path,world_size=4,microbatch=2,accumulation=2,context=16)
            targets=[]
            for micro in range(2):
                for rank in range(4):targets.extend(data.batch(0,micro,rank)[1].flatten())
            self.assertEqual(sorted(targets),list(range(70001,70257)))
            self.assertEqual(split_for('Some   Document'),split_for('some document'))
            valid=Path(directory)/'valid.u32';write_cache(valid,np.arange(1000)%128,vocab_size=128)
            cfg=tiny_config();model,_=initial_model(cfg,'grain_tiller');opt=build_optimizer(model,'grain_tiller',accumulation=1)
            update(model,opt,[torch.randint(0,128,(2,16))]);before=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
            val=validation(model,valid,cfg,torch.device('cpu'),0,1)
            self.assertTrue(math.isfinite(val));equal(self,before,opt.state_dict());self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__': unittest.main()
