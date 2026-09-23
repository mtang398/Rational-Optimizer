"""Portable admission, trainer/checkpoint and publication-data contracts."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from .model import ModernMHADecoder
from .train import schedule, validate_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class PackageTests(unittest.TestCase):
    def test_frozen_optimizer_math_identity(self):
        manifest = json.loads((HERE/'SOURCE_PROVENANCE.json').read_text())
        files = {r['file']:r for r in manifest['files']}
        self.assertEqual(hashlib.sha256((HERE/'core.py').read_bytes()).hexdigest(),files['core.py']['original_sha256'])
        original = (HERE/'tiller_swiglu.py').read_text().replace('from . import core','from optimizer_design._tiller import core').replace('from .packing import AttentionLayout, QKVBundle','from .tiller_gqa_v1 import AttentionLayout, QKVBundle')
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),files['tiller_swiglu.py']['original_sha256'])

    def test_real_parameter_inventory_and_full_schedule(self):
        for name,count in [('2b_3b_tokens.json',2056136704),('1b_1000_steps.json',1092250368)]:
            cfg=json.loads((HERE/'configs'/name).read_text()); validate_config(cfg)
            with torch.device('meta'): model=ModernMHADecoder(cfg['model'])
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            self.assertIs(model.lm_head.weight,model.embed_tokens.weight)
            self.assertEqual(schedule(199,cfg,.0003),.0003)
            self.assertGreater(schedule(cfg['updates']-1,cfg,.0003),.00025)

    def test_complete_matching_result_rows_and_perplexities(self):
        with (HERE/'results/training-every-step.csv').open() as f: steps=list(csv.DictReader(f))
        self.assertEqual([int(r['update']) for r in steps],list(range(1,11446)))
        with (HERE/'results/loss-ppl-every-100-steps.csv').open() as f: rows=list(csv.DictReader(f))
        self.assertEqual([int(r['update']) for r in rows],list(range(100,11401,100))+[11445])
        for r in rows:
            self.assertEqual(int(r['loss_tokens']),int(r['update'])*262144)
            for name in ('muon','tiller'):
                self.assertAlmostEqual(float(r[name+'_validation_ppl']),math.exp(float(r[name+'_validation_loss'])),places=12)
            self.assertAlmostEqual(float(r['validation_loss_gap_tiller_minus_muon']),float(r['tiller_validation_loss'])-float(r['muon_validation_loss']),places=12)

    def test_cli_training_and_resume(self):
        self._cli_resume(1)

    def test_four_rank_cli_training_and_resume(self):
        self._cli_resume(4)

    def _cli_resume(self, world):
        # Actual launcher, two optimizer paths and the refresh boundary.
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg=json.loads((HERE/'config.json').read_text())
            cfg['model'].update(hidden_size=32,intermediate_size=64,num_hidden_layers=2,head_dim=2,vocab_size=97)
            cfg.update(world_size=world,microbatch=2,accumulation=2,context=16,group_width=16,
                       updates=10,loss_tokens_per_update=64*world,loss_tokens_per_condition=640*world,
                       checkpoint_every=3,validation_every=2,small_validation_tokens=32)
            (p/'config.json').write_text(json.dumps(cfg))
            manifest=dict(tokenizer_revision=cfg['tokenizer_revision'],dtype='<u4',splits={})
            for split in ('train','validation'):
                path=p/(split+'.bin');np.random.default_rng(1337).integers(0,97,1000*world,dtype=np.uint32).astype('<u4').tofile(path)
                manifest['splits'][split]=dict(path=str(path),tokens=1000*world,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (p/'manifest.json').write_text(json.dumps(manifest))
            for condition in ('swiglu_muon','swiglu_tiller'):
                out=p/condition
                prefix=[sys.executable]
                if world>1: prefix+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}']
                command=prefix+['-m','experiments.swiglu_tiller.train','--condition',condition,'--data',str(p/'manifest.json'),'--output',str(out),'--config',str(p/'config.json'),'--device','cpu']
                for extra in (['--stop-after','3'],['--resume','latest']):
                    result=subprocess.run(command+extra,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},capture_output=True,text=True,timeout=180)
                    self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-6000:])
                summary=json.loads((out/'result.json').read_text())
                self.assertTrue(summary['completed']);self.assertEqual(summary['update'],10)
                rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['step'] for r in rows],list(range(1,11)))
                self.assertTrue(all(math.isfinite(r['loss']) for r in rows))
                self.assertEqual(rows[-1]['loss_tokens'],640*world)
                self.assertAlmostEqual(rows[-1]['validation_ppl'],math.exp(rows[-1]['validation_loss']))
                if condition=='swiglu_tiller':
                    self.assertEqual([r['step'] for r in rows if r['functional_refresh']],[1,9])
                    self.assertEqual(rows[-1]['ledger_step'],10)


if __name__ == '__main__': unittest.main()
