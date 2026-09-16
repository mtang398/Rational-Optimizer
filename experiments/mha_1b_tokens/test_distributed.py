"""Four CPU/Gloo ranks: real CLI training versus stop/save/reload/resume.

Tiny synthetic caches are test fixtures, not scientific training/evaluation data.
This command does not submit jobs. Run on an existing CPU allocation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from .conditions import CONDITIONS
from .data import write_cache
from .io import atomic_json
from .test_portable import tiny_config, equal


def close_tree(test, a, b, stats, path=""):
    # Robust-FD factors represent S.T @ S; SVD row signs are not state semantics.
    if path.endswith("/factorized_rfd_persistent_scores"):
        test.assertEqual(a.shape,b.shape)
        torch.testing.assert_close(a.T@a,b.T@b,rtol=2e-5,atol=1e-7)
        stats["fd_gram_max_difference"] = max(stats.get("fd_gram_max_difference",0.),float((a.T@a-b.T@b).abs().max()))
        return
    if torch.is_tensor(a):
        if a.is_floating_point():
            torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-7)
            stats['max_absolute_difference'] = max(stats['max_absolute_difference'], float((a-b).abs().max()) if a.numel() else 0.)
        else: equal(test,a,b)
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for k in a: close_tree(test,a[k],b[k],stats,path+"/"+str(k))
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for i,(x,y) in enumerate(zip(a,b)): close_tree(test,x,y,stats,path+"/"+str(i))
    elif isinstance(a,float):
        test.assertTrue(abs(a-b) <= 1e-7+2e-5*abs(b), (a,b))
    else: test.assertEqual(a,b)


def execute(root, condition, *, interrupted):
    env = dict(os.environ, RATIONAL_OPT_TORCH_FALLBACK='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    destination = root / (condition+('-resumed' if interrupted else '-full'))
    commands = [(['--stop-after', '99' if condition == 'grain_tiller100_muon' else '8'], 'first'), (['--resume', 'latest'], 'resume')] if interrupted else [([], 'full')]
    for extra, label in commands:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '-m', 'experiments.mha_1b_tokens.test_worker', '--device', 'cpu', '--condition', condition,
                   '--config', str(root/(condition+'.json')), '--data', str(root/'manifest.json'), '--output', str(destination), *extra]
        print(f'{condition}: {label}, four CPU/Gloo ranks', flush=True)
        with (root/(condition+'-'+label+('-split' if interrupted else '')+'.log')).open('w') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    pointer = json.loads((destination/'latest.json').read_text())
    return Path(pointer['manifest']).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='retain evidence in a NEW directory')
    args = parser.parse_args()
    temporary = tempfile.TemporaryDirectory() if args.output is None else None
    root = Path(temporary.name) if temporary else args.output
    if not temporary: root.mkdir(parents=True, exist_ok=False)
    test = unittest.TestCase(); cfg = tiny_config()
    split_meta = {}
    for split, count in [('train', 7000), ('validation', 129)]:
        path = root/(split+'.u32'); meta = write_cache(path, np.arange(count,dtype=np.int64)%128, vocab_size=128)
        split_meta[split] = dict(path=path.name, tokens=meta['tokens'], sha256=meta['sha256'])
    atomic_json(root/'manifest.json', dict(dtype='<u4', tokenizer_revision=cfg['tokenizer_revision'], splits=split_meta))
    report = {}
    for condition in CONDITIONS:
        local = dict(cfg)
        if condition != 'grain_tiller100_muon':
            local.update(updates=12, loss_tokens_per_condition=12*cfg['loss_tokens_per_update'], validation_every=4)
        atomic_json(root/(condition+'.json'), local)
        a = execute(root, condition, interrupted=False); b = execute(root, condition, interrupted=True)
        ma = torch.load(a/'model.pt', weights_only=False); mb = torch.load(b/'model.pt', weights_only=False)
        stats = {'max_absolute_difference':0.}
        close_tree(test, ma, mb, stats)
        reference_opt = None
        for rank in range(4):
            ra = torch.load(a/f'rank-{rank}.pt', weights_only=False); rb = torch.load(b/f'rank-{rank}.pt', weights_only=False)
            close_tree(test, ra['optimizer'], rb['optimizer'], stats); equal(test, ra['rng']['torch'], rb['rng']['torch'])
            test.assertEqual(ra['next_batch_hash'], rb['next_batch_hash']); test.assertEqual(ra['update'],local['updates'])
            if reference_opt is None: reference_opt = ra['optimizer']
            else: equal(test,reference_opt,ra['optimizer'])
        report[condition] = dict(passed=True, deterministic_cpu_reduction=True, updates=local['updates'], world_size=4, model_optimizer_close=True, exact_counters_rng_next_batch=True, tolerance={'atol':1e-7,'rtol':2e-5}, **stats)
    atomic_json(root/'report.json',report); print(json.dumps(report,indent=2),flush=True)
    if temporary: temporary.cleanup()

if __name__ == '__main__': main()
