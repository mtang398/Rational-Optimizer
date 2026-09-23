"""Atomic verified checkpoints at completed update boundaries, including every rank."""
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .io import atomic_json,digest

def rank():return dist.get_rank() if dist.is_initialized() else 0

def barrier():
    if dist.is_initialized():dist.barrier()

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])

def prune_periodic(directory,identity,keep=2):
    """Retain recent periodic saves; never remove milestone/final checkpoints."""
    import re,shutil
    if type(keep) is not int or keep<2:raise ValueError('retain at least two periodic checkpoints')
    candidates=[]
    for path in Path(directory).glob('checkpoint-*-periodic'):
        match=re.fullmatch(r'checkpoint-(\d+)-periodic',path.name)
        if not match or path.is_symlink() or not path.is_dir():continue
        manifest=path/'manifest.json'
        if not manifest.exists():continue
        meta=json.loads(manifest.read_text())
        if meta.get('identity')!=identity:raise RuntimeError('refuse to prune another checkpoint identity')
        candidates.append((int(match[1]),path))
    for _,path in sorted(candidates)[:-keep]:shutil.rmtree(path)

def tensor_cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:tensor_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [tensor_cpu(x) for x in value]
    if isinstance(value,tuple):return tuple(tensor_cpu(x) for x in value)
    return value

def atomic_torch(path,value):
    with path.open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())

def save(directory,model,optimizer,update,identity,next_batch_hash):
    directory=Path(directory);temporary=directory.with_name(directory.name+'.partial')
    if rank()==0:
        if directory.exists() or temporary.exists():raise RuntimeError('checkpoint target already exists')
        temporary.mkdir(parents=True)
    barrier()
    if rank()==0:atomic_torch(temporary/'model.pt',model.state_dict())
    atomic_torch(temporary/f'rank-{rank()}.pt',{'optimizer':optimizer.state_dict(),'rng':capture_rng(),'update':update,'data_position':update,'next_batch_hash':next_batch_hash,'identity':identity})
    barrier()
    if rank()==0:
        artifacts={p.name:digest(p) for p in sorted(temporary.glob('*.pt'))}
        manifest={'schema':1,'update':update,'identity':identity,'artifacts':artifacts,'world_size':dist.get_world_size() if dist.is_initialized() else 1}
        atomic_json(temporary/'manifest.json',manifest)
        for name,expected in artifacts.items():
            if digest(temporary/name)!=expected:raise RuntimeError('checkpoint readback checksum failure')
        os.rename(temporary,directory)
        fd=os.open(directory.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    barrier();return directory/'manifest.json'

def load(manifest_path,model,optimizer,identity,next_hash_fn):
    manifest_path=Path(manifest_path);meta=json.loads(manifest_path.read_text());directory=manifest_path.parent
    expected={'model.pt'}|{f'rank-{i}.pt' for i in range(dist.get_world_size() if dist.is_initialized() else 1)}
    if meta.get('schema')!=1 or set(meta.get('artifacts',{}))!=expected:raise RuntimeError('checkpoint schema/artifact inventory mismatch')
    if meta['identity']!=identity:raise RuntimeError('checkpoint source/config/data/schedule identity differs')
    if meta['world_size']!=(dist.get_world_size() if dist.is_initialized() else 1):raise RuntimeError('checkpoint world size differs')
    if rank()==0:
        for name,expected in meta['artifacts'].items():
            if digest(directory/name)!=expected:raise RuntimeError('checkpoint artifact checksum mismatch')
    barrier()
    model.load_state_dict(torch.load(directory/'model.pt',map_location='cpu',weights_only=False))
    state=torch.load(directory/f'rank-{rank()}.pt',map_location='cpu',weights_only=False)
    if state['identity']!=identity or state['update']!=meta['update'] or state['data_position']!=meta['update']:raise RuntimeError('rank checkpoint identity/position differs')
    if state['optimizer']['completed_steps']!=meta['update']:raise RuntimeError('optimizer/data update count mismatch')
    optimizer.load_state_dict(state['optimizer'])
    if state['next_batch_hash']!=next_hash_fn(meta['update']):raise RuntimeError('next batch changed across resume')
    restore_rng(state['rng']);barrier();return meta['update']
