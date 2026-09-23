"""Portable torchrun trainer. All training tokens come from verified local caches."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from . import checkpoint
from .optim import CONDITIONS
from .data import PackedBatches
from .evaluation_state import evaluation_state
from .io import atomic_json, digest, object_digest, source_identity
from .model import ModernMHADecoder
from .optim import build_optimizer


def schedule(completed, cfg, base_lr):
    step = completed + 1
    if step <= cfg['warmup_updates']:
        return base_lr * step / cfg['warmup_updates']
    progress = (step-cfg['warmup_updates']) / (cfg['schedule_horizon_updates']-cfg['warmup_updates'])
    return base_lr * (cfg['min_lr_ratio'] + (1-cfg['min_lr_ratio']) * .5 * (1+math.cos(math.pi*progress)))


def initial_model(cfg, condition):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    model = ModernMHADecoder(cfg['model'])
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode()); h.update(parameter.detach().contiguous().numpy().tobytes())
    return model, h.hexdigest()


def validate_config(cfg):
    c = cfg['model']
    if c['num_attention_heads'] != 16 or c['num_key_value_heads'] != 16 or 16*c['head_dim'] != c['hidden_size']:
        raise ValueError('original TILLER attention requires 16 equal heads and square output')
    if c['intermediate_size'] % cfg['group_width']:
        raise ValueError('SwiGLU hidden width must divide into complete groups')
    if set(cfg['conditions']) != set(CONDITIONS) or cfg['seed'] != 1337:
        raise ValueError('condition inventory or seed changed')
    if cfg['world_size']*cfg['microbatch']*cfg['accumulation']*cfg['context'] != cfg['loss_tokens_per_update']:
        raise ValueError('global loss-bearing token batch mismatch')
    if cfg['updates']*cfg['loss_tokens_per_update'] != cfg['loss_tokens_per_condition']:
        raise ValueError('target tokens/updates mismatch')
    if not 0 <= cfg['warmup_updates'] < cfg['schedule_horizon_updates'] or cfg['updates'] > cfg['schedule_horizon_updates']:
        raise ValueError('invalid schedule horizon')


def batch_hash(data, update, rank):
    try: x, y, mask = data.batch(update, 0, rank)
    except IndexError: return 'END'
    return hashlib.sha256(x.tobytes()+y.tobytes()+mask.tobytes()).hexdigest()


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else contextlib.nullcontext()


def loss(logits, labels):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


def baseline_train_step(model, optimizer, data, completed, cfg, device, rank, world):
    optimizer.zero_grad()  # Switch at the completed-100 boundary, before hooks/forward.
    if optimizer.router is not None and (completed+1) % 100 == 0:
        optimizer.router.set_telemetry_capture(True)
        optimizer.attention.set_telemetry_capture(True)
    lr = schedule(completed, cfg, cfg['conditions'][optimizer.condition]['lr'])
    for group in optimizer.param_groups: group['lr'] = lr
    count = torch.zeros((), device=device, dtype=torch.int64)
    summed = torch.zeros((), device=device)
    for micro in range(cfg['accumulation']):
        x, y, mask = data.batch(completed, micro, rank)
        if not mask.all(): raise RuntimeError('packing protocol requires every target loss-bearing')
        count += int(mask.sum())
        sync = model.no_sync() if isinstance(model, DDP) and micro+1 < cfg['accumulation'] else contextlib.nullcontext()
        with sync:
            with autocast(device): value = loss(model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device))
            (value/cfg['accumulation']).backward()
        summed += value.detach()/cfg['accumulation']
    if dist.is_initialized(): dist.all_reduce(count)
    if count.item() != cfg['loss_tokens_per_update']: raise RuntimeError('actual mask count differs')
    norm = optimizer.clip_grad_norm()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    start = time.perf_counter(); optimizer.step()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    optimizer_seconds = time.perf_counter()-start
    if dist.is_initialized(): dist.all_reduce(summed)
    value = float(summed/world)
    if not math.isfinite(value): raise RuntimeError('nonfinite training loss')
    return dict(loss=value, lr=lr, loss_tokens_this_update=int(count), gradient_norm=float(norm),
                optimizer_seconds=optimizer_seconds,
                phase=getattr(optimizer, 'phase', 'tiller' if optimizer.router is not None else 'baseline'),
                functional_probe_refreshed=bool(optimizer.router._rfd_functional_refresh) if optimizer.router is not None else False,
                optimizer_child_lrs={role:[g['lr'] for g in child.param_groups] for role,child in zip(optimizer.roles,optimizer.children)})



def train_step(model, optimizer, data, completed, cfg, device, rank, world):
    if optimizer.condition == 'swiglu_muon':
        return baseline_train_step(model, optimizer, data, completed, cfg, device, rank, world)
    optimizer.zero_grad()
    prepared = [data.batch(completed, m, rank) for m in range(cfg['accumulation'])]
    masks = [torch.from_numpy(m).to(device=device, dtype=torch.bool) for _,_,m in prepared]
    if any(not bool(mask.all()) for mask in masks):
        raise RuntimeError('bound cache requires every target loss-bearing')
    optimizer.begin_step(masks)
    if (completed+1) % 100 == 0: optimizer.attention.set_telemetry_capture(True)
    if optimizer.batch['global_tokens'] != cfg['loss_tokens_per_update']:
        raise RuntimeError('actual global loss count differs')
    lr = schedule(completed, cfg, cfg['conditions'][optimizer.condition]['lr'])
    for group in optimizer.param_groups: group['lr'] = lr
    summed = torch.zeros((), device=device)
    observed = 0
    for micro,(x,y,mask) in enumerate(prepared):
        x,y = torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
        sync = model.no_sync() if isinstance(model,DDP) and micro+1 < cfg['accumulation'] else contextlib.nullcontext()
        with sync:
            with autocast(device):
                logits = model(x)
                losses = F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction='none').reshape_as(y)
            optimizer.backward(losses)
            summed += losses.detach()[masks[micro]].sum()
        observed += int(mask.sum())
    count = torch.tensor(observed,device=device,dtype=torch.int64)
    if dist.is_initialized(): dist.all_reduce(count); dist.all_reduce(summed)
    if int(count) != optimizer.batch['global_tokens']: raise RuntimeError('completed mask count differs')
    if device.type == 'cuda': torch.cuda.synchronize(device)
    start=time.perf_counter(); metrics=optimizer.step()
    if device.type == 'cuda': torch.cuda.synchronize(device)
    if optimizer.completed_steps != completed+1 or optimizer.total_loss_tokens != (completed+1)*cfg['loss_tokens_per_update']:
        raise RuntimeError('optimizer/data counters differ')
    value=float(summed/count)
    if not math.isfinite(value): raise RuntimeError('nonfinite training loss')
    return dict(metrics, loss=value, gradient_norm=metrics['preclip_norm'], optimizer_seconds=time.perf_counter()-start,
                phase='tiller', optimizer_child_lrs={role:[g['lr'] for g in child.param_groups] for role,child in zip(optimizer.roles,optimizer.children)})

@torch.no_grad()
def validation(model, path, cfg, device, rank, world):
    ids = np.memmap(path, dtype='<u4', mode='r'); context = cfg['context']; tokens = cfg['small_validation_tokens']
    if tokens % context or len(ids) <= tokens: raise ValueError('validation cache too short or unaligned')
    total = torch.zeros(2, device=device, dtype=torch.float64)
    with evaluation_state(model, device):
        for index in range(rank, tokens//context, world):
            block = torch.from_numpy(np.array(ids[index*context:(index+1)*context+1], dtype=np.int64)).to(device)[None]
            with autocast(device): value = loss(model(block[:, :-1]), block[:, 1:])
            total += torch.stack((value.double()*context, value.new_tensor(context).double()))
    if dist.is_initialized(): dist.all_reduce(total)
    if int(total[1]) != tokens: raise RuntimeError('validation token count differs')
    value = float(total[0]/total[1])
    if not math.isfinite(value): raise RuntimeError('nonfinite validation loss')
    return value


def reconcile_metrics(path, completed):
    """Preserve abandoned records, publish only the durable prefix as canonical."""
    if not path.exists(): return
    lines = path.read_text().splitlines(True)
    kept = []; abandoned = []
    for index, line in enumerate(lines):
        try: record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines)-1: raise RuntimeError('corrupt metrics before final line')
            abandoned.append(line); continue
        (kept if record['step'] <= completed else abandoned).append(line)
    if abandoned:
        archive = path.with_name(f'metrics-abandoned-{time.time_ns()}.jsonl')
        with archive.open('x') as stream:
            stream.writelines(abandoned); stream.flush(); os.fsync(stream.fileno())
        temporary = path.with_name(f'.metrics-{time.time_ns()}.partial')
        with temporary.open('x') as stream:
            stream.writelines(kept); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)


def runtime_identity(device, condition):
    return {'packages': {name:importlib.metadata.version(name) for name in ('torch','numpy')},
            'precision':'bf16_autocast_fp32_master' if device.type == 'cuda' else 'cpu_fp32_reference',
            'cuda_toolkit':torch.version.cuda}


def main():
    launch_started = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition', required=True, choices=CONDITIONS)
    p.add_argument('--data', type=Path, required=True, help='local uint32 manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--resume', help="verified manifest path, or 'latest' under this output")
    p.add_argument('--microbatch', type=int); p.add_argument('--accumulation', type=int)
    p.add_argument('--max-hours', type=float, help='optional walltime chunk, including checkpoint margin')
    p.add_argument('--checkpoint-margin-seconds', type=float, default=900)
    p.add_argument('--stop-after', type=int, help='checkpoint at this completed update; does not change training horizon')
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda', help='CPU is for small correctness tests only')
    args = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1)); rank = int(os.environ.get('RANK', 0)); local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16-capable CUDA GPU required')
    if world > 1: dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    cfg = json.loads(args.config.read_text()); cfg['world_size'] = world
    if args.microbatch is not None: cfg['microbatch'] = args.microbatch
    if args.accumulation is not None: cfg['accumulation'] = args.accumulation
    validate_config(cfg)
    if args.max_hours is not None and args.max_hours*3600 <= args.checkpoint_margin_seconds: raise ValueError('walltime is shorter than checkpoint margin')
    if args.stop_after is not None and not 1 <= args.stop_after <= cfg['updates']: raise ValueError('invalid stop boundary')
    manifest = json.loads(args.data.read_text()); paths = {}
    if manifest['tokenizer_revision'] != cfg['tokenizer_revision'] or manifest['dtype'] != '<u4': raise ValueError('tokenizer/cache identity differs')
    for split in ('train', 'validation'):
        item = manifest['splits'][split]; path = Path(item['path'])
        paths[split] = path if path.is_absolute() else args.data.parent/path
        if rank == 0:
            print(f'Verifying {split} cache...', flush=True)
            if digest(paths[split]) != item['sha256'] or paths[split].stat().st_size != 4*item['tokens']: raise RuntimeError('cache checksum/size mismatch')
    checkpoint.barrier()
    data = PackedBatches(paths['train'], world_size=world, microbatch=cfg['microbatch'], accumulation=cfg['accumulation'], context=cfg['context'])
    data.batch(cfg['updates']-1, cfg['accumulation']-1, world-1)  # Full horizon must be local before training.
    identity = dict(schema=1, condition=args.condition, config=object_digest(cfg), data=digest(args.data),
                    source=source_identity(), runtime=runtime_identity(device,args.condition), world_size=world)
    args.output.mkdir(parents=True, exist_ok=True); lock = None
    if rank == 0:
        lock = (args.output/'trainer.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = args.output/'identity.json'
        if old.exists():
            if not args.resume or json.loads(old.read_text()) != identity: raise RuntimeError('existing output requires explicit resume with matching identity')
        elif args.resume: raise RuntimeError('resume requires the original output directory')
        else:
            atomic_json(old, identity); atomic_json(args.output/'resolved_config.json', cfg)
    checkpoint.barrier()
    model, shared_hash = initial_model(cfg, args.condition); count = sum(p.numel() for p in model.parameters())
    model = model.to(device)
    if world > 1: model = DDP(model, device_ids=[local] if device.type == 'cuda' else None, broadcast_buffers=False)
    raw = model.module if isinstance(model, DDP) else model
    optimizer = build_optimizer(model, args.condition, cfg['conditions'][args.condition]['lr'], cfg['accumulation'], cfg['group_width'])
    completed = 0
    if args.resume:
        saved = json.loads((args.output/'latest.json').read_text())['manifest'] if args.resume == 'latest' else args.resume
        completed = checkpoint.load(saved, raw, optimizer, identity, lambda u: batch_hash(data, u, rank))
        if rank == 0: reconcile_metrics(args.output/'metrics.jsonl', completed)
        checkpoint.barrier()
    if rank == 0:
        atomic_json(args.output/'startup.json', dict(parameter_count=count, shared_initial_sha256=shared_hash, resumed_update=completed, identity=identity))
        print(f'{args.condition}: {count:,} parameters; starting after update {completed}; target {cfg["updates"]}', flush=True)
    start = launch_started; saved_update = completed if args.resume else -1
    def save(kind):
        nonlocal saved_update
        manifest_path = checkpoint.save(args.output/f'checkpoint-{completed:06d}-{kind}', raw, optimizer, completed, identity, batch_hash(data, completed, rank))
        if rank == 0:
            atomic_json(args.output/'latest.json', dict(manifest=str(manifest_path.resolve()), update=completed, tokens=completed*cfg['loss_tokens_per_update']))
            checkpoint.prune_periodic(args.output, identity, cfg['checkpoint_keep_recent'])
        checkpoint.barrier(); saved_update = completed
    while completed < cfg['updates']:
        tick = time.perf_counter()
        metrics = train_step(model, optimizer, data, completed, cfg, device, rank, world)
        completed += 1
        metrics.update(step=completed, loss_tokens=completed*cfg['loss_tokens_per_update'], seconds=time.perf_counter()-tick,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        if completed % cfg['validation_every'] == 0 or completed == cfg['updates']:
            metrics['validation_loss'] = validation(raw, paths['validation'], cfg, device, rank, world)
            metrics['validation_ppl'] = math.exp(metrics['validation_loss'])
        if rank == 0:
            with (args.output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metrics, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps(metrics, allow_nan=False), flush=True)
        boundary = False
        final = completed == cfg['updates']
        if boundary or final or completed % cfg['checkpoint_every'] == 0:
            save('final' if final else 'switch' if boundary else 'periodic')
        stop = torch.tensor(int((args.stop_after is not None and completed >= args.stop_after) or
                               (args.max_hours is not None and time.monotonic()-start >= args.max_hours*3600-args.checkpoint_margin_seconds)), device=device)
        if dist.is_initialized(): dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item(): break
    if completed > 0 and completed != saved_update and completed != cfg['updates']: save('continuation')
    if rank == 0:
        atomic_json(args.output/'result.json', dict(completed=completed == cfg['updates'], update=completed,
                    loss_tokens=completed*cfg['loss_tokens_per_update'], identity=identity))
    checkpoint.barrier()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
