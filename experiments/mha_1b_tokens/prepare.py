"""CPU preparation: import verified caches, or tokenize deterministic document splits."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
from .data import document_id, split_for, overlaps, word_ngrams
from .io import atomic_json, digest


def local_documents(paths):
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                yield json.loads(line)['text']


def import_manifest(source, output, cfg):
    """Reference immutable existing caches without copying billions of tokens."""
    old = json.loads(source.read_text()); splits = {}
    if old.get('tokenizer_revision', old.get('tokenizer', {}).get('revision')) != cfg['tokenizer_revision']:
        raise ValueError('tokenizer revision differs')
    for split in ('train', 'validation'):
        item = old['splits'][split]; path = Path(item['path'])
        path = path if path.is_absolute() else source.parent/path
        if digest(path) != item['sha256'] or path.stat().st_size % 4: raise RuntimeError('cache checksum or uint32 size differs')
        tokens = path.stat().st_size//4
        if item.get('tokens', tokens) != tokens: raise RuntimeError('cache count differs')
        # Chunked validation catches uint16 files and invalid token IDs.
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024), b''):
                if np.frombuffer(chunk, dtype='<u4').max(initial=0) >= cfg['model']['vocab_size']:
                    raise RuntimeError('out-of-vocabulary uint32 token')
        splits[split] = dict(path=str(path.resolve()), tokens=tokens, sha256=item['sha256'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'manifest.json', dict(dtype='<u4', tokenizer_revision=cfg['tokenizer_revision'], splits=splits,
                imported_manifest_sha256=digest(source), provenance=old,
                note='Original selection/deduplication/overlap evidence remains in imported provenance.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--import-manifest', type=Path)
    mode.add_argument('--jsonl', type=Path, nargs='+', help='ordered UTF-8 JSONL documents with a text field')
    mode.add_argument('--fineweb', action='store_true', help='pinned FineWeb-Edu sample-100BT, CPU streaming during preparation only')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, help='already staged local pinned tokenizer; otherwise fetch tokenizer files only')
    p.add_argument('--exclude-jsonl', type=Path, nargs='+', help='evaluation texts; remove training documents sharing a normalized 13-word span')
    p.add_argument('--train-tokens', type=int, default=1_000_079_361)
    p.add_argument('--validation-tokens', type=int, default=131_073)
    args = p.parse_args(); cfg = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.import_manifest:
        import_manifest(args.import_manifest, args.output, cfg); return
    if min(args.train_tokens, args.validation_tokens) < 2: raise ValueError('cache must include input and label tokens')
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        marker = json.loads((args.tokenizer/'pinned_revision.json').read_text())
        if marker['revision'] != cfg['tokenizer_revision']: raise ValueError('tokenizer pin differs')
        for name, sha in marker['files'].items():
            if digest(args.tokenizer/name) != sha: raise ValueError('tokenizer file changed')
    else:
        from huggingface_hub import snapshot_download
        location = snapshot_download(cfg['tokenizer_id'], revision=cfg['tokenizer_revision'],
                                     allow_patterns=['config.json','tokenizer*','vocab.json','merges.txt','special_tokens_map.json'])
        tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True)
        args.tokenizer = args.output/'tokenizer'; tokenizer.save_pretrained(args.tokenizer)
        atomic_json(args.tokenizer/'pinned_revision.json', dict(revision=cfg['tokenizer_revision'], files={p.name:digest(p) for p in args.tokenizer.iterdir() if p.is_file()}))
    if tokenizer.eos_token_id is None: raise ValueError('EOS token required')
    excluded = set()
    if args.exclude_jsonl:
        for text in local_documents(args.exclude_jsonl): excluded.update(word_ngrams(text))
    if args.fineweb:
        from datasets import load_dataset
        rows = load_dataset(cfg['dataset_id'], name=cfg['dataset_config'], revision=cfg['dataset_revision'], split='train', streaming=True)
        documents = (row['text'] for row in rows)
    else: documents = local_documents(args.jsonl)
    targets = dict(train=args.train_tokens, validation=args.validation_tokens)
    counts = Counter(); stats = Counter(); streams = {}; db = sqlite3.connect(args.output/'document_ids.sqlite')
    db.execute('CREATE TABLE seen (identity TEXT PRIMARY KEY)')
    selection = (args.output/'selected_documents.jsonl').open('x')
    for split in targets: streams[split] = (args.output/f'{split}.u32.partial').open('xb')
    try:
        for text in documents:
            stats['documents_seen'] += 1
            ident = document_id(text)
            if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (ident,)).rowcount == 0:
                stats['exact_duplicates_removed'] += 1; continue
            split = split_for(text)
            if split not in targets or counts[split] >= targets[split]: continue
            if split == 'train' and excluded and overlaps(text, excluded):
                stats['overlap_documents_removed'] += 1; continue
            ids = tokenizer.encode(text, add_special_tokens=False)+[tokenizer.eos_token_id]
            remaining = targets[split]-counts[split]; used = ids[:remaining]
            array = np.asarray(used, dtype='<u4')
            if used and (min(used) < 0 or max(used) >= cfg['model']['vocab_size'] or array.tolist() != used): raise RuntimeError('uint32 token round trip failed')
            array.tofile(streams[split]); counts[split] += len(used)
            selection.write(json.dumps(dict(id=ident, split=split, tokens=len(used), original_tokens=len(ids)))+'\n')
            if stats['documents_seen'] % 10000 == 0:
                db.commit(); print(dict(stats, tokens=dict(counts)), flush=True)
            if all(counts[k] == v for k,v in targets.items()): break
        if any(counts[k] != v for k,v in targets.items()): raise RuntimeError(f'input exhausted: {dict(counts)}, wanted {targets}')
        db.commit(); selection.flush(); os.fsync(selection.fileno())
        splits = {}
        for split, stream in streams.items():
            stream.flush(); os.fsync(stream.fileno()); stream.close()
            path = args.output/f'{split}.u32'; (args.output/f'{split}.u32.partial').rename(path)
            splits[split] = dict(path=path.name, tokens=counts[split], sha256=digest(path))
        atomic_json(args.output/'manifest.json', dict(dtype='<u4', splits=splits, tokenizer_revision=cfg['tokenizer_revision'],
                    source=dict(dataset=cfg['dataset_id'], config=cfg['dataset_config'], revision=cfg['dataset_revision']) if args.fineweb else {str(p):digest(p) for p in args.jsonl},
                    selected_documents_sha256=digest(args.output/'selected_documents.jsonl'), stats=dict(stats),
                    processing=dict(split='normalized-content SHA256 buckets; train/validation/test disjoint',
                        duplicates='normalized exact removal; near-duplicates NOT filtered by this starter',
                        overlap='normalized 13-word spans' if excluded else 'not screened; provide --exclude-jsonl before benchmark claims',
                        exclude_inputs={str(p):digest(p) for p in (args.exclude_jsonl or [])},
                        packing='EOS appended; final document truncated to cache target; no padding; every target predicted; cross-document attention')))
        print(f'Ready: {args.output / "manifest.json"}', flush=True)
    finally:
        for stream in streams.values(): stream.close()
        selection.close(); db.close()


if __name__ == '__main__': main()
