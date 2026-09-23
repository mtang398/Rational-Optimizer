"""Document splitting and lossless uint32 cache/packing primitives."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
from .io import atomic_json, digest

NORMALIZATION='NFKC+casefold+unicode-whitespace-v1'

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def document_id(text):
    return hashlib.sha256(normalized(text).encode('utf-8')).hexdigest()

def split_for(text):
    # Splits are assigned before document selection, packing or token counting.
    bucket=int(document_id(text)[:16],16)%10000
    return 'validation' if bucket<20 else 'test' if bucket<70 else 'train'

def word_ngrams(text,n=13):
    words=re.findall(r'\w+',normalized(text))
    return {' '.join(words[i:i+n]) for i in range(max(0,len(words)-n+1))}

def overlaps(text,benchmark_ngrams):
    return bool(word_ngrams(text)&benchmark_ngrams)

def write_cache(path,ids,*,vocab_size):
    path=Path(path)
    ids=np.asarray(ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=vocab_size):
        raise ValueError('invalid integer token IDs')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('xb') as stream:
        ids.astype('<u4').tofile(stream);stream.flush()
        import os
        os.fsync(stream.fileno())
    reread=np.fromfile(temporary,dtype='<u4')
    if not np.array_equal(ids,reread): raise RuntimeError('uint32 round trip failed')
    if path.exists(): raise FileExistsError('immutable cache already exists')
    temporary.rename(path)
    manifest={'dtype':'<u4','tokens':len(ids),'sha256':digest(path),'vocab_size':vocab_size}
    atomic_json(path.with_suffix(path.suffix+'.json'),manifest)
    return manifest

class PackedBatches:
    """Every update owns consecutive disjoint targets; one-token input overlap only.

Each sequence reads context+1 tokens, producing context input/target pairs.
Target positions are [start+1,start+context], with start=sequence_index*context.
There is no dropped label at a sequence boundary, padding, or EOS masking.
EOS is a normal predicted token. Attention can span documents within a sequence.
"""
    def __init__(self,path,*,world_size=4,microbatch=8,accumulation=4,context=2048):
        self.tokens=np.memmap(path,dtype='<u4',mode='r')
        self.world=world_size;self.micro=microbatch;self.accum=accumulation;self.context=context
        if min(world_size,microbatch,accumulation,context)<1: raise ValueError('invalid batch geometry')
        self.loss_tokens=world_size*microbatch*accumulation*context
    def batch(self,update,microstep,rank):
        if update<0 or not 0<=microstep<self.accum or not 0<=rank<self.world: raise ValueError('invalid data position')
        first=((update*self.accum+microstep)*self.world+rank)*self.micro
        starts=(first+np.arange(self.micro,dtype=np.int64))*self.context
        if starts[-1]+self.context>=len(self.tokens): raise IndexError('cache exhausted; replay forbidden')
        indices=starts[:,None]+np.arange(self.context+1,dtype=np.int64)
        block=np.asarray(self.tokens[indices],dtype=np.int64)
        return block[:,:-1].copy(),block[:,1:].copy(),np.ones((self.micro,self.context),dtype=bool)

def whole_updates(target=12_000_000_000,batch=262144):
    updates=(target+batch//2)//batch
    return updates,updates*batch
