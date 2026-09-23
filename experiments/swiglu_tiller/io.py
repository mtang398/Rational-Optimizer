import hashlib, json, os, tempfile
from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def object_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for directory in ('experiments/swiglu_tiller',):
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.cu','.cpp','.h','.cuh') and '__pycache__' not in p.parts and 'results' not in p.parts and 'scalingopt' not in p.parts)
    return object_digest({str(p.relative_to(root)):digest(p) for p in sorted(paths)})
