import contextlib, random
import numpy as np
import torch

@contextlib.contextmanager
def evaluation_state(model,device):
    modules=list(model.modules());training=[module.training for module in modules]
    sentinel=object();observers=[getattr(module,'_rlb_optimizer_track_stats',sentinel) for module in modules]
    python_rng=random.getstate();numpy_rng=np.random.get_state()
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for module in modules:
                if hasattr(module,'_rlb_optimizer_track_stats'):module._rlb_optimizer_track_stats=False
            yield
    finally:
        for module,was_training,observer in zip(modules,training,observers):
            module.training=was_training
            if observer is not sentinel:module._rlb_optimizer_track_stats=observer
        random.setstate(python_rng);np.random.set_state(numpy_rng)
