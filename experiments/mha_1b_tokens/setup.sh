#!/usr/bin/env bash
# Run from a CUDA build host. Uses the active Python environment; does not create one.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
"${PYTHON:-python}" -m pip install -r requirements.txt
"${PYTHON:-python}" setup.py build_ext --inplace
export PYTHONPATH="$repo_root/activation:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" - <<'PY'
import torch
from rational_opt import _C
print('PyTorch:', torch.__version__, 'CUDA toolkit:', torch.version.cuda)
print('Rational extension:', _C.__file__)
print('Muon available:', hasattr(torch.optim, 'Muon'))
if not hasattr(torch.optim, 'Muon'): raise RuntimeError('pinned PyTorch Muon required')
PY
