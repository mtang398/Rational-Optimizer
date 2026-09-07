# CUDA extension

`rational_ext.cpp` registers the PyTorch operators and
`rational_cuda_kernel.cu` implements the fused GRAIN forward pass, backward
pass, affine statistics, and restricted-basis operations.

From the repository root:

```bash
.venv/bin/python setup.py build_ext --inplace
```

The resulting module is `activation/rational_opt/_C*.so`. The build uses the
same source for training and for the CUDA smoke test in
`experiments/protocol/smoke_cuda_extension.py`.
