# Training harness

`transformer_lm_compare.py` provides the matched language-model training and
evaluation path. `run_lm_optimizer_sweep.sbatch` is the generic Slurm/DDP
wrapper.

Comparable rows keep fixed width, depth, heads, feed-forward width, parameter
routing, token budget, dataset source/config/slice, seed, batch order, batch
size, accumulation, sequence length, LR, minimum LR, schedule, warmup, WD,
betas, epsilon, clipping, initialization, evaluation, and diagnostics.

The active candidate is registered only through its isolated exact entrypoint.
That entrypoint installs the complete optimizer, exact LR/WD audit, DDP matrix
and coefficient synchronization checks, and frozen-cell verification before
calling the shared trainer.

GPU jobs use four RTX A6000 devices, the NVCC-built fused activation,
`RATIONAL_OPT_TORCH_FALLBACK=0`, and peer-to-peer communication enabled. The
launcher does not pin a named node; it records the assigned hardware and
topology for every result.
