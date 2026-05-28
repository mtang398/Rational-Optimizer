# Read First

This is the operating rule for the repo. Read it before running experiments.

## GPU Limit

Use at most 4 GPUs total.

```text
--gres=gpu:nvidia_rtx_6000_ada_generation:4
--nproc_per_node=4
```

Check active jobs before launching:

```bash
squeue -u mt872
```

## Research Rule

The project is designing an RLB-specific optimizer. Do not claim an optimizer win from a different global learning-rate schedule.

Hard controls:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
```

Also report beta2-tuned AdamW controls when the optimizer uses smooth moments. The current best keeps the same LR schedule but uses smoother optimizer state inside the MatrixPolicy branch.

## Current Best

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
schedule:    same lr=3e-4, min_lr=3e-5, warmup=200, cosine decay
result:      3.493210 loss, 32.89 PPL at step 3051, seed 1337
```

Important controls under the same LR schedule:

```text
SiLU/SwiGLU + AdamW          3.621982 loss, 37.41 PPL
RLB + AdamW                  3.617501 loss, 37.24 PPL
SiLU/SwiGLU + AdamW b2=.999  3.549346 loss, 34.79 PPL
RLB + AdamW b2=.999          3.550018 loss, 34.81 PPL
```

The current optimizer beats the original controls by more than 4 PPL and about 0.125 loss. It beats beta2-tuned controls by about 1.9 PPL and 0.056 loss. This is good progress but still below the desired 0.2-0.3 tuned-control loss gap.

## Artifact Policy

Commit compact summaries and plots under:

```text
experiments/results/
```

Do not commit raw local probe folders under `experiments/runs/wikitext103/`.
