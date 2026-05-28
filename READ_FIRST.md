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

Only run LR ablations after the same-LR optimizer has a much larger gap. The current same-LR winner has a good PPL gap but only about `0.07` loss gap, so LR ablations are diagnostic only and not the main story.

## Current Best

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
schedule:    same lr=3e-4, min_lr=3e-5 as the controls
result:      3.548665 loss, 34.77 PPL at step 3051, seed 1337
```

Control results under the same full schedule:

```text
RLB + AdamW                 3.617501 loss, 37.24 PPL
RLB + Jacobian              3.614862 loss, 37.15 PPL
SiLU/SwiGLU + AdamW         3.621982 loss, 37.41 PPL
```

The optimizer works by applying a strong layer/side-specific policy to RLB `W_in` and `W_out`, while leaving the global LR and the rest of AdamW unchanged.

## Artifact Policy

Commit compact summaries and plots under:

```text
experiments/results/
```

Do not commit raw local probe folders under `experiments/runs/wikitext103/`.
