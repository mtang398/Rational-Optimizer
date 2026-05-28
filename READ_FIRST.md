# Read First

This repo is for designing an RLB-specific optimizer. Do not claim wins from a different global LR schedule.

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

## Current Best

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
mechanism:   early RLB-matrix Muon switch, then MatrixPolicy AdamW
schedule:    same lr=3e-4, min_lr=3e-5, warmup=200, cosine
result:      3.476232 loss, 32.34 PPL at step 3051, seed 1337
```

Important same-LR controls:

```text
SiLU+AdamW                  3.621982 loss, 37.41 PPL
RLB+AdamW                   3.617501 loss, 37.24 PPL
SiLU+AdamW beta2=0.999      3.549346 loss, 34.79 PPL
RLB+AdamW beta2=0.999       3.550018 loss, 34.81 PPL
RLB Smooth-MatrixPolicy     3.493210 loss, 32.89 PPL
RLB MatrixPolicy-Muon       3.476232 loss, 32.34 PPL
```

The current best clears 2-3 PPL versus beta2-tuned controls, but does not yet clear the requested 0.2-0.3 final loss gap. Do not run or report LR ablations as the main story until the same-LR loss gap is much larger.

## Artifact Policy

Commit compact current summaries and plots under:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Do not commit raw local probe folders under:

```text
experiments/runs/wikitext103/
```
