# Read First

This file is the operating rule for the repo. Read it before running or modifying experiments.

## GPU Limit

Use at most 4 GPUs total.

The accepted Slurm shape is one node with four training processes:

```text
--gres=gpu:nvidia_rtx_6000_ada_generation:4
--nproc_per_node=4
```

Check active jobs before launching:

```bash
squeue -u mt872
```

## Current Task

The active research task is optimizer design for the no-GLU Rational Local Basis FFN, abbreviated RLB.

The optimizer is applied to RLB. It is not an activation comparison by itself. The fair comparison grid is:

```text
SiLU/SwiGLU + AdamW
SiLU/SwiGLU + Muon
RLB + AdamW
RLB + Muon
RLB + rational_onpolicy_balance
RLB + rational_quotient_onpolicy
RLB + rational_jacobian_onpolicy
RLB + rational_quotient_jacobian_onpolicy   prototype
RLB + rational_adaptive_metric_onpolicy     prototype
```

Rational-specific optimizers are used only with RLB activations. Standard optimizers are AdamW and Muon.

## Current Evidence

Completed three-seed full run:

```text
run:    rlb_optimizer_empirical_ngram_full
jobs:   763059, 813929, 821187
seeds:  1337, 2024, 31415
budget: WikiText-103 100M-token causal LM
model:  12-layer LLaMA-style decoder, d_model 768, 12 heads
```

Key aggregate losses:

```text
AdamW + SiLU/SwiGLU                         3.610129  PPL 36.973  sec/step 0.188997
AdamW + RLB h3072                           3.606629  PPL 36.845  sec/step 0.205268
RLB h3072 + rational_onpolicy_balance       3.606226  PPL 36.831  sec/step 0.209027
RLB h3072 + rational_quotient_onpolicy      3.606664  PPL 36.847  sec/step 0.205176
RLB h3072 + rational_jacobian_onpolicy      3.605394  PPL 36.800  sec/step 0.204885
```

The current best measured optimizer row is:

```text
rational_jacobian_onpolicy + rlb_fused_fixed_strong_ffn
```

Its mean loss gap is `-0.004736` versus AdamW + SiLU/SwiGLU and `-0.001236` versus AdamW on the same RLB activation. Its mean step time is essentially the same as AdamW on the same RLB activation.

Prototype probes on 2026-05-27 added `rational_adaptive_metric_onpolicy` and `rational_quotient_jacobian_onpolicy`. They beat AdamW/SILU in seed-1337 probes but did not beat the existing rational-specific incumbents, so do not replace the current recommendation without a stronger multi-seed result.

## Repo Layout

```text
activation/        rational activation implementation and CUDA extension
training/          WikiText-103 model training, Slurm launcher, aggregation
optimizer_design/  RLB-specific optimizer components and math notes
experiments/       cache, active run logs, JSONL outputs, aggregate results
```

## Artifact Policy

Keep the active dataset cache and the active full-sweep run directory:

```text
experiments/cache/
experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full/
experiments/results/rlb_optimizer_empirical_ngram_full/
```

Remove repo-local generated artifacts when they appear:

```text
build/
training/__pycache__/
optimizer_design/__pycache__/
activation/rational_opt/__pycache__/
```
