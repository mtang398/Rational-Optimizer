# Training

This folder contains the LM benchmark harness, synthetic task generators, and optimizer wiring. Its purpose is to enforce fair comparisons between activation/optimizer pairs.

## Fair Comparison Contract

Rows are comparable only when the following are fixed:

```text
model width/depth/head count
FFN parameter budget
token budget
seed
batch size and gradient accumulation
sequence length
base LR schedule and warmup
weight decay
dataset and dataset config
evaluation cadence
```

Changing the global LR schedule is an LR ablation, not evidence for an RLB-specific optimizer. LR sweeps are useful only after a large same-LR gap exists.

## Required Rows

Every serious run should include:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
SiLU/SwiGLU + Muon
RLB + Muon
RLB + rational_matrix_policy_onpolicy
```

Additional rational optimizers are ablations. They should be interpreted by what mechanism they test, not treated as baselines.

## MatrixPolicy Wiring

For `--optimizer rational_matrix_policy_onpolicy`, the harness collects each RLB layer into optimizer groups:

```text
A_l = W_in,l  -> matrix_role = in
B_l = W_out,l -> matrix_role = out
R_l           -> rational coefficient parameters
```

The optimizer then applies:

```text
ordinary Transformer parameters -> AdamW or optional backbone Muon
rational coefficients           -> coefficient optimizer when enabled
RLB matrices                    -> RationalMatrixPolicyOptimizer
RLB gauge class                 -> on-policy gauge rebalance wrapper
```

The initialization gauge-stress flags are:

```text
--rlb-init-gauge-log-scale
--rlb-init-gauge-seed
```

They sample positive scales `a_g` and apply the function-preserving transform

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

before training. This creates equivalent initial functions with different matrix conditioning.

## Logging Standard

Optimizer-discrimination runs must log dense curves:

```text
training:   step 1, then at least every 10 steps
validation: step 1, then at least every 25 steps
final:      always include final step
```

Plots should start at step 1. Late-only plots hide the early phase where the rational matrix policy is most likely to differ from generic AdamW or Muon.

## Task Standard

Synthetic tasks are useful only if they leave headroom. If the strongest controls reach loss `<0.1`, final PPL is too compressed to support a large optimizer claim. Such tasks can still test curve speed or implementation stability.

Preferred short-task families should stress mechanisms rather than text formatting:

| task family | mechanism |
| --- | --- |
| rule-chain composition | multi-step symbolic function composition. |
| key-value recall | in-context binding and delayed retrieval. |
| carry arithmetic | sharp local decision boundaries from carries. |
| stack brackets | nonlinear state tracking. |
| noisy copy/transform | robust sequence transformation under distractors. |

The gauge-stress benchmark should be evaluated before adding many more tasks, because it tests the actual RLB symmetry directly.

## Runtime Rule

GPU jobs should request A6000s only and keep total active allocation at or below 8 A6000s.

```text
--gres=gpu:nvidia_rtx_a6000:4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```
