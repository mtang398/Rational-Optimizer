# Training

This folder contains the language-model harness, synthetic task generators, and comparison plumbing used by the optimizer experiments.

## Fair Comparison Contract

Every serious comparison should keep these fixed across rows:

```text
model size
token budget
seed
batch shape
sequence length
base LR schedule
evaluation cadence
dataset and dataset config
```

The required control set is:

```text
SiLU/SwiGLU+AdamW
RLB+AdamW
SiLU/SwiGLU+Muon
RLB+Muon
RLB MatrixPolicy variants
```

Learning-rate scheduler changes are not optimizer wins for this project. LR ablations only become useful after a rational-specific optimizer shows a large same-LR advantage.

## RLB Training Hooks

The harness supports RLB-specific optimizer tests:

| hook | purpose |
| --- | --- |
| `rational_matrix_policy_onpolicy` | Role/depth-aware optimizer for RLB matrices. |
| group-stat policy flags | Live activity and pressure signals for per-group scaling. |
| `--rlb-init-gauge-log-scale` | Equivalent-function positive gauge stress at initialization. |
| `--rlb-init-gauge-seed` | Reproducible gauge sampling. |
| `log_interval` / `eval_interval` | Dense training and validation curves from step 1. |

The gauge flags apply only to RLB FFNs. A local parameter-level validation confirmed that the gauge transform preserves per-group `W_out @ W_in` products up to `1.4e-9`; full forward validation on CPU is blocked by the rational CUDA extension path, so the running GPU gauge stress is the relevant end-to-end test.

## Task Interpretation

| task | current role | caveat |
| --- | --- | --- |
| WikiText-103 | Main small-LM benchmark. | Current best gap is real but modest. |
| synthetic/code | Tests program-like local structure. | RLB drops faster early, but final loss saturates. |
| synthetic/symbolic | Tests rewrite/parity/bracket/copy patterns. | Too easy for a final optimizer claim. |
| synthetic/reasoning_mix | Tests mixed arithmetic/code/symbolic patterns. | Useful curve signal, but final PPL is compressed. |

For saturated synthetic tasks, curve shape matters more than the final row. Training loss must be plotted too, because validation-only plots can hide optimizer phase behavior.

## Logging Standard

Fresh optimizer-discrimination runs should log:

```text
training:   step 1, then at least every 10 steps
validation: step 1, then at least every 25 steps
final:      always include final step
```

Plots should start at step 1. Starting at step 1000 hides the part of the curve where the rational optimizer advantage is most likely to appear.

## Benchmark Design

Good short tasks should leave headroom. A task is not a meaningful optimizer benchmark if the strongest controls reach loss `<0.1` by the target budget. In that regime, PPL differences are too compressed and final-loss rankings can become noise.

Preferred next task families:

| task | target behavior |
| --- | --- |
| `synthetic/rule_chain_hard` | Multi-hop symbolic composition with distractors and held-out symbols. |
| `synthetic/key_value_recall` | In-context binding and delayed retrieval under noise. |
| `synthetic/carry_arithmetic` | Multi-digit arithmetic where carries create sharp decision boundaries. |
| `synthetic/stack_brackets` | Deeper typed-stack state tracking. |
| `synthetic/noisy_copy_transform` | Span copy/reverse/map with variable noise and length. |

These tasks should be used after the gauge-stress result clarifies whether MatrixPolicy is exploiting RLB geometry or merely winning on a narrow curve regime.

## Runtime Rule

Use A6000 GPUs only and do not exceed 8 active A6000s total.

```text
--gres=gpu:nvidia_rtx_a6000:4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```
