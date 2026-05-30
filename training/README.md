# Training

This folder contains the LM benchmark harness, dataset streaming support, synthetic generators, and optimizer wiring. Its purpose is to enforce fair comparisons between activation/optimizer pairs.

## Fair Comparison Contract

Rows are comparable only when the following are fixed:

```text
model width, depth, head count, and FFN parameter budget
token budget
seed
batch size and gradient accumulation
sequence length
base LR schedule and warmup
weight decay
dataset, dataset config, and dataset slice
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

Additional rational optimizers are ablations. They should be interpreted by the mechanism they test, not treated as baselines.

## MatrixPolicy Wiring

For `--optimizer rational_matrix_policy_onpolicy`, the harness collects each RLB layer into optimizer groups:

```text
A_l = W_in,l  -> matrix_role = in
B_l = W_out,l -> matrix_role = out
R_l           -> rational coefficient parameters
```

The current real-corpus best row applies:

```text
ordinary Transformer parameters -> AdamW
rational coefficients           -> coefficient updates in the on-policy wrapper
RLB matrices                    -> RationalMatrixPolicyOptimizer
RLB gauge class                 -> exact post-step gauge rebalance
```

The exact real-corpus group-stat flags are:

```text
--rational-matrix-policy-backbone-optimizer adamw
--rational-matrix-policy-group-gain-strength 0.20
--rational-matrix-policy-group-pressure-strength 0.10
--rational-matrix-policy-group-activity-damping 0.20
--rational-matrix-policy-group-start 0.02
--rational-matrix-policy-group-end 0.30
--rational-matrix-policy-group-min-scale 0.75
--rational-matrix-policy-group-max-scale 1.35
```

## Logging And Analysis Standard

Optimizer-discrimination runs must log dense curves:

```text
training:   step 1, then at least every 10 steps
validation: step 1, then at least every 50 steps for real LM screens
final:      always include final step
```

Plots must start at step 1. Late-only plots hide the early phase where the rational matrix policy differs most from generic AdamW or Muon.

Use these metrics before final-loss tables:

```text
step-matched training loss and validation loss
mean validation loss AUC through early and mid horizons
mean training loss AUC through early and mid horizons
time to fixed validation thresholds when thresholds are meaningful
final loss/PPL after checking whether the task is saturated
```

## Real-Corpus Streaming

The harness supports large Hugging Face corpora without downloading full datasets. Use these flags for pretraining-like screens:

```text
--dataset-streaming
--dataset-text-column text
--train-split train
--validation-split train
--validation-skip-documents N
--validation-skip-tokens 110000000
--max-train-tokens 100000000
--max-val-tokens 4000000
```

`validation-skip-tokens` is the preferred way to make a disjoint validation token cache when the dataset has only a `train` split. Cache filenames include the split, stream/map mode, text column, document skip, token skip, tokenizer, and token budget so different corpus slices do not collide.

The May 30 launcher currently supports these task keys:

```text
fineweb_edu -> HuggingFaceFW/fineweb-edu, sample-10BT
fineweb     -> HuggingFaceFW/fineweb, sample-10BT
dclm        -> mlfoundations/dclm-baseline-1.0
dolma_sample -> allenai/dolma, v1_6-sample
```

FineWeb and FineWeb-Edu are completed and summarized. DCLM and Dolma remain useful next targets, but they need environment work in this setup: DCLM needs zstd support and Dolma needs a compatible loader or converted local slice.

## Synthetic Tasks

Synthetic tasks are debugging tools unless they leave real headroom. If all strong rows reach a compressed loss floor, final PPL is not a meaningful optimizer claim. The earlier saturated synthetic result packages were removed from the tracked public evidence for this reason.

Future synthetic tasks should be harder and mechanism-targeted:

| task family | mechanism |
| --- | --- |
| rational-teacher LM | tests whether RLB optimizer fits a hidden rational transition. |
| length/composition extrapolation | tests learned algorithmic structure beyond training lengths. |
| phase-mix task | tests whether early speed survives a delayed hard subtask. |
| gauge-stressed task with diagnostics | tests the exact RLB symmetry with measured gauge drift. |

## Runtime Rule

GPU jobs should request A6000s only and keep total active allocation at or below 8 A6000s.

```text
--gres=gpu:nvidia_rtx_a6000:4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```

For long Slurm runs, use requeue support:

```text
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```
