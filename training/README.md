# Training

This directory contains the LM benchmark harness, dataset streaming support, synthetic generators, and optimizer wiring. Its job is to enforce fair comparisons between activation/optimizer pairs.

## Fair Comparison Contract

Rows are comparable only when these are fixed:

```text
model width, depth, head count, and FFN parameter budget
token budget
seed set
batch size and gradient accumulation
sequence length
base LR schedule and warmup
weight decay
dataset, dataset config, and dataset slice
evaluation cadence
```

Changing the global LR schedule is an LR ablation, not evidence for an RLB-specific optimizer. LR sweeps are useful after a same-LR advantage exists.

## Required Control Rows

Every serious run should include:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
SiLU/SwiGLU + Muon
RLB + Muon
RLB + rational_matrix_policy_onpolicy
```

The current public tables use `silu` and `rlb_fused_fixed_strong_ffn` as the two activation rows.

## MatrixPolicy Wiring

For `--optimizer rational_matrix_policy_onpolicy`, the harness collects each RLB layer into optimizer groups:

```text
A_l = W_in,l  -> matrix_role = in
B_l = W_out,l -> matrix_role = out
R_l           -> rational coefficient parameters
```

Current best group-stat flags:

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

## Real-Corpus Protocol

The main screen streams Hugging Face datasets and builds bounded token caches:

```text
--dataset-streaming
--dataset-text-column text
--train-split train
--validation-split train
--validation-skip-documents 0
--validation-skip-tokens 110000000
--max-train-tokens 100000000
--max-val-tokens 4000000
```

`validation-skip-tokens` is the preferred way to make a disjoint validation token cache when the dataset has only a `train` split. Cache filenames include split, stream/map mode, text column, document skip, token skip, tokenizer, and token budget so corpus slices do not collide.

Current completed task keys:

```text
fineweb_edu -> HuggingFaceFW/fineweb-edu, sample-10BT
fineweb     -> HuggingFaceFW/fineweb, sample-10BT
```

Available but not yet paper-grade in this environment:

```text
dclm         -> mlfoundations/dclm-baseline-1.0; needs zstd support
Dolma sample -> allenai/dolma, v1_6-sample; needs loader/conversion validation
```

## Launcher

Main launcher:

```bash
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

Useful environment overrides:

```bash
REAL_LM_TASKS="fineweb fineweb_edu" \
SEEDS="2027" \
RUN_SUFFIX="20260531_seed2027_100m" \
OUTPUT_ROOT="experiments/runs/real_lm_multiseed_20260531" \
sbatch experiments/scripts/run_real_lm_screen_20260530.sh
```

The launcher skips completed activation JSONL files and archives only incomplete activation JSONL files on rerun/requeue. This prevents a requeue from discarding already completed rows.

## Logging And Analysis Standard

Optimizer-discrimination runs should log dense curves:

```text
training: step 1, then at least every 10 steps
validation: step 1, then at least every 50 steps for real LM screens
final: always include the final step
```

Report:

```text
final validation loss and PPL
validation loss AUC through early/mid/full horizons
training loss curves from step 1
wall-clock seconds per step and tokens per second
nonfinite/divergent rows, without hiding them
```

## Runtime Rules

GPU jobs must respect these limits:

```text
--gres=gpu:nvidia_rtx_a6000:4
max 4 A6000 GPUs per job
max 8 A6000 GPUs active total
repository size below 200G
```

Standard runtime environment:

```text
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```

For long Slurm runs, use requeue support:

```text
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```
