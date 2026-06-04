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

The completed preliminary screen includes:

```text
SiLU/SwiGLU + AdamW
RLB + AdamW
SiLU/SwiGLU + Muon
RLB + Muon
RLB + rational_matrix_policy_onpolicy
```

The current public tables use `silu` and `rlb_fused_fixed_strong_ffn` as the two activation rows.

The harness now exposes these additional broad baselines for matched paper runs:

```text
--optimizer soap_adamw              # SOAP/Shampoo-style AdamW eigenbasis baseline
--optimizer lion
--optimizer ademamix
--optimizer schedule_free_adamw     # schedule-free-style AdamW baseline
--optimizer adafactor_came          # Adafactor/CAME-style factored adaptive baseline
```

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

Additional task keys for accepted-anchor and modern-corpus runs:

```text
c4_en        -> allenai/c4, en; accepted optimizer anchor for C4/C4-EN style validation
openwebtext  -> Skylion007/openwebtext; Sophia-style GPT-2 pretraining anchor
pile         -> EleutherAI/pile; Sophia-style large-corpus anchor, smoke first
dclm         -> mlfoundations/dclm-baseline-1.0; modern DataComp/DCLM corpus, smoke first
dolma_sample -> allenai/dolma, v1_6-sample; transfer/modern corpus, smoke first
```

## Launcher

Main paper runs use manifest rows:

```bash
python3 experiments/scripts/build_iclr26_main_manifest.py \
  --output experiments/manifests/iclr26_main_manifest.csv \
  --print-summary

CONFIRM_ICLR26_MANIFEST=1 \
MANIFEST=experiments/manifests/iclr26_main_manifest.csv \
ROW_START=0 \
ROW_LIMIT=1 \
sbatch experiments/scripts/run_iclr26_manifest_job.sh
```

The manifest launcher skips completed JSONL files and archives incomplete JSONL files on rerun/requeue. This prevents a requeue from discarding already completed rows.

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

The training loop now also emits paper-mechanism telemetry:

```text
grad_global_norm_before_clip
grad_clip_triggered
forward_backward_seconds
optimizer_step_seconds
cuda_max_memory_allocated/reserved
probe_logit_rms and probe KL/delta metrics
RLB output/derivative/atom/gauge/denominator metrics
MatrixPolicy role update, weight, LR-scale, Muon-mix, pressure, activity, and group-scale metrics
SVD entropy for attention and RLB matrices at configured intervals
```

Before launching new headline-benchmark jobs, run manifest preflight rows and verify these fields appear in JSONL on rank 0. Also verify at least one smoke row for each optimizer name used by the benchmark because CPU checks do not exercise CUDA/DDP interaction.

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
RATIONAL_OPT_TORCH_FALLBACK=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```

The CUDA venv used for paper jobs must include `ninja` for extension builds and `zstandard` for DCLM zstd shards.

For long Slurm runs, use requeue support:

```text
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
```
