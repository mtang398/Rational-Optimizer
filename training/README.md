# Training

This folder contains the WikiText-103 benchmark entrypoints.

## Files

```text
transformer_wikitext103_compare.py      model, activations, optimizer wiring, train/eval loop
run_wikitext103_optimizer_sweep.sbatch  accepted 4-GPU Slurm sweep launcher
aggregate_wikitext103_multiseed.py      JSONL aggregation into CSV/JSON/README summaries
```

## Benchmark

```text
dataset:      Salesforce/wikitext, wikitext-103-raw-v1
task:         causal language modeling
tokenizer:    GPT-2 tokenizer
model:        LLaMA-style decoder-only Transformer
size:         about 123M parameters
depth:        12 layers
width:        d_model 768
heads:        12
sequence:     256 tokens
```

## Current Best Optimizer

The active RLB-specific optimizer is:

```text
rational_matrix_policy_onpolicy
```

It now defaults to the verified same-LR MatrixPolicy-Y settings. Use it only with RLB activations. The hard controls remain:

```text
adamw + silu
adamw + rlb_fused_fixed_strong_ffn
```

## Run Current Best

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1   RUN_NAME=rlb_matrix_policy_same_lr   STEPS=3051 SEEDS=1337   OPTIMIZERS=rational_matrix_policy_onpolicy   ACTIVATIONS=rlb_fused_fixed_strong_ffn   EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100   sbatch --time=02:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4   training/run_wikitext103_optimizer_sweep.sbatch
```

## GPU Rule

Use at most 4 GPUs total. Check active jobs before launching:

```bash
squeue -u mt872
```

## Accepted Optimizers

The launcher accepts the active and historical optimizer names from `ACTIVE_OPTIMIZERS` in `transformer_wikitext103_compare.py`. Rational-specific optimizers are skipped on non-RLB activations by the launcher.

Do not use high-LR ablations as the headline result. The optimizer must first win clearly under the same LR/schedule.
