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
ffn_dim:      2048
sequence:     256 tokens
full budget:  100M training tokens
```

`STEPS=0` makes the script compute training steps from the token budget and global batch.

## GPU Rule

Use at most 4 GPUs total. Check active jobs before launching:

```bash
squeue -u mt872
```

## Accepted Optimizer Set

The accepted launcher names for the current comparison are:

```text
adamw
muon
rational_onpolicy_balance
rational_quotient_onpolicy
rational_jacobian_onpolicy
rational_quotient_jacobian_onpolicy
rational_adaptive_metric_onpolicy
```

The accepted activation names are:

```text
silu
rlb_fused_fixed_strong_ffn
rlb_fused_fixed_strong_h2880_ffn
```

The launcher skips rational-specific optimizers on non-RLB activations. This keeps the comparison as optimizer-on-RLB, not optimizer-versus-activation. The prototype optimizers are accepted by the launcher, but the verified three-seed recommendation remains `rational_jacobian_onpolicy`.

## Full Sweep

The completed result combines these jobs:

```text
763059   AdamW, Muon, rational_onpolicy_balance, and baseline RLB rows
813929   rational_quotient_onpolicy on both RLB widths
821187   rational_jacobian_onpolicy on both RLB widths
826667   seed-1337 adaptive metric probe on both RLB widths
828678   seed-1337 quotient-Jacobian probe; h3072 completed, h2880 stopped early
```

Run the active full comparison set:

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_optimizer_empirical_ngram_full \
  STEPS=0 \
  SEEDS="1337 2024 31415" \
  OPTIMIZERS="adamw muon rational_onpolicy_balance rational_quotient_onpolicy rational_jacobian_onpolicy rational_quotient_jacobian_onpolicy rational_adaptive_metric_onpolicy" \
  ACTIVATIONS="silu rlb_fused_fixed_strong_ffn rlb_fused_fixed_strong_h2880_ffn" \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  sbatch --time=08:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4 training/run_wikitext103_optimizer_sweep.sbatch
```

## Aggregation

Aggregate the combined run:

```bash
.venv-cu128/bin/python training/aggregate_wikitext103_multiseed.py \
  --run-dir experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full \
  --out-dir experiments/results/rlb_optimizer_empirical_ngram_full \
  --baseline silu \
  --baseline-optimizer adamw \
  --classic-optimizer adamw \
  --job-id 763059+813929+821187 \
  --log-path experiments/runs/logs/ract-wt103-opt-821187.out
```

Primary fields are:

```text
optimizer
activation
seed
val_loss
val_ppl
mean_seconds_per_step
loss_gap_vs_external_baseline
loss_gap_vs_classic_same_activation
```
