#!/bin/bash
#SBATCH --job-name=muon-ctrl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

RUN_NAME="${RUN_NAME:-muon_controls_full_a6000fb_ga2_20260528}" \
STEPS="${STEPS:-3051}" \
SEEDS="${SEEDS:-1337}" \
OPTIMIZERS="${OPTIMIZERS:-muon}" \
ACTIVATIONS="${ACTIVATIONS:-silu rlb_fused_fixed_strong_ffn}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
LOG_INTERVAL="${LOG_INTERVAL:-100}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
EXTRA_ARGS="${EXTRA_ARGS:---batch-size 16 --grad-accum 2}" \
bash training/run_wikitext103_optimizer_sweep.sbatch
