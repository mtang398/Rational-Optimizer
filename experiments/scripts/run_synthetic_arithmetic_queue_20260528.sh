#!/bin/bash
#SBATCH --job-name=arith-100m
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

COMMON_EXTRA_ARGS="--dataset-name synthetic/arithmetic --dataset-config v1 --output-dir experiments/runs/synthetic_arithmetic --batch-size 16 --grad-accum 2"

run_variant() {
  local run_name="$1"
  local optimizers="$2"
  local activations="$3"
  local extra_args="${4:-}"

  echo "=== ${run_name} ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS:-3051}" \
  SEEDS="${SEEDS:-1337}" \
  OPTIMIZERS="${optimizers}" \
  ACTIVATIONS="${activations}" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-250}" \
  EVAL_BATCHES="${EVAL_BATCHES:-20}" \
  LOG_INTERVAL="${LOG_INTERVAL:-100}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  EXTRA_ARGS="${COMMON_EXTRA_ARGS} ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

run_variant \
  "synthetic_arithmetic_adamw_controls_a6000fb_ga2_20260528" \
  "adamw" \
  "silu rlb_fused_fixed_strong_ffn"

run_variant \
  "synthetic_arithmetic_matrix_policy_a6000fb_ga2_20260528" \
  "rational_matrix_policy_onpolicy" \
  "rlb_fused_fixed_strong_ffn"

run_variant \
  "synthetic_arithmetic_muon_controls_a6000fb_ga2_20260528" \
  "muon" \
  "silu rlb_fused_fixed_strong_ffn"
