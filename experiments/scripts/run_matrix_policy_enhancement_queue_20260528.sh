#!/bin/bash
#SBATCH --job-name=rlb-enhance
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

run_variant() {
  local run_name="$1"
  local extra_args="$2"

  echo "=== ${run_name} ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS:-1250}" \
  SEEDS="${SEEDS:-1337}" \
  OPTIMIZERS="rational_matrix_policy_onpolicy" \
  ACTIVATIONS="rlb_fused_fixed_strong_ffn" \
  EVAL_INTERVAL="${EVAL_INTERVAL:-250}" \
  EVAL_BATCHES="${EVAL_BATCHES:-20}" \
  LOG_INTERVAL="${LOG_INTERVAL:-100}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
  EXTRA_ARGS="--batch-size 16 --grad-accum 2 ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

run_variant \
  "rlb_matrix_policy_beta2tail995_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-adam-beta2-final 0.995 --rational-matrix-policy-adam-beta2-decay-start 0.42 --rational-matrix-policy-adam-beta2-decay-end 0.78"

run_variant \
  "rlb_matrix_policy_beta2tail990_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-adam-beta2-final 0.990 --rational-matrix-policy-adam-beta2-decay-start 0.42 --rational-matrix-policy-adam-beta2-decay-end 0.78"

run_variant \
  "rlb_matrix_policy_muon_transport020_a6000fb_ga2_probe_20260528" \
  "--rational-transport-strength 0.20 --rational-transport-final-strength 0.0 --rational-transport-start 0.04 --rational-transport-end 0.32 --rational-transport-decay-start 0.38 --rational-transport-decay-end 0.56 --rational-transport-max-log-step 0.010 --rational-transport-derivative-weight 0.60 --rational-transport-headroom 0.90"

run_variant \
  "rlb_matrix_policy_muon_transport015_beta2tail995_a6000fb_ga2_probe_20260528" \
  "--rational-transport-strength 0.15 --rational-transport-final-strength 0.0 --rational-transport-start 0.04 --rational-transport-end 0.32 --rational-transport-decay-start 0.38 --rational-transport-decay-end 0.56 --rational-transport-max-log-step 0.010 --rational-transport-derivative-weight 0.60 --rational-transport-headroom 0.90 --rational-matrix-policy-adam-beta2-final 0.995 --rational-matrix-policy-adam-beta2-decay-start 0.42 --rational-matrix-policy-adam-beta2-decay-end 0.78"
