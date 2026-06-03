#!/bin/bash
#SBATCH --job-name=rlb-extra
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
  "rlb_matrix_policy_group_policy030_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-group-gain-strength 0.30 --rational-matrix-policy-group-pressure-strength 0.15 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.06 --rational-matrix-policy-group-end 0.60 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"

run_variant \
  "rlb_matrix_policy_statgate_group018_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-adam-stat-strength 0.25 --rational-matrix-policy-adam-pressure-balance 0.15 --rational-matrix-policy-adam-stat-start 0.18 --rational-matrix-policy-adam-stat-end 0.55 --rational-matrix-policy-group-gain-strength 0.18 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.12 --rational-matrix-policy-group-start 0.06 --rational-matrix-policy-group-end 0.55 --rational-matrix-policy-group-min-scale 0.80 --rational-matrix-policy-group-max-scale 1.25"

run_variant \
  "rlb_matrix_policy_late_muon005_group012_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-final-muon 0.05 --rational-matrix-policy-muon-decay-depth-shift 0.08 --rational-matrix-policy-muon-input-decay-shift -0.02 --rational-matrix-policy-muon-output-decay-shift 0.04 --rational-matrix-policy-group-gain-strength 0.12 --rational-matrix-policy-group-pressure-strength 0.08 --rational-matrix-policy-group-activity-damping 0.08 --rational-matrix-policy-group-start 0.08 --rational-matrix-policy-group-end 0.55 --rational-matrix-policy-group-min-scale 0.85 --rational-matrix-policy-group-max-scale 1.20"

run_variant \
  "rlb_matrix_policy_statgate_late_muon003_a6000fb_ga2_probe_20260528" \
  "--rational-matrix-policy-final-muon 0.03 --rational-matrix-policy-muon-decay-depth-shift 0.06 --rational-matrix-policy-muon-input-decay-shift -0.01 --rational-matrix-policy-muon-output-decay-shift 0.03 --rational-matrix-policy-adam-stat-strength 0.20 --rational-matrix-policy-adam-pressure-balance 0.12 --rational-matrix-policy-adam-stat-start 0.18 --rational-matrix-policy-adam-stat-end 0.55"
