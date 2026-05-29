#!/bin/bash
#SBATCH --job-name=synth-fair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

STEPS="${STEPS:-1250}"
SEEDS="${SEEDS:-1337}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
RUN_SUFFIX="${RUN_SUFFIX:-20260529_fair_full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/synthetic_fair_full_20260529}"
TASKS="${SYNTHETIC_TASKS:-synthetic/code synthetic/symbolic synthetic/reasoning_mix}"
RLB_ACTIVATION="${RLB_ACTIVATION:-rlb_fused_fixed_strong_ffn}"

run_variant() {
  local dataset_name="$1"
  local safe_name="$2"
  local run_tag="$3"
  local optimizers="$4"
  local activations="$5"
  local extra_args="${6:-}"
  local run_name="${safe_name}_${run_tag}_${RUN_SUFFIX}"

  echo "=== ${run_name} (${dataset_name}) ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS}" \
  SEEDS="${SEEDS}" \
  OPTIMIZERS="${optimizers}" \
  ACTIVATIONS="${activations}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  EXTRA_ARGS="--dataset-name ${dataset_name} --dataset-config v1 --output-dir ${OUTPUT_ROOT}/${safe_name} --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

for dataset_name in ${TASKS}; do
  safe_name="${dataset_name//\//_}"

  run_variant \
    "${dataset_name}" \
    "${safe_name}" \
    "adamw_controls" \
    "adamw" \
    "silu ${RLB_ACTIVATION}"

  run_variant \
    "${dataset_name}" \
    "${safe_name}" \
    "muon_controls" \
    "muon" \
    "silu ${RLB_ACTIVATION}"

  run_variant \
    "${dataset_name}" \
    "${safe_name}" \
    "matrix_policy" \
    "rational_matrix_policy_onpolicy" \
    "${RLB_ACTIVATION}" \
    "--rational-matrix-policy-backbone-optimizer adamw"

  run_variant \
    "${dataset_name}" \
    "${safe_name}" \
    "matrix_policy_groupstat" \
    "rational_matrix_policy_onpolicy" \
    "${RLB_ACTIVATION}" \
    "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-group-gain-strength 0.20 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"
done
