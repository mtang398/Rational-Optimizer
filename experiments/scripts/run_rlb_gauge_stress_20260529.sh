#!/bin/bash
#SBATCH --job-name=rlb-gauge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

PYTHON="${PYTHON:-${PWD}/.venv-cu128/bin/python}"
STEPS="${STEPS:-750}"
SEEDS="${SEEDS:-1337}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
GAUGE_LOG_SCALES="${GAUGE_LOG_SCALES:-0.0 2.0}"
GAUGE_SEED="${GAUGE_SEED:-271828}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/rlb_gauge_stress_20260529}"
TASKS="${SYNTHETIC_TASKS:-synthetic/code synthetic/reasoning_mix}"
RLB_ACTIVATION="${RLB_ACTIVATION:-rlb_fused_fixed_strong_ffn}"

request_requeue() {
  echo "=== received USR1 at $(date -Is); requesting requeue for job ${SLURM_JOB_ID:-manual} ==="
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol requeue "${SLURM_JOB_ID}" || true
  fi
  exit 0
}
trap request_requeue USR1

echo "=== RLB gauge stress job ${SLURM_JOB_ID:-manual}; tasks: ${TASKS}; gauges: ${GAUGE_LOG_SCALES} ==="

jsonl_complete() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  "${PYTHON}" - "${path}" "${STEPS}" <<'PYJSON'
import json
import sys
path = sys.argv[1]
steps = int(sys.argv[2])
summary_complete = False
last_eval_step = None
try:
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "summary" and int(record.get("steps", -1)) == steps:
                summary_complete = True
            if record.get("event") == "eval":
                last_eval_step = record.get("step")
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if summary_complete and int(last_eval_step or -1) == steps else 1)
PYJSON
}

archive_incomplete_run() {
  local run_dir="$1"
  if [[ ! -d "${run_dir}" ]]; then
    return 0
  fi
  local stamp
  stamp="$(date +%Y%m%d%H%M%S)"
  local archive_dir="${run_dir}.incomplete_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_${stamp}"
  echo "=== archiving incomplete run ${run_dir} -> ${archive_dir} ==="
  mv "${run_dir}" "${archive_dir}"
}

run_variant() {
  local dataset_name="$1"
  local safe_name="$2"
  local run_tag="$3"
  local optimizer="$4"
  local gauge_log_scale="$5"
  local extra_args="${6:-}"
  local run_name="${safe_name}_${run_tag}_gauge${gauge_log_scale//./p}_20260529"
  local run_dir="${OUTPUT_ROOT}/${safe_name}/${run_name}"
  local path="${run_dir}/${RLB_ACTIVATION}.jsonl"

  if jsonl_complete "${path}"; then
    echo "=== ${run_name} already complete; skipping ==="
    return 0
  fi
  archive_incomplete_run "${run_dir}"

  echo "=== ${run_name} (${dataset_name}, optimizer=${optimizer}, gauge=${gauge_log_scale}) ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS}" \
  SEEDS="${SEEDS}" \
  OPTIMIZERS="${optimizer}" \
  ACTIVATIONS="${RLB_ACTIVATION}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  EXTRA_ARGS="--dataset-name ${dataset_name} --dataset-config v1 --output-dir ${OUTPUT_ROOT}/${safe_name} --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} --rlb-init-gauge-log-scale ${gauge_log_scale} --rlb-init-gauge-seed ${GAUGE_SEED} ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

for dataset_name in ${TASKS}; do
  safe_name="${dataset_name//\//_}"
  for gauge in ${GAUGE_LOG_SCALES}; do
    run_variant "${dataset_name}" "${safe_name}" "rlb_adamw" "adamw" "${gauge}"
    run_variant "${dataset_name}" "${safe_name}" "rlb_muon" "muon" "${gauge}"
    run_variant \
      "${dataset_name}" \
      "${safe_name}" \
      "matrix_policy" \
      "rational_matrix_policy_onpolicy" \
      "${gauge}" \
      "--rational-matrix-policy-backbone-optimizer adamw"
    run_variant \
      "${dataset_name}" \
      "${safe_name}" \
      "matrix_policy_groupstat" \
      "rational_matrix_policy_onpolicy" \
      "${gauge}" \
      "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-group-gain-strength 0.20 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"
  done
done
