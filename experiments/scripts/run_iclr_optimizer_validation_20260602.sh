#!/bin/bash
#SBATCH --job-name=iclr-opt-val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export PATH="${PWD}/.venv-cu128/bin:${PATH}"

PYTHON="${PYTHON:-${PWD}/.venv-cu128/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/iclr_optimizer_validation_20260602}"
RUN_SUFFIX="${RUN_SUFFIX:-20260602_tiny_cuda_ddp}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-experiments/cache/tokens_iclr_validation}"
RLB_ACTIVATION="${RLB_ACTIVATION:-rlb_fused_fixed_strong_ffn}"
SEEDS="${SEEDS:-1337}"
STEPS="${STEPS:-6}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000}"
MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-12000}"
MAX_REPO_GIB="${MAX_REPO_GIB:-190}"
BROAD_OPTIMIZERS="${BROAD_OPTIMIZERS:-adamw lion ademamix schedule_free_adamw adafactor_came soap_adamw}"
INCLUDE_MATRIX_POLICY="${INCLUDE_MATRIX_POLICY:-1}"

check_repo_size() {
  local used_kib
  used_kib="$(du -s "${PWD}" | awk '{print $1}')"
  local limit_kib=$((MAX_REPO_GIB * 1024 * 1024))
  if (( used_kib > limit_kib )); then
    echo "Repo is above ${MAX_REPO_GIB} GiB cap; refusing to continue. Used KiB=${used_kib}" >&2
    exit 80
  fi
}

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
with open(path) as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "summary" and int(record.get("steps", -1)) == steps:
            summary_complete = True
        if record.get("event") == "eval":
            last_eval_step = record.get("step")
sys.exit(0 if summary_complete and int(last_eval_step or -1) == steps else 1)
PYJSON
}

archive_incomplete_jsonl() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  local stamp
  stamp="$(date +%Y%m%d%H%M%S)"
  local archive_path="${path}.incomplete_${SLURM_JOB_ID:-manual}_${stamp}"
  echo "=== archiving incomplete ${path} -> ${archive_path} ==="
  mv "${path}" "${archive_path}"
}

run_validation_variant() {
  local optimizer="$1"
  local activations="$2"
  local extra_args="${3:-}"
  local run_name="validation_${optimizer}_${RUN_SUFFIX}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local pending=""
  local activation

  check_repo_size
  for activation in ${activations}; do
    if jsonl_complete "${run_dir}/${activation}.jsonl"; then
      echo "=== ${run_name}/${activation} already complete; skipping ==="
    else
      archive_incomplete_jsonl "${run_dir}/${activation}.jsonl"
      pending="${pending} ${activation}"
    fi
  done
  pending="${pending# }"
  if [[ -z "${pending}" ]]; then
    return 0
  fi

  echo "=== validating optimizer=${optimizer}; activations=${pending}; steps=${STEPS}; GPUs=${NPROC_PER_NODE} ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS}" \
  SEEDS="${SEEDS}" \
  OPTIMIZERS="${optimizer}" \
  ACTIVATIONS="${pending}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  SKIP_BUILD_EXT="1" \
  EXTRA_ARGS="--dataset-name synthetic/reasoning_mix --dataset-config none --dataset-text-column text --cache-dir ${TOKEN_CACHE_DIR} --output-dir ${OUTPUT_ROOT} --max-train-tokens ${MAX_TRAIN_TOKENS} --max-val-tokens ${MAX_VAL_TOKENS} --seq-len 64 --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} --layers 2 --d-model 128 --heads 4 --ffn-dim 384 --rational-group-size 64 --rational-max-groups 8 --warmup-steps 2 --eval-batches ${EVAL_BATCHES} --probe-batch-size 1 --telemetry-rlb-stat-every 1 --telemetry-rlb-stat-samples 64 --telemetry-denominator-probe-points 65 --matrix-spectrum-interval 2 --matrix-spectrum-max-dim 128 ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

mkdir -p experiments/runs/logs
check_repo_size
"${PYTHON}" setup.py build_ext --inplace

echo "=== ICLR optimizer validation ${SLURM_JOB_ID:-manual}; one job uses 4 A6000s; keep at most two active jobs total ==="
for optimizer in ${BROAD_OPTIMIZERS}; do
  run_validation_variant "${optimizer}" "silu ${RLB_ACTIVATION}"
done

if [[ "${INCLUDE_MATRIX_POLICY}" == "1" ]]; then
  run_validation_variant \
    "rational_matrix_policy_onpolicy" \
    "${RLB_ACTIVATION}" \
    "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-group-gain-strength 0.20 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"
fi

"${PYTHON}" experiments/scripts/check_iclr_validation_jsonl.py \
  --run-root "${OUTPUT_ROOT}" \
  --output-md "${OUTPUT_ROOT}/validation_summary.md"
