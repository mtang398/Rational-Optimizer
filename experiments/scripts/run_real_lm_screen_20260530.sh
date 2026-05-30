#!/bin/bash
#SBATCH --job-name=real-lm-screen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
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
REAL_LM_TASKS="${REAL_LM_TASKS:-fineweb_edu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/real_lm_screen_20260530}"
RUN_SUFFIX="${RUN_SUFFIX:-20260530_real_lm_100m}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-experiments/cache/tokens_real_lm}"
RLB_ACTIVATION="${RLB_ACTIVATION:-rlb_fused_fixed_strong_ffn}"
SEEDS="${SEEDS:-1337}"
STEPS="${STEPS:-3050}"
MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-100000000}"
MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-4000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
INCLUDE_MUON="${INCLUDE_MUON:-1}"
INCLUDE_PLAIN_MATRIX_POLICY="${INCLUDE_PLAIN_MATRIX_POLICY:-0}"
MAX_REPO_GIB="${MAX_REPO_GIB:-190}"
DEFAULT_VAL_SKIP_DOCS="${DEFAULT_VAL_SKIP_DOCS:-0}"
VAL_SKIP_TOKENS="${VAL_SKIP_TOKENS:-110000000}"

request_requeue() {
  echo "=== received USR1 at $(date -Is); requesting requeue for job ${SLURM_JOB_ID:-manual} ==="
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol requeue "${SLURM_JOB_ID}" || true
  fi
  exit 0
}
trap request_requeue USR1

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

run_complete() {
  local run_dir="$1"
  local activations="$2"
  local activation
  for activation in ${activations}; do
    jsonl_complete "${run_dir}/${activation}.jsonl" || return 1
  done
  return 0
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

task_spec() {
  local task="$1"
  case "${task}" in
    fineweb_edu)
      DATASET_NAME="HuggingFaceFW/fineweb-edu"
      DATASET_CONFIG="sample-10BT"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_DOCS="${FINEWEB_EDU_VAL_SKIP_DOCS:-${DEFAULT_VAL_SKIP_DOCS}}"
      ;;
    fineweb)
      DATASET_NAME="HuggingFaceFW/fineweb"
      DATASET_CONFIG="sample-10BT"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_DOCS="${FINEWEB_VAL_SKIP_DOCS:-${DEFAULT_VAL_SKIP_DOCS}}"
      ;;
    dclm)
      DATASET_NAME="mlfoundations/dclm-baseline-1.0"
      DATASET_CONFIG="none"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_DOCS="${DCLM_VAL_SKIP_DOCS:-${DEFAULT_VAL_SKIP_DOCS}}"
      ;;
    dolma_sample)
      DATASET_NAME="allenai/dolma"
      DATASET_CONFIG="v1_6-sample"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_DOCS="${DOLMA_VAL_SKIP_DOCS:-${DEFAULT_VAL_SKIP_DOCS}}"
      ;;
    *)
      echo "Unknown REAL_LM_TASK '${task}'. Valid: fineweb_edu fineweb dclm dolma_sample" >&2
      exit 2
      ;;
  esac
}

run_variant() {
  local task="$1"
  local run_tag="$2"
  local optimizers="$3"
  local activations="$4"
  local extra_args="${5:-}"
  local run_name="${task}_${run_tag}_${RUN_SUFFIX}"
  local run_dir="${OUTPUT_ROOT}/${task}/${run_name}"

  check_repo_size
  if run_complete "${run_dir}" "${activations}"; then
    echo "=== ${run_name} already complete; skipping ==="
    return 0
  fi
  archive_incomplete_run "${run_dir}"

  echo "=== ${run_name}: ${DATASET_NAME}/${DATASET_CONFIG}; steps=${STEPS}; train_tokens=${MAX_TRAIN_TOKENS}; val_tokens=${MAX_VAL_TOKENS}; val_skip_tokens=${VAL_SKIP_TOKENS} ==="
  RUN_NAME="${run_name}"   STEPS="${STEPS}"   SEEDS="${SEEDS}"   OPTIMIZERS="${optimizers}"   ACTIVATIONS="${activations}"   EVAL_INTERVAL="${EVAL_INTERVAL}"   EVAL_BATCHES="${EVAL_BATCHES}"   LOG_INTERVAL="${LOG_INTERVAL}"   NPROC_PER_NODE="${NPROC_PER_NODE}"   EXTRA_ARGS="--dataset-name ${DATASET_NAME} --dataset-config ${DATASET_CONFIG} --dataset-streaming --dataset-text-column ${TEXT_COLUMN} --train-split ${TRAIN_SPLIT} --validation-split ${VAL_SPLIT} --validation-skip-documents ${VAL_SKIP_DOCS} --validation-skip-tokens ${VAL_SKIP_TOKENS} --cache-dir ${TOKEN_CACHE_DIR} --output-dir ${OUTPUT_ROOT}/${task} --max-train-tokens ${MAX_TRAIN_TOKENS} --max-val-tokens ${MAX_VAL_TOKENS} --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} ${extra_args}"   bash training/run_wikitext103_optimizer_sweep.sbatch
}

echo "=== real LM screen job ${SLURM_JOB_ID:-manual}; tasks=${REAL_LM_TASKS}; one job uses 4 A6000s; keep at most two active for 8-GPU cap ==="
check_repo_size

for task in ${REAL_LM_TASKS}; do
  task_spec "${task}"
  run_variant "${task}" "adamw_controls" "adamw" "silu ${RLB_ACTIVATION}"
  if [[ "${INCLUDE_MUON}" == "1" ]]; then
    run_variant "${task}" "muon_controls" "muon" "silu ${RLB_ACTIVATION}"
  fi
  if [[ "${INCLUDE_PLAIN_MATRIX_POLICY}" == "1" ]]; then
    run_variant       "${task}"       "matrix_policy"       "rational_matrix_policy_onpolicy"       "${RLB_ACTIVATION}"       "--rational-matrix-policy-backbone-optimizer adamw"
  fi
  run_variant     "${task}"     "matrix_policy_groupstat"     "rational_matrix_policy_onpolicy"     "${RLB_ACTIVATION}"     "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-group-gain-strength 0.20 --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"
done
