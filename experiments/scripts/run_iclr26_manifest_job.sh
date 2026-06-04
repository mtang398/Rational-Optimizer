#!/bin/bash
#SBATCH --job-name=iclr26-main
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

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-0}"
export PYTHONPATH="${PWD}/activation:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MAX_JOBS=8
export PATH="${PWD}/.venv-cu128/bin:${PATH}"

PYTHON="${PYTHON:-${PWD}/.venv-cu128/bin/python}"
MANIFEST="${MANIFEST:-experiments/manifests/iclr26_main_manifest.csv}"
ROW_START="${ROW_START:-0}"
ROW_LIMIT="${ROW_LIMIT:-1}"
CONFIRM_ICLR26_MANIFEST="${CONFIRM_ICLR26_MANIFEST:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/iclr26_main}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-experiments/cache/tokens_iclr26_main}"
MAX_REPO_GIB="${MAX_REPO_GIB:-190}"
MAX_EVAL_INTERVAL="${MAX_EVAL_INTERVAL:-50}"
BUILD_EXT="${BUILD_EXT:-1}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"

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

load_row_env() {
  local index="$1"
  local env_file="$2"
  "${PYTHON}" - "${MANIFEST}" "${index}" "${env_file}" <<'PYROW'
import csv
import shlex
import sys
manifest, index_s, env_path = sys.argv[1:4]
index = int(index_s)
with open(manifest, newline="") as handle:
    rows = list(csv.DictReader(handle))
if index < 0 or index >= len(rows):
    raise SystemExit(3)
row = rows[index]
with open(env_path, "w") as out:
    for key, value in row.items():
        safe_key = "ROW_" + key.upper()
        out.write(f"{safe_key}={shlex.quote(value)}\n")
PYROW
}

jsonl_complete() {
  local path="$1"
  local steps="$2"
  [[ -f "${path}" ]] || return 1
  "${PYTHON}" - "${path}" "${steps}" <<'PYJSON'
import json
import sys
path = sys.argv[1]
steps = int(sys.argv[2])
summary_complete = False
last_eval_step = None
try:
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
except FileNotFoundError:
    sys.exit(1)
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
  local archive_path="${path}.incomplete_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_${stamp}"
  echo "=== archiving incomplete ${path} -> ${archive_path} ==="
  mv "${path}" "${archive_path}"
}

run_manifest_row() {
  local index="$1"
  local row_env
  row_env="$(mktemp)"
  if ! load_row_env "${index}" "${row_env}"; then
    echo "=== manifest row ${index} does not exist; stopping chunk ==="
    rm -f "${row_env}"
    return 1
  fi
  # shellcheck disable=SC1090
  source "${row_env}"
  rm -f "${row_env}"

  if (( ROW_EVAL_INTERVAL > MAX_EVAL_INTERVAL )); then
    echo "Row ${ROW_ROW_ID} eval interval ${ROW_EVAL_INTERVAL} exceeds max ${MAX_EVAL_INTERVAL}" >&2
    exit 2
  fi

  local run_dir="${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET}/${ROW_ROW_ID}"
  local jsonl="${run_dir}/${ROW_ACTIVATION}.jsonl"
  if jsonl_complete "${jsonl}" "${ROW_STEPS}"; then
    echo "=== row ${ROW_ROW_INDEX} ${ROW_ROW_ID} already complete; skipping ==="
    return 0
  fi
  archive_incomplete_jsonl "${jsonl}"
  mkdir -p "${run_dir}"

  echo "=== row=${ROW_ROW_INDEX}; id=${ROW_ROW_ID}; phase=${ROW_PHASE}; dataset=${ROW_DATASET}; method=${ROW_METHOD}; seed=${ROW_SEED}; one job uses 4 A6000s ==="

  RUN_NAME="${ROW_ROW_ID}" \
  STEPS="${ROW_STEPS}" \
  SEEDS="${ROW_SEED}" \
  OPTIMIZERS="${ROW_OPTIMIZER}" \
  ACTIVATIONS="${ROW_ACTIVATION}" \
  EVAL_INTERVAL="${ROW_EVAL_INTERVAL}" \
  EVAL_BATCHES="${ROW_EVAL_BATCHES}" \
  LOG_INTERVAL="10" \
  NPROC_PER_NODE="4" \
  SKIP_BUILD_EXT="1" \
  EXTRA_ARGS="--dataset-name ${ROW_DATASET_NAME} --dataset-config ${ROW_DATASET_CONFIG} --dataset-streaming --dataset-text-column ${ROW_TEXT_COLUMN} --train-split ${ROW_TRAIN_SPLIT} --validation-split ${ROW_VAL_SPLIT} --validation-skip-tokens ${ROW_VAL_SKIP_TOKENS} --cache-dir ${TOKEN_CACHE_DIR}/${ROW_DATASET} --output-dir ${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET} --max-train-tokens ${ROW_TRAIN_TOKENS} --max-val-tokens ${ROW_VAL_TOKENS} --batch-size ${ROW_BATCH_SIZE} --grad-accum ${ROW_GRAD_ACCUM} --layers ${ROW_LAYERS} --d-model ${ROW_D_MODEL} --heads ${ROW_HEADS} --ffn-dim ${ROW_FFN_DIM} --lr ${ROW_LR} --min-lr ${ROW_MIN_LR} --weight-decay ${ROW_WEIGHT_DECAY} --probe-batch-size 1 --matrix-spectrum-interval 250 ${ROW_EXTRA_ARGS} ${COMMON_EXTRA_ARGS}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

if [[ "${CONFIRM_ICLR26_MANIFEST}" != "1" ]]; then
  echo "Refusing to start without CONFIRM_ICLR26_MANIFEST=1." >&2
  echo "Generate and inspect ${MANIFEST} first. This launcher is for manifest rows only." >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest not found: ${MANIFEST}" >&2
  exit 2
fi
if (( ROW_LIMIT <= 0 )); then
  echo "ROW_LIMIT must be positive." >&2
  exit 2
fi

mkdir -p experiments/runs/logs
check_repo_size
if [[ "${BUILD_EXT}" == "1" ]]; then
  BUILD_EXT_ROOT="${OUTPUT_ROOT}/_build/${SLURM_JOB_ID:-manual}"
  mkdir -p "${BUILD_EXT_ROOT}/temp" "${BUILD_EXT_ROOT}/lib"
  "${PYTHON}" setup.py build_ext --inplace --build-temp "${BUILD_EXT_ROOT}/temp" --build-lib "${BUILD_EXT_ROOT}/lib"
fi

end=$((ROW_START + ROW_LIMIT))
for ((idx=ROW_START; idx<end; idx++)); do
  check_repo_size
  run_manifest_row "${idx}" || break
done
