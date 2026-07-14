#!/bin/bash
#SBATCH --job-name=mp-ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mt872/rationalOPT/abalation/logs/%x-%j.out

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
MANIFEST="${MANIFEST:-abalation/manifests/matrixpolicy_ablation_e1_e2_manifest.csv}"
ROW_START="${ROW_START:-0}"
ROW_LIMIT="${ROW_LIMIT:-1}"
CONFIRM_ICLR26_MANIFEST="${CONFIRM_ICLR26_MANIFEST:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-abalation/runs/matrixpolicy_ablation_e1_e2}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-experiments/cache/tokens_iclr26_main}"
MAX_REPO_GIB="${MAX_REPO_GIB:-190}"
MAX_EVAL_INTERVAL="${MAX_EVAL_INTERVAL:-50}"
BUILD_EXT="${BUILD_EXT:-1}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
TIMING_NODE_DENYLIST="${TIMING_NODE_DENYLIST:-sablab-gpu-12 seo-compute-01}"
TIMING_GUARD_MIN_STEP="${TIMING_GUARD_MIN_STEP:-300}"
TIMING_GUARD_MAX_SECONDS_PER_STEP="${TIMING_GUARD_MAX_SECONDS_PER_STEP:-0.0}"
TIMING_GUARD_MAX_REQUEUES="${TIMING_GUARD_MAX_REQUEUES:-4}"
MAX_PREEMPT_REQUEUES="${MAX_PREEMPT_REQUEUES:-8}"
FORCE_RERUN_COMPLETE_JSONL="${FORCE_RERUN_COMPLETE_JSONL:-0}"
CURRENT_JSONL=""
CURRENT_ROW_ID=""
CURRENT_ATTEMPT_RECORDED=0
TRAINING_PID=""

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ROW_START="${SLURM_ARRAY_TASK_ID}"
  ROW_LIMIT="${ROW_LIMIT:-1}"
fi

slurm_requeue_ref() {
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    printf "%s_%s" "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
  else
    printf "%s" "${SLURM_JOB_ID:-}"
  fi
}

record_attempt_end() {
  local reason="$1"
  if [[ "${CURRENT_ATTEMPT_RECORDED}" == "1" || -z "${CURRENT_JSONL}" || ! -f "${CURRENT_JSONL}" ]]; then
    return 0
  fi
  CURRENT_ATTEMPT_RECORDED=1
  "${PYTHON}" - "${CURRENT_JSONL}" "${reason}" "${CURRENT_ROW_ID}" <<'PYATTEMPT' || true
import json
import os
import sys
import time

path, reason, row_id = sys.argv[1:4]
last_active = None
attempt_id = None
try:
    with open(path) as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if record.get("timing_attempt_id"):
                attempt_id = record["timing_attempt_id"]
            for key in ("active_seconds_after_event", "active_seconds_at_val_loss"):
                value = record.get(key)
                if isinstance(value, (int, float)):
                    last_active = value if last_active is None else max(last_active, value)
event = {
    "event": "attempt_interrupted",
    "reason": reason,
    "row_id": row_id,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0),
    "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
    "timing_attempt_id": attempt_id,
    "active_seconds_lower_bound": last_active,
    "recorded_unix_time": time.time(),
}
with open(path, "a") as handle:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
PYATTEMPT
}

request_requeue() {
  echo "=== received USR1 at $(date -Is); requesting requeue for job ${SLURM_JOB_ID:-manual} ==="
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  if [[ -n "${TRAINING_PID}" ]]; then
    kill -TERM -- "-${TRAINING_PID}" 2>/dev/null || true
  fi
  record_attempt_end "slurm_usr1_preemption"
  local job_ref
  job_ref="$(slurm_requeue_ref)"
  if [[ -n "${job_ref}" ]] && command -v scontrol >/dev/null 2>&1 && (( restart_count < MAX_PREEMPT_REQUEUES )); then
    if scontrol requeue "${job_ref}"; then
      exit 0
    fi
  fi
  echo "=== preemption requeue failed or exceeded cap; row ${CURRENT_ROW_ID} remains unresolved ===" >&2
  exit 89
}
trap request_requeue USR1

request_timing_requeue() {
  local reason="$1"
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  echo "=== timing guard requesting requeue at $(date -Is): ${reason}; restart_count=${restart_count}; max=${TIMING_GUARD_MAX_REQUEUES} ===" >&2
  record_attempt_end "timing_guard_requeue"
  if [[ "${DISABLE_TIMING_GUARD_REQUEUE:-0}" == "1" ]]; then
    echo "=== DISABLE_TIMING_GUARD_REQUEUE=1; failing timing row instead of requeueing ===" >&2
    exit 88
  fi
  local job_ref
  job_ref="$(slurm_requeue_ref)"
  if [[ -n "${job_ref}" ]] && command -v scontrol >/dev/null 2>&1 && (( restart_count < TIMING_GUARD_MAX_REQUEUES )); then
    if scontrol requeue "${job_ref}"; then
      exit 0
    fi
    echo "=== scontrol requeue failed for timing row ${SLURM_JOB_ID}; failing timing row ===" >&2
    exit 88
  fi
  echo "=== timing guard could not requeue safely; failing timing row ===" >&2
  exit 88
}

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

jsonl_terminal() {
  local path="$1"
  local steps="$2"
  local phase="$3"
  [[ -f "${path}" ]] || return 1
  "${PYTHON}" - "${path}" "${steps}" "${phase}" <<'PYJSON'
import json
import sys
path = sys.argv[1]
steps = int(sys.argv[2])
phase = sys.argv[3]
summary_complete = False
summary_terminal_failure = False
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
                summary_terminal_failure = bool(record.get("stopped_early")) and bool(record.get("stop_reason"))
            if record.get("event") == "eval":
                last_eval_step = record.get("step")
except FileNotFoundError:
    sys.exit(1)
full_completion = summary_complete and int(last_eval_step or -1) == steps
scientific_failure = phase == "E9_100m" and summary_complete and summary_terminal_failure
sys.exit(0 if full_completion or scientific_failure else 1)
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

require_nvlink_for_timing_row() {
  case "${ROW_PHASE}" in
    E1_matrixpolicy_safe_speed_100m|E2_matrixpolicy_safe_speed_300m|E1_fineweb_edu_seed2027_runtime_repair_100m|E1_rational_only_100m|E2_rational_only_300m|E1_global_rational_optimizers_100m|E2_global_rational_optimizers_300m|E1_matrixpolicy_ablation_no_role_depth_100m|E2_matrixpolicy_ablation_no_role_depth_300m|E1_matrixpolicy_ablation_bypass_muon_100m|E2_matrixpolicy_ablation_bypass_muon_300m|E1_matrixpolicy_ablation_role_depth_v2_100m|E2_matrixpolicy_ablation_role_depth_v2_300m|E1_matrixpolicy_ablation_role_depth_v3_100m|E2_matrixpolicy_ablation_role_depth_v3_300m|E1_matrixpolicy_ablation_role_depth_v4_100m|E2_matrixpolicy_ablation_role_depth_v4_300m|E1_matrixpolicy_ablation_role_depth_v5_100m|E2_matrixpolicy_ablation_role_depth_v5_300m|E1_matrixpolicy_ablation_role_depth_v6_100m|E2_matrixpolicy_ablation_role_depth_v6_300m|E9_100m|E9_preflight_80step)
      ;;
    *)
      return 0
      ;;
  esac

  if [[ "${ALLOW_NON_NVLINK_TIMING:-0}" == "1" ]]; then
    echo "=== ALLOW_NON_NVLINK_TIMING=1; allowing timing row ${ROW_ROW_ID} without NVLink guard ==="
    return 0
  fi

  local node="${SLURMD_NODENAME:-${SLURM_NODELIST:-}}"
  if [[ -z "${node}" ]]; then
    echo "Refusing timing row ${ROW_ROW_ID}: cannot determine Slurm node for NVLink guard." >&2
    exit 87
  fi
  local bad_node
  for bad_node in ${TIMING_NODE_DENYLIST//,/ }; do
    if [[ -n "${bad_node}" && "${node}" == "${bad_node}" ]]; then
      request_timing_requeue "timing row ${ROW_ROW_ID} landed on denylisted node ${node}"
    fi
  done
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "Refusing timing row ${ROW_ROW_ID}: scontrol unavailable for NVLink guard." >&2
    exit 87
  fi

  local node_info
  node_info="$(scontrol show node "${node}" 2>/dev/null || true)"
  if [[ "${node_info}" != *"nvlink"* ]]; then
    echo "Refusing timing row ${ROW_ROW_ID}: node ${node} lacks nvlink feature; not writing timing JSONL." >&2
    echo "${node_info}" | sed -n '1,6p' >&2
    exit 87
  fi
  echo "=== NVLink timing guard passed for ${ROW_ROW_ID} on ${node} ==="
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
  if [[ "${ROW_PHASE}" == E9_* && -n "${COMMON_EXTRA_ARGS}" ]]; then
    echo "E9 forbids COMMON_EXTRA_ARGS because manifest arguments are frozen." >&2
    exit 2
  fi
  require_nvlink_for_timing_row

  local run_dir="${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET}/${ROW_ROW_ID}"
  local jsonl="${run_dir}/${ROW_ACTIVATION}.jsonl"
  if jsonl_terminal "${jsonl}" "${ROW_STEPS}" "${ROW_PHASE}"; then
    if [[ "${FORCE_RERUN_COMPLETE_JSONL}" == "1" ]]; then
      local stamp
      stamp="$(date +%Y%m%d%H%M%S)"
      local archive_path="${jsonl}.rerun_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_${stamp}"
      echo "=== FORCE_RERUN_COMPLETE_JSONL=1; archiving complete ${jsonl} -> ${archive_path} ==="
      mv "${jsonl}" "${archive_path}"
    else
      echo "=== row ${ROW_ROW_INDEX} ${ROW_ROW_ID} already complete; skipping ==="
      return 0
    fi
  fi
  archive_incomplete_jsonl "${jsonl}"
  mkdir -p "${run_dir}"
  CURRENT_JSONL="${jsonl}"
  CURRENT_ROW_ID="${ROW_ROW_ID}"
  CURRENT_ATTEMPT_RECORDED=0

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== allocated GPU metadata for ${ROW_ROW_ID} ==="
    if ! nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,clocks.max.sm,power.limit --format=csv,noheader; then
      [[ "${ROW_PHASE}" != E9_* ]] || exit 86
    fi
  elif [[ "${ROW_PHASE}" == E9_* ]]; then
    echo "E9 requires nvidia-smi metadata." >&2
    exit 86
  fi

  local timing_guard_extra=""
  case "${ROW_PHASE}" in
    E1_matrixpolicy_safe_speed_100m|E2_matrixpolicy_safe_speed_300m|E1_rational_only_100m|E2_rational_only_300m|E1_matrixpolicy_ablation_no_role_depth_100m|E2_matrixpolicy_ablation_no_role_depth_300m|E1_matrixpolicy_ablation_bypass_muon_100m|E2_matrixpolicy_ablation_bypass_muon_300m|E1_matrixpolicy_ablation_role_depth_v2_100m|E2_matrixpolicy_ablation_role_depth_v2_300m|E1_matrixpolicy_ablation_role_depth_v3_100m|E2_matrixpolicy_ablation_role_depth_v3_300m|E1_matrixpolicy_ablation_role_depth_v4_100m|E2_matrixpolicy_ablation_role_depth_v4_300m|E1_matrixpolicy_ablation_role_depth_v5_100m|E2_matrixpolicy_ablation_role_depth_v5_300m|E1_matrixpolicy_ablation_role_depth_v6_100m|E2_matrixpolicy_ablation_role_depth_v6_300m|E9_100m|E9_preflight_80step)
      if [[ "${ROW_OPTIMIZER}" == "rational_matrix_policy_onpolicy" || "${ROW_PHASE}" == E9_* ]]; then
        timing_guard_extra="--timing-guard-min-step ${TIMING_GUARD_MIN_STEP} --timing-guard-max-seconds-per-step ${TIMING_GUARD_MAX_SECONDS_PER_STEP}"
      fi
      ;;
  esac

  echo "=== row=${ROW_ROW_INDEX}; id=${ROW_ROW_ID}; phase=${ROW_PHASE}; dataset=${ROW_DATASET}; method=${ROW_METHOD}; seed=${ROW_SEED}; one job uses 4 A6000s ==="

  set +e
  export E9_ROW_ID="${ROW_ROW_ID}"
  export E9_ARM_ID="${ROW_ARM_ID:-}"
  export E9_DESIGN_VERSION="${ROW_DESIGN_VERSION:-}"
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
  EXTRA_ARGS="--dataset-name ${ROW_DATASET_NAME} --dataset-config ${ROW_DATASET_CONFIG} --dataset-streaming --dataset-text-column ${ROW_TEXT_COLUMN} --train-split ${ROW_TRAIN_SPLIT} --validation-split ${ROW_VAL_SPLIT} --validation-skip-tokens ${ROW_VAL_SKIP_TOKENS} --cache-dir ${TOKEN_CACHE_DIR}/${ROW_DATASET} --output-dir ${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET} --max-train-tokens ${ROW_TRAIN_TOKENS} --max-val-tokens ${ROW_VAL_TOKENS} --batch-size ${ROW_BATCH_SIZE} --grad-accum ${ROW_GRAD_ACCUM} --seq-len ${ROW_SEQ_LEN} --layers ${ROW_LAYERS} --d-model ${ROW_D_MODEL} --heads ${ROW_HEADS} --ffn-dim ${ROW_FFN_DIM} --lr ${ROW_LR} --min-lr ${ROW_MIN_LR} --weight-decay ${ROW_WEIGHT_DECAY} --probe-batch-size 1 --matrix-spectrum-interval 250 ${ROW_EXTRA_ARGS} ${COMMON_EXTRA_ARGS} ${timing_guard_extra}" \
  setsid bash training/run_lm_optimizer_sweep.sbatch &
  TRAINING_PID=$!
  wait "${TRAINING_PID}"
  local status=$?
  TRAINING_PID=""
  set -e
  if (( status == 88 )); then
    request_timing_requeue "training timing guard failed for ${ROW_ROW_ID}"
  fi
  return "${status}"
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
  set +e
  run_manifest_row "${idx}"
  status=$?
  set -e
  if (( status != 0 )); then
    echo "=== manifest row ${idx} failed with status ${status}; exiting nonzero ===" >&2
    exit "${status}"
  fi
done
