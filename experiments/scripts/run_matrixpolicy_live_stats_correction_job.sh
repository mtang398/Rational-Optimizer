#!/bin/bash
#SBATCH --job-name=mp-syncfix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

set -euo pipefail

ROOT="/home/mt872/rationalOPT"
CAMPAIGN_ROOT="${ROOT}/experiments/corrections/matrixpolicy_live_stats_20260712"
RUNTIME_ROOT="${CAMPAIGN_ROOT}/runtime"
PYTHON="${ROOT}/.venv-cu128/bin/python"
TOKEN_CACHE_DIR="${CAMPAIGN_ROOT}/cache/tokens_iclr26_main"
HF_CACHE_DIR="${CAMPAIGN_ROOT}/cache/huggingface"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
ROW_START="${ROW_START:?ROW_START is required}"
MAX_PREEMPT_REQUEUES="${MAX_PREEMPT_REQUEUES:-8}"
NODE_DENYLIST="${NODE_DENYLIST:-sablab-gpu-12 seo-compute-01}"
CURRENT_JSONL=""
CURRENT_ROW_ID=""
CURRENT_ATTEMPT_RECORDED=0
TRAINING_PID=""

cd "${RUNTIME_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export RATIONALOPT_WORKSPACE_ROOT="${ROOT}"
export RATIONAL_OPT_TORCH_FALLBACK=0
export PYTHONPATH="${RUNTIME_ROOT}/activation:${RUNTIME_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PATH="${ROOT}/.venv-cu128/bin:${PATH}"
export E9_STRICT_GPU_METADATA=1

if (( ${SLURM_RESTART_COUNT:-0} > MAX_PREEMPT_REQUEUES )); then
  echo "Preemption retry cap exceeded before row start: ${SLURM_RESTART_COUNT}" >&2
  exit 89
fi

"${PYTHON}" "${RUNTIME_ROOT}/experiments/scripts/build_matrixpolicy_live_stats_correction.py" verify-runtime >/dev/null

load_row_env() {
  local env_path="$1"
  "${PYTHON}" - "${MANIFEST}" "${ROW_START}" "${env_path}" <<'PYROW'
import csv
import shlex
import sys

manifest, index_text, env_path = sys.argv[1:4]
index = int(index_text)
with open(manifest, newline="") as handle:
    rows = list(csv.DictReader(handle))
if not 0 <= index < len(rows):
    raise SystemExit(f"manifest row {index} is outside 0..{len(rows) - 1}")
with open(env_path, "w") as handle:
    for key, value in rows[index].items():
        handle.write(f"ROW_{key.upper()}={shlex.quote(value)}\n")
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

path, steps_text, phase = sys.argv[1:4]
steps = int(steps_text)
summary = None
last_eval_step = None
with open(path) as handle:
    for raw in handle:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "summary":
            summary = record
        elif record.get("event") == "eval":
            last_eval_step = record.get("step")
complete = (
    summary is not None
    and int(summary.get("steps", -1)) == steps
    and int(last_eval_step or -1) == steps
)
scientific_failure = (
    phase == "E9_100m"
    and summary is not None
    and bool(summary.get("stopped_early"))
    and bool(summary.get("stop_reason"))
)
raise SystemExit(0 if complete or scientific_failure else 1)
PYJSON
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
event = {
    "event": "attempt_interrupted",
    "reason": reason,
    "row_id": row_id,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0),
    "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
    "recorded_unix_time": time.time(),
}
with open(path, "a") as handle:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
PYATTEMPT
}

request_requeue() {
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  echo "=== received USR1 at $(date -Is); row=${CURRENT_ROW_ID}; restart=${restart_count} ==="
  if [[ -n "${TRAINING_PID}" ]]; then
    kill -TERM -- "-${TRAINING_PID}" 2>/dev/null || true
  fi
  record_attempt_end "slurm_usr1_preemption"
  if (( restart_count < MAX_PREEMPT_REQUEUES )); then
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  echo "Preemption retry cap reached for ${CURRENT_ROW_ID}" >&2
  exit 89
}
trap request_requeue USR1

require_approved_node() {
  local node="${SLURMD_NODENAME:-${SLURM_NODELIST:-}}"
  if [[ -z "${node}" ]]; then
    echo "Cannot determine allocated node" >&2
    exit 87
  fi
  local denied
  for denied in ${NODE_DENYLIST//,/ }; do
    if [[ "${node}" == "${denied}" ]]; then
      echo "Refusing denylisted timing node ${node}" >&2
      exit 87
    fi
  done
  local node_info
  node_info="$(scontrol show node "${node}")"
  if [[ "${node_info}" != *"nvlink"* ]]; then
    echo "Refusing node without the nvlink feature: ${node}" >&2
    exit 87
  fi
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  if [[ "${#gpu_names[@]}" -ne 4 ]]; then
    echo "Expected four allocated GPUs, found ${#gpu_names[@]}" >&2
    exit 86
  fi
  local name
  for name in "${gpu_names[@]}"; do
    if [[ "${name}" != *"A6000"* ]]; then
      echo "Expected RTX A6000, found ${name}" >&2
      exit 86
    fi
  done
  echo "=== approved 4xA6000 NVLink node ${node} ==="
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,clocks.max.sm,power.limit --format=csv,noheader
  nvidia-smi topo -m
}

row_env="$(mktemp)"
load_row_env "${row_env}"
# shellcheck disable=SC1090
source "${row_env}"
rm -f "${row_env}"

if [[ "${ROW_CORRECTION_CAMPAIGN:-}" != "matrixpolicy_live_stats_20260712" ]]; then
  echo "Manifest row is not part of the correction campaign" >&2
  exit 2
fi
require_approved_node

run_dir="${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET}/${ROW_ROW_ID}"
jsonl="${run_dir}/${ROW_ACTIVATION}.jsonl"
if jsonl_terminal "${jsonl}" "${ROW_STEPS}" "${ROW_PHASE}"; then
  echo "=== row ${ROW_ROW_INDEX} ${ROW_ROW_ID} is already terminal ==="
  exit 0
fi
if [[ -f "${jsonl}" ]]; then
  stamp="$(date +%Y%m%d%H%M%S)"
  archive="${jsonl}.incomplete_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_${stamp}"
  echo "=== archiving incomplete attempt to ${archive} ==="
  mv "${jsonl}" "${archive}"
fi
mkdir -p "${run_dir}"
CURRENT_JSONL="${jsonl}"
CURRENT_ROW_ID="${ROW_ROW_ID}"

export E9_ROW_ID="${ROW_ROW_ID}"
export E9_ARM_ID="${ROW_ARM_ID:-}"
export E9_DESIGN_VERSION="${ROW_DESIGN_VERSION:-matrixpolicy-live-stat-syncfix-2026-07-12}"
export E9_MANIFEST_SHA256="${CAMPAIGN_MANIFEST_SHA256:?CAMPAIGN_MANIFEST_SHA256 is required}"
export E9_FREEZE_SHA256="${CAMPAIGN_FREEZE_SHA256:?CAMPAIGN_FREEZE_SHA256 is required}"
export E9_RUNTIME_FREEZE_SHA256="${CAMPAIGN_RUNTIME_SHA256:?CAMPAIGN_RUNTIME_SHA256 is required}"

row_extra=()
if [[ -n "${ROW_EXTRA_ARGS}" ]]; then
  read -r -a row_extra <<< "${ROW_EXTRA_ARGS}"
fi

command=(
  "${PYTHON}" -m torch.distributed.run
  --standalone
  --nproc_per_node=4
  "${RUNTIME_ROOT}/training/transformer_lm_compare.py"
  --activation "${ROW_ACTIVATION}"
  --optimizer "${ROW_OPTIMIZER}"
  --run-name "${ROW_ROW_ID}"
  --seed "${ROW_SEED}"
  --steps "${ROW_STEPS}"
  --eval-interval "${ROW_EVAL_INTERVAL}"
  --eval-batches "${ROW_EVAL_BATCHES}"
  --log-interval 10
  --dataset-name "${ROW_DATASET_NAME}"
  --dataset-config "${ROW_DATASET_CONFIG}"
  --dataset-streaming
  --dataset-text-column "${ROW_TEXT_COLUMN}"
  --train-split "${ROW_TRAIN_SPLIT}"
  --validation-split "${ROW_VAL_SPLIT}"
  --validation-skip-tokens "${ROW_VAL_SKIP_TOKENS}"
  --cache-dir "${TOKEN_CACHE_DIR}/${ROW_DATASET}"
  --hf-cache "${HF_CACHE_DIR}"
  --output-dir "${OUTPUT_ROOT}/${ROW_PHASE}/${ROW_DATASET}"
  --max-train-tokens "${ROW_TRAIN_TOKENS}"
  --max-val-tokens "${ROW_VAL_TOKENS}"
  --batch-size "${ROW_BATCH_SIZE}"
  --grad-accum "${ROW_GRAD_ACCUM}"
  --seq-len "${ROW_SEQ_LEN}"
  --layers "${ROW_LAYERS}"
  --d-model "${ROW_D_MODEL}"
  --heads "${ROW_HEADS}"
  --ffn-dim "${ROW_FFN_DIM}"
  --lr "${ROW_LR}"
  --min-lr "${ROW_MIN_LR}"
  --weight-decay "${ROW_WEIGHT_DECAY}"
  --probe-batch-size 1
  --matrix-spectrum-interval 250
)
command+=("${row_extra[@]}")

echo "=== correction row=${ROW_ROW_INDEX}; id=${ROW_ROW_ID}; phase=${ROW_PHASE}; dataset=${ROW_DATASET}; seed=${ROW_SEED} ==="
set +e
setsid "${command[@]}" &
TRAINING_PID=$!
wait "${TRAINING_PID}"
status=$?
TRAINING_PID=""
set -e
if (( status != 0 )); then
  record_attempt_end "training_exit_${status}"
  exit "${status}"
fi
if ! jsonl_terminal "${jsonl}" "${ROW_STEPS}" "${ROW_PHASE}"; then
  echo "Training exited zero without a terminal JSONL" >&2
  exit 85
fi
echo "=== completed ${ROW_ROW_ID} at $(date -Is) ==="
