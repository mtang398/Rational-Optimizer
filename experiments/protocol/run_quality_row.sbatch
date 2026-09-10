#!/bin/bash -p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-14%3
#SBATCH --no-requeue
#SBATCH --export=NIL
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
if test -n "${SLURM_SUBMIT_DIR:-}"; then
  repo_root="${SLURM_SUBMIT_DIR}"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/../.." && pwd)"
fi
cd "${repo_root}"
if ! test -f experiments/protocol/SOURCE_FREEZE.sha256; then
  echo "run from the repository root or submit with SLURM_SUBMIT_DIR set to it" >&2
  exit 90
fi
package="experiments/protocol"
python="${PYTHON:-${PWD}/.venv/bin/python}"
ordinal="${SLURM_ARRAY_TASK_ID:?missing suite row ordinal}"
suite="${1:-18l_1024d_100m_tokens_3050_steps}"
candidate_optimizer="${2:-tiller_v1}"

export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}"
export RATIONAL_OPT_TORCH_FALLBACK=0
export NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL NCCL_SHM_DISABLE=0
export NCCL_ALGO="Ring;allreduce:Tree" NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets" TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TILLER_STRICT_GPU_METADATA=1
export TILLER_ARM_ID="${candidate_optimizer}"

index="$("${python}" -m experiments.protocol.row_tools matrix-index \
  --suite "${suite}" --optimizer "${candidate_optimizer}" --ordinal "${ordinal}")"
output_root="${PWD}/${package}/runs"
control_root="${PWD}/experiments/runs/activation_optimizer"
result_root="${PWD}/${package}/results/row-$(printf '%02d' "${index}")"

mkdir -p "${package}/logs" "${output_root}" "${result_root}" "${package}/failed_attempts"
exec 9>"${result_root}/execution.lock"
if ! flock -n 9; then
  echo "matrix row ${index} is already executing" >&2
  exit 91
fi
gpu_selection_env="${result_root}/GPU_SELECTION.env"
gpu_selection_json="${result_root}/GPU_SELECTION.json"
"${python}" -m experiments.protocol.select_nvlink_gpus \
  --output-env "${gpu_selection_env}" \
  --output-json "${gpu_selection_json}"
. "${gpu_selection_env}"

"${python}" -m experiments.protocol.verify_repository --quick
sha256sum -c "${package}/SOURCE_FREEZE.sha256"
"${python}" -m experiments.protocol.audit_runtime_hardware \
  --output "${result_root}/HARDWARE.json"
"${python}" -c 'from rational_opt import _C; assert hasattr(_C, "local_basis_restricted_backward"); assert hasattr(_C, "local_basis_affine_statistics"); print(_C.__file__)'

mapfile -t row < <("${python}" -m experiments.protocol.row_tools \
  describe --matrix-index "${index}" --output-root "${output_root}" \
  --control-root "${control_root}")
model="${row[0]}"
dataset="${row[1]}"
seed="${row[2]}"
steps="${row[3]}"
# Keep the same observation cadence across both budgets and append the exact
# endpoint. The trainer rejects observation steps beyond the selected row.
case "${steps}" in
  3050)
    export TILLER_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050
    ;;
  9150)
    export TILLER_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050,3500,4000,5000,6000,7000,8000,9000,9150
    ;;
  *)
    echo "unsupported endpoint for locked telemetry schedule: ${steps}" >&2
    exit 93
    ;;
esac
control_jsonl="${row[4]}"
candidate_jsonl="${row[5]}"
control_run="${row[6]}"
candidate_run="${row[7]}"
result_json="${result_root}/RESULT.json"
screen="${result_root}/STEP1000_SCREEN.json"
control_wall_clock="${control_jsonl%.jsonl}.wall_clock.json"
candidate_wall_clock="${result_root}/CANDIDATE_WALL_CLOCK.json"
candidate_stdout="${result_root}/CANDIDATE.stdout.log"
failure_stage="row_setup"
failure_json="${result_root}/FAILURE-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.json"

record_failure() {
  local code=$?
  trap - EXIT
  if test "${code}" -ne 0; then
    set +e
    "${python}" -m experiments.protocol.record_failure \
      --matrix-index "${index}" --stage "${failure_stage}" --exit-code "${code}" \
      --screen "${screen}" --output "${failure_json}"
    for log_path in "${candidate_stdout}"; do
      if test -f "${log_path}"; then gzip -f "${log_path}"; fi
    done
  fi
  exit "${code}"
}
trap record_failure EXIT

if test -s "${result_json}"; then
  "${python}" -m experiments.protocol.row_tools \
    terminal --path "${control_jsonl}" --steps "${steps}"
  "${python}" -m experiments.protocol.row_tools \
    terminal --path "${candidate_jsonl}" --steps "${steps}"
  "${python}" -m experiments.protocol.analyze_row \
    --matrix-index "${index}" --control "${control_jsonl}" \
    --candidate "${candidate_jsonl}" --output "${result_json}" \
    --control-wall-clock "${control_wall_clock}" \
    --candidate-wall-clock "${candidate_wall_clock}"
  sha256sum -c "${package}/SOURCE_FREEZE.sha256"
  exit 0
fi

archive_arm() {
  local jsonl="$1"
  local run="$2"
  local run_dir
  run_dir="$(dirname "${jsonl}")"
  if test -e "${run_dir}"; then
    local stamp
    stamp="$(date +%Y%m%d%H%M%S)"
    mv "${run_dir}" "${package}/failed_attempts/${run}-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}-${stamp}"
  fi
}
archive_arm "${candidate_jsonl}" "${candidate_run}"

failure_stage="control_reuse_validation"
"${python}" -m experiments.protocol.row_tools \
  terminal --path "${control_jsonl}" --steps "${steps}"
test -s "${control_wall_clock}"
set +e
"${python}" -m experiments.protocol.monitor_step1000 \
  --matrix-index "${index}" --control "${control_jsonl}" \
  --candidate "${candidate_jsonl}" --output "${screen}"
status=$?
set -e
if test "${status}" -ne 3; then
  exit "${status}"
fi

failure_stage="candidate_endpoint_step1000_screen"
setsid "${python}" -m experiments.protocol.run_timed \
  --output "${candidate_wall_clock}" --arm candidate -- \
  "${python}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m experiments.protocol.suite --matrix-index "${index}" \
  --arm candidate --output-root "${output_root}" \
  >"${candidate_stdout}" 2>&1 &
train_pid=$!
while kill -0 "${train_pid}" 2>/dev/null; do
  set +e
  "${python}" -m experiments.protocol.monitor_step1000 \
    --matrix-index "${index}" --control "${control_jsonl}" \
    --candidate "${candidate_jsonl}" --output "${screen}"
  status=$?
  set -e
  if test "${status}" -eq 0; then break; fi
  if test "${status}" -eq 4; then
    kill -TERM -- "-${train_pid}" 2>/dev/null || true
    wait "${train_pid}" || true
    echo "candidate interrupted: negative paired step-1,000 lead for ${model}/${dataset}/seed${seed}" >&2
    exit 4
  fi
  if test "${status}" -ne 3; then
    kill -TERM -- "-${train_pid}" 2>/dev/null || true
    wait "${train_pid}" || true
    exit "${status}"
  fi
  sleep 20
done
wait "${train_pid}"
"${python}" -m experiments.protocol.row_tools \
  terminal --path "${candidate_jsonl}" --steps "${steps}"
"${python}" -m experiments.protocol.monitor_step1000 \
  --matrix-index "${index}" --control "${control_jsonl}" \
  --candidate "${candidate_jsonl}" --output "${screen}"
failure_stage="endpoint_analysis"
"${python}" -m experiments.protocol.analyze_row \
  --matrix-index "${index}" --control "${control_jsonl}" \
  --candidate "${candidate_jsonl}" --output "${result_json}" \
  --control-wall-clock "${control_wall_clock}" \
  --candidate-wall-clock "${candidate_wall_clock}"
failure_stage="post_run_source_freeze"
sha256sum -c "${package}/SOURCE_FREEZE.sha256"
gzip -f "${candidate_stdout}"
