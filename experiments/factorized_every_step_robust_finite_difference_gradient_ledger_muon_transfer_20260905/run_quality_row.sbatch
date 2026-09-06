#!/bin/bash -p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-23%2
#SBATCH --requeue
#SBATCH --export=NIL
#SBATCH --output=/home/mt872/rationalOPT/experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/logs/%x-%A_%a.out

set -euo pipefail
cd /home/mt872/rationalOPT
package="experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905"
python="${PWD}/.venv-cu128/bin/python"
index="${SLURM_ARRAY_TASK_ID:?missing matrix index}"
output_root="${PWD}/${package}/runs"
result_root="${PWD}/${package}/results/row-$(printf '%02d' "${index}")"

export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}:${PWD}/.venv-cu128/lib/python3.12/site-packages:/home/mt872/.local/lib/python3.12/site-packages"
export RATIONAL_OPT_TORCH_FALLBACK=0
export NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=NVL NCCL_SHM_DISABLE=0
export NCCL_ALGO="Ring;allreduce:Tree" NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets" TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 E9_STRICT_GPU_METADATA=1
export E9_ARM_ID="factorized_every_step_rfd_gradient_ledger_muon_v1"

# On this cluster an exclusive request can expose every physical device on an
# eight-GPU host.  Restrict the process to the first four Slurm allocation IDs.
gpu_source="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a allocated_gpu_ids <<<"${gpu_source}"
if test "${#allocated_gpu_ids[@]}" -lt 4; then
  echo "expected at least four allocated GPU identifiers, found ${gpu_source}" >&2
  exit 92
fi
export CUDA_VISIBLE_DEVICES="${allocated_gpu_ids[0]#gpu:},${allocated_gpu_ids[1]#gpu:},${allocated_gpu_ids[2]#gpu:},${allocated_gpu_ids[3]#gpu:}"
export CAMPAIGN_SELECTED_PHYSICAL_GPU_IDS="${CUDA_VISIBLE_DEVICES}"

mkdir -p "${package}/logs" "${output_root}" "${result_root}" "${package}/failed_attempts"
exec 9>"${result_root}/execution.lock"
if ! flock -n 9; then
  echo "matrix row ${index} is already executing" >&2
  exit 91
fi

sha256sum -c "${package}/SOURCE_FREEZE.sha256"
"${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.audit_runtime_hardware \
  --output "${result_root}/HARDWARE.json"
"${python}" -c 'from rational_opt import _C; assert hasattr(_C, "local_basis_restricted_backward"); assert hasattr(_C, "local_basis_affine_statistics"); print(_C.__file__)'

mapfile -t row < <("${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.row_tools \
  describe --matrix-index "${index}" --output-root "${output_root}")
model="${row[0]}"
dataset="${row[1]}"
seed="${row[2]}"
steps="${row[3]}"
# Keep the same locked observation cadence at both scales, with the exact
# endpoint appended.  The trainer rejects observation steps beyond the row's
# endpoint, so M0 must not inherit the M1-only checkpoints.
case "${steps}" in
  3050)
    export E9_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050
    ;;
  9150)
    export E9_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050,3500,4000,5000,6000,7000,8000,9000,9150
    ;;
  *)
    echo "unsupported endpoint for locked telemetry schedule: ${steps}" >&2
    exit 93
    ;;
esac
candidate_jsonl="${row[5]}"
candidate_run="${row[7]}"
candidate_stdout="${result_root}/CANDIDATE.stdout.log"
failure_stage="row_setup"
failure_json="${result_root}/FAILURE-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}.json"

record_failure() {
  local code=$?
  trap - EXIT
  if test "${code}" -ne 0; then
    set +e
    "${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.record_failure \
      --matrix-index "${index}" --stage "${failure_stage}" --exit-code "${code}" \
      --output "${failure_json}"
    for log_path in "${candidate_stdout}"; do
      if test -f "${log_path}"; then gzip -f "${log_path}"; fi
    done
  fi
  exit "${code}"
}
trap record_failure EXIT

if "${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.row_tools \
    terminal --path "${candidate_jsonl}" --steps "${steps}"; then
  "${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.row_tools \
    terminal --path "${candidate_jsonl}" --steps "${steps}"
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

failure_stage="candidate_endpoint"
"${python}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.suite \
  --matrix-index "${index}" --arm candidate --output-root "${output_root}" \
  >"${candidate_stdout}" 2>&1
"${python}" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.row_tools \
  terminal --path "${candidate_jsonl}" --steps "${steps}"
failure_stage="post_run_source_freeze"
sha256sum -c "${package}/SOURCE_FREEZE.sha256"
gzip -f "${candidate_stdout}"
