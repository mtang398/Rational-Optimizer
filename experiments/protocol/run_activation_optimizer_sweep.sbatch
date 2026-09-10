#!/bin/bash -p
#SBATCH --job-name=activation-optimizer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --array=0-29%3
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

python="${PYTHON:-${PWD}/.venv/bin/python}"
ordinal="${SLURM_ARRAY_TASK_ID:?missing stage row ordinal}"
stage="${1:-muon}"
suite="${2:-18l_1024d_100m_tokens_3050_steps}"

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
case "${suite}" in
  *3050_steps)
    export TILLER_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050
    ;;
  *9150_steps)
    export TILLER_OPTIMIZER_TELEMETRY_STEPS=1,100,200,500,1000,1500,2000,2500,3000,3050,3500,4000,5000,6000,7000,8000,9000,9150
    ;;
  *)
    echo "unsupported endpoint for locked telemetry schedule: ${suite}" >&2
    exit 93
    ;;
esac

index="$("${python}" -m experiments.protocol.row_tools manifest-index \
  --suite "${suite}" --stage "${stage}" --ordinal "${ordinal}")"

gpu_source="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
IFS=',' read -r -a allocated_gpu_ids <<<"${gpu_source}"
if test "${#allocated_gpu_ids[@]}" -lt 4; then
  echo "expected at least four allocated GPU identifiers, found ${gpu_source}" >&2
  exit 92
fi
export CUDA_VISIBLE_DEVICES="${allocated_gpu_ids[0]#gpu:},${allocated_gpu_ids[1]#gpu:},${allocated_gpu_ids[2]#gpu:},${allocated_gpu_ids[3]#gpu:}"
export CAMPAIGN_SELECTED_PHYSICAL_GPU_IDS="${CUDA_VISIBLE_DEVICES}"

mkdir -p experiments/protocol/logs experiments/runs/activation_optimizer
"${python}" -m experiments.protocol.verify_repository --quick
sha256sum -c experiments/protocol/SOURCE_FREEZE.sha256
mkdir -p experiments/cache/build
exec 8>experiments/cache/build/grain-extension.lock
flock 8
"${python}" setup.py build_ext --inplace
flock -u 8
"${python}" -m experiments.protocol.audit_runtime_hardware \
  --output "experiments/runs/activation_optimizer/hardware-${SLURM_ARRAY_JOB_ID}-${index}.json"
"${python}" -m experiments.protocol.run_activation_row \
  --row-index "${index}" --world-size 4
