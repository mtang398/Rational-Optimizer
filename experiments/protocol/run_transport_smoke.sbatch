#!/bin/bash -p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=00:06:00
#SBATCH --export=NIL
#SBATCH --output=slurm-nccl-nvlink-smoke-%j.out

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"
python="${PYTHON:-${PWD}/.venv/bin/python}"
export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}"
export NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ALGO='Ring;allreduce:Tree' NCCL_DEBUG=INFO

# Preserve Slurm's CUDA-visible allocation namespace; SLURM_JOB_GPUS can use
# global physical IDs that are invalid inside the job cgroup.
original_cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
gpu_source="${original_cuda_visible_devices:-0,1,2,3}"
IFS=',' read -r -a allocated_gpu_ids <<<"${gpu_source}"
if test "${#allocated_gpu_ids[@]}" -lt 4; then
  echo "expected at least four allocated GPU identifiers, found ${gpu_source}" >&2
  exit 92
fi
export CUDA_VISIBLE_DEVICES="${allocated_gpu_ids[0]#gpu:},${allocated_gpu_ids[1]#gpu:},${allocated_gpu_ids[2]#gpu:},${allocated_gpu_ids[3]#gpu:}"
export CAMPAIGN_ORIGINAL_CUDA_VISIBLE_DEVICES="${original_cuda_visible_devices}"
export CAMPAIGN_SLURM_JOB_GPUS="${SLURM_JOB_GPUS:-}"
export CAMPAIGN_SELECTED_VISIBLE_GPU_IDS="${CUDA_VISIBLE_DEVICES}"

run_case() {
  local label="$1"
  local level="$2"
  local cumem="$3"
  local port="$4"
  if test "${level}" = default; then unset NCCL_P2P_LEVEL; else export NCCL_P2P_LEVEL="${level}"; fi
  if test "${cumem}" = default; then unset NCCL_CUMEM_ENABLE; else export NCCL_CUMEM_ENABLE="${cumem}"; fi
  printf 'NCCL_NVLINK_BEGIN %s level=%s cumem=%s\n' "${label}" "${level}" "${cumem}"
  set +e
  timeout --signal=TERM --kill-after=10 60 \
    "${python}" -m torch.distributed.run --standalone --master-port "${port}" \
    --nproc_per_node=4 -m experiments.protocol.smoke_nccl
  local status=$?
  set -e
  printf 'NCCL_NVLINK_END %s status=%s\n' "${label}" "${status}"
  return "${status}"
}

set +e
run_case nvlink_only NVL default 29611
nvlink_status=$?
run_case legacy_cuda_ipc default 0 29612
legacy_status=$?
set -e
if test "${nvlink_status}" -ne 0 && test "${legacy_status}" -ne 0; then
  echo "both NCCL transport probes failed" >&2
  exit 1
fi
