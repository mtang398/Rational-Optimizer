#!/bin/bash -p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=00:06:00
#SBATCH --export=NIL
#SBATCH --output=/home/mt872/rationalOPT/experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/logs/nccl-nvlink-smoke-%j.out

set -euo pipefail
cd /home/mt872/rationalOPT
python="${PWD}/.venv-cu128/bin/python"
export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}:${PWD}/.venv-cu128/lib/python3.12/site-packages:/home/mt872/.local/lib/python3.12/site-packages"
export NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ALGO='Ring;allreduce:Tree' NCCL_DEBUG=INFO

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
    --nproc_per_node=4 -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.smoke_nccl
  local status=$?
  set -e
  printf 'NCCL_NVLINK_END %s status=%s\n' "${label}" "${status}"
}

run_case nvlink_only NVL default 29611
run_case legacy_cuda_ipc default 0 29612
