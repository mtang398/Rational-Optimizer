#!/bin/bash -p
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=nvlink
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/home/mt872/rationalOPT/experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/logs/nccl-smoke-%j.out

set -euo pipefail
cd /home/mt872/rationalOPT
package="experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905"
python="${PWD}/.venv-cu128/bin/python"
export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}:${PWD}/.venv-cu128/lib/python3.12/site-packages:/home/mt872/.local/lib/python3.12/site-packages"
export NCCL_SHM_DISABLE=0 TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO

run_case() {
  local label="$1"
  local p2p="$2"
  local algorithm="$3"
  local port="$4"
  export NCCL_P2P_DISABLE="${p2p}"
  if test "${algorithm}" = default; then
    unset NCCL_ALGO
  else
    export NCCL_ALGO="${algorithm}"
  fi
  printf 'NCCL_SMOKE_BEGIN %s p2p=%s algorithm=%s\n' "${label}" "${p2p}" "${algorithm}"
  set +e
  timeout --signal=TERM --kill-after=10 60 \
    "${python}" -m torch.distributed.run --standalone --master-port "${port}" \
    --nproc_per_node=4 -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.smoke_nccl
  local status=$?
  set -e
  printf 'NCCL_SMOKE_END %s status=%s\n' "${label}" "${status}"
}

run_case p2p_custom 0 'Ring;allreduce:Tree' 29601
run_case p2p_default 0 default 29602
run_case no_p2p_custom 1 'Ring;allreduce:Tree' 29603
