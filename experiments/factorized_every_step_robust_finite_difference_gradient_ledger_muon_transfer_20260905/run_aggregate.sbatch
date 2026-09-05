#!/bin/bash -p
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/home/mt872/rationalOPT/experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/logs/%x-%j.out

set -euo pipefail
cd /home/mt872/rationalOPT
export PATH="/usr/local/slurm/current/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PWD}/activation:${PWD}:${PWD}/.venv-cu128/lib/python3.12/site-packages:/home/mt872/.local/lib/python3.12/site-packages"
sha256sum -c experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/SOURCE_FREEZE.sha256
"${PWD}/.venv-cu128/bin/python" -m experiments.factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905.analyze_all \
  --output experiments/factorized_every_step_robust_finite_difference_gradient_ledger_muon_transfer_20260905/RESULTS.json
