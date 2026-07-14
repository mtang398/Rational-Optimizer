#!/bin/bash
#SBATCH --job-name=e9-100m
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --constraint=nvlink
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/mt872/rationalOPT/abalation/logs/e9/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

if [[ -z "${E9_RUNTIME_FREEZE_SHA256:-}" ]]; then
  echo "Missing E9_RUNTIME_FREEZE_SHA256; refusing mutable-checkout execution." >&2
  exit 2
fi
python3 abalation/scripts/submit_e9_100m.py verify \
  --expected-runtime-freeze "${E9_RUNTIME_FREEZE_SHA256}"

export MANIFEST="${MANIFEST:-abalation/manifests/e9_100m_manifest.csv}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-abalation/runs/e9_100m}"
export CONFIRM_ICLR26_MANIFEST=1
export ROW_LIMIT=1
export BUILD_EXT="${BUILD_EXT:-0}"
export SKIP_PREPARE="${SKIP_PREPARE:-1}"
export E9_STRICT_GPU_METADATA=1
export TIMING_GUARD_MAX_SECONDS_PER_STEP="${TIMING_GUARD_MAX_SECONDS_PER_STEP:-1.20}"
export TIMING_GUARD_MIN_STEP="${TIMING_GUARD_MIN_STEP:-300}"

exec bash abalation/scripts/run_matrixpolicy_ablation_manifest_job.sh
