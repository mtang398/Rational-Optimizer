#!/bin/bash
#SBATCH --job-name=synth-dense
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
export EVAL_BATCHES="${EVAL_BATCHES:-10}"
export LOG_INTERVAL="${LOG_INTERVAL:-10}"
export RUN_SUFFIX="${RUN_SUFFIX:-20260529_dense_curve}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/synthetic_dense_curves_20260529}"
export SYNTHETIC_TASKS="${SYNTHETIC_TASKS:-synthetic/code synthetic/symbolic synthetic/reasoning_mix}"

echo "=== dense synthetic curves: eval_interval=${EVAL_INTERVAL}, eval_batches=${EVAL_BATCHES}, log_interval=${LOG_INTERVAL}, output=${OUTPUT_ROOT}, suffix=${RUN_SUFFIX} ==="
exec bash experiments/scripts/run_synthetic_fair_full_20260529.sh
