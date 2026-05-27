#!/bin/bash
#SBATCH --job-name=ract-wt103-opt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export PYTHONPATH="${PWD}/activation:${PYTHONPATH:-}"
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MAX_JOBS=8
export PATH="${PWD}/.venv-cu128/bin:${PATH}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

RUN_NAME="${RUN_NAME:-optimizer_sweep_${SLURM_JOB_ID}}"
PYTHON="${PWD}/.venv-cu128/bin/python"
ACTIVATIONS="${ACTIVATIONS:-silu rlb_fused_fixed_strong_ffn rlb_fused_fixed_strong_h2880_ffn}"
OPTIMIZERS="${OPTIMIZERS:-adamw muon rational_onpolicy_balance rational_quotient_onpolicy rational_jacobian_onpolicy rational_quotient_jacobian_onpolicy rational_adaptive_metric_onpolicy}"
SEEDS="${SEEDS:-1337}"
STEPS="${STEPS:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ACTIVE_OPTIMIZERS="adamw muon rational_onpolicy_balance rational_quotient_onpolicy rational_jacobian_onpolicy rational_quotient_jacobian_onpolicy rational_adaptive_metric_onpolicy"

for optimizer in ${OPTIMIZERS}; do
  case " ${ACTIVE_OPTIMIZERS} " in
    *" ${optimizer} "*) ;;
    *)
      echo "Accepted optimizers: ${ACTIVE_OPTIMIZERS}" >&2
      exit 2
      ;;
  esac
done

"${PYTHON}" setup.py build_ext --inplace
"${PYTHON}" training/transformer_wikitext103_compare.py \
  --prepare-only \
  --run-name "${RUN_NAME}"

for seed in ${SEEDS}; do
  for optimizer in ${OPTIMIZERS}; do
    for activation in ${ACTIVATIONS}; do
      if [[ "${optimizer}" == rational_* && "${activation}" != rlb_* ]]; then
        continue
      fi
      "${PYTHON}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        training/transformer_wikitext103_compare.py \
        --activation "${activation}" \
        --optimizer "${optimizer}" \
        --run-name "${RUN_NAME}" \
        --seed "${seed}" \
        --steps "${STEPS}" \
        --eval-interval "${EVAL_INTERVAL}" \
        --eval-batches "${EVAL_BATCHES}" \
        --log-interval "${LOG_INTERVAL}" \
        ${EXTRA_ARGS}
    done
  done
done
