#!/bin/bash
#SBATCH --job-name=ract-wt103-opt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
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
OPTIMIZERS="${OPTIMIZERS:-adamw muon lion ademamix schedule_free_adamw adafactor_came soap_adamw rational_matrix_policy_onpolicy}"
SEEDS="${SEEDS:-1337}"
STEPS="${STEPS:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-25}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BUILD_EXT_ARGS="${BUILD_EXT_ARGS:-}"
SKIP_BUILD_EXT="${SKIP_BUILD_EXT:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ACTIVE_OPTIMIZERS="adamw muon lion ademamix schedule_free_adamw adafactor_came soap_adamw rational_matrix_policy_onpolicy matrixpolicyV7"

for optimizer in ${OPTIMIZERS}; do
  case " ${ACTIVE_OPTIMIZERS} " in
    *" ${optimizer} "*) ;;
    *)
      echo "Accepted optimizers: ${ACTIVE_OPTIMIZERS}" >&2
      exit 2
      ;;
  esac
done

if [[ "${SKIP_BUILD_EXT}" != "1" ]]; then
  BUILD_EXT_ROOT="${BUILD_EXT_ROOT:-${TMPDIR:-/tmp}/rationalopt_build_ext_${SLURM_JOB_ID:-manual}_$$}"
  mkdir -p "${BUILD_EXT_ROOT}"
  "${PYTHON}" setup.py build_ext --inplace --build-temp "${BUILD_EXT_ROOT}/temp" --build-lib "${BUILD_EXT_ROOT}/lib" ${BUILD_EXT_ARGS}
fi
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  "${PYTHON}" training/transformer_lm_compare.py \
    --prepare-only \
    --run-name "${RUN_NAME}" \
    ${EXTRA_ARGS}
fi

for seed in ${SEEDS}; do
  for optimizer in ${OPTIMIZERS}; do
    for activation in ${ACTIVATIONS}; do
      if [[ ( "${optimizer}" == rational_* || "${optimizer}" == matrixpolicyV* ) && "${activation}" != rlb_* ]]; then
        continue
      fi
      "${PYTHON}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        training/transformer_lm_compare.py \
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
