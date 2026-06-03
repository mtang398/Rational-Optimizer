#!/bin/bash
#SBATCH --job-name=iclr-protocol-lock
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/mt872/rationalOPT/experiments/runs/logs/%x-%j.out

set -euo pipefail

cd /home/mt872/rationalOPT

export RATIONAL_OPT_TORCH_FALLBACK="${RATIONAL_OPT_TORCH_FALLBACK:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export HF_HOME="${PWD}/experiments/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export PATH="${PWD}/.venv-cu128/bin:${PATH}"

PYTHON="${PYTHON:-${PWD}/.venv-cu128/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/iclr26_phase1_protocol_lock}"
RUN_SUFFIX="${RUN_SUFFIX:-phase1_protocol_lock}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-experiments/cache/tokens_iclr26_phase1_protocol_lock}"
RLB_ACTIVATION="${RLB_ACTIVATION:-rlb_fused_fixed_strong_ffn}"
TASKS="${TASKS:-dclm fineweb_edu}"
OPTIMIZER_FAMILIES="${OPTIMIZER_FAMILIES:-adamw muon lion ademamix schedule_free_adamw adafactor_came soap_adamw rational_matrix_policy_onpolicy}"
PROTOCOL_STAGE="${PROTOCOL_STAGE:-surface}"
SEEDS="${SEEDS:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
MAX_EVAL_INTERVAL="${MAX_EVAL_INTERVAL:-50}"
MAX_REPO_GIB="${MAX_REPO_GIB:-190}"
MAX_CONFIGS="${MAX_CONFIGS:-0}"
CONFIG_START="${CONFIG_START:-0}"
CONFIG_LIMIT="${CONFIG_LIMIT:-${MAX_CONFIGS}}"
CONFIRM_ICLR_PHASE1="${CONFIRM_ICLR_PHASE1:-0}"

if [[ "${PROTOCOL_STAGE}" == "confirm" ]]; then
  STEPS="${STEPS:-3050}"
  MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-100000000}"
  MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-4000000}"
else
  STEPS="${STEPS:-1525}"
  MAX_TRAIN_TOKENS="${MAX_TRAIN_TOKENS:-50000000}"
  MAX_VAL_TOKENS="${MAX_VAL_TOKENS:-4000000}"
fi

LRS="${LRS:-0.0001 0.0002 0.0003 0.0005}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-0.03 0.10 0.20}"
ADAMW_LRS="${ADAMW_LRS:-${LRS}}"
MUON_LRS="${MUON_LRS:-${LRS}}"
LION_LRS="${LION_LRS:-0.00003 0.00006 0.0001 0.0002}"
ADEMAMIX_LRS="${ADEMAMIX_LRS:-${LRS}}"
SCHEDULE_FREE_LRS="${SCHEDULE_FREE_LRS:-${LRS}}"
CAME_LRS="${CAME_LRS:-${LRS}}"
SOAP_LRS="${SOAP_LRS:-${LRS}}"
MATRIX_POLICY_LRS="${MATRIX_POLICY_LRS:-${LRS}}"
ADAMW_WEIGHT_DECAYS="${ADAMW_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
MUON_WEIGHT_DECAYS="${MUON_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
LION_WEIGHT_DECAYS="${LION_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
ADEMAMIX_WEIGHT_DECAYS="${ADEMAMIX_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
SCHEDULE_FREE_WEIGHT_DECAYS="${SCHEDULE_FREE_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
CAME_WEIGHT_DECAYS="${CAME_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
SOAP_WEIGHT_DECAYS="${SOAP_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
MATRIX_POLICY_WEIGHT_DECAYS="${MATRIX_POLICY_WEIGHT_DECAYS:-${WEIGHT_DECAYS}}"
MUON_MOMENTA="${MUON_MOMENTA:-0.90 0.95}"
ADEMAMIX_ALPHAS="${ADEMAMIX_ALPHAS:-2.0 5.0 8.0}"
ADEMAMIX_BETA3S="${ADEMAMIX_BETA3S:-0.999 0.9999}"
ADEMAMIX_BETA2="${ADEMAMIX_BETA2:-0.999}"
SCHEDULE_FREE_BETA1S="${SCHEDULE_FREE_BETA1S:-0.90 0.95}"
CAME_CONFIDENCE_SCALES="${CAME_CONFIDENCE_SCALES:-0.5 1.0 2.0}"
SOAP_FREQS="${SOAP_FREQS:-10 50 100}"
SOAP_ONE_SIDED_VALUES="${SOAP_ONE_SIDED_VALUES:-false true}"
MATRIX_ADAM_LR_SCALES="${MATRIX_ADAM_LR_SCALES:-2.0 3.0 4.0}"
MATRIX_GROUP_GAINS="${MATRIX_GROUP_GAINS:-0.0 0.20}"
COMMON_EXTRA_ARGS="${COMMON_EXTRA_ARGS:-}"
BUILD_EXT="${BUILD_EXT:-1}"

request_requeue() {
  echo "=== received USR1 at $(date -Is); requesting requeue for job ${SLURM_JOB_ID:-manual} ==="
  if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol requeue "${SLURM_JOB_ID}" || true
  fi
  exit 0
}
trap request_requeue USR1

check_eval_density() {
  if (( EVAL_INTERVAL > MAX_EVAL_INTERVAL )); then
    echo "EVAL_INTERVAL=${EVAL_INTERVAL} is too sparse for paper curves; max allowed is ${MAX_EVAL_INTERVAL}." >&2
    exit 2
  fi
}

check_repo_size() {
  local used_kib
  used_kib="$(du -s "${PWD}" | awk '{print $1}')"
  local limit_kib=$((MAX_REPO_GIB * 1024 * 1024))
  if (( used_kib > limit_kib )); then
    echo "Repo is above ${MAX_REPO_GIB} GiB cap; refusing to continue. Used KiB=${used_kib}" >&2
    exit 80
  fi
}

family_lrs() {
  case "$1" in
    adamw) echo "${ADAMW_LRS}" ;;
    muon) echo "${MUON_LRS}" ;;
    lion) echo "${LION_LRS}" ;;
    ademamix) echo "${ADEMAMIX_LRS}" ;;
    schedule_free_adamw) echo "${SCHEDULE_FREE_LRS}" ;;
    adafactor_came) echo "${CAME_LRS}" ;;
    soap_adamw) echo "${SOAP_LRS}" ;;
    rational_matrix_policy_onpolicy) echo "${MATRIX_POLICY_LRS}" ;;
    *) echo "${LRS}" ;;
  esac
}

family_weight_decays() {
  case "$1" in
    adamw) echo "${ADAMW_WEIGHT_DECAYS}" ;;
    muon) echo "${MUON_WEIGHT_DECAYS}" ;;
    lion) echo "${LION_WEIGHT_DECAYS}" ;;
    ademamix) echo "${ADEMAMIX_WEIGHT_DECAYS}" ;;
    schedule_free_adamw) echo "${SCHEDULE_FREE_WEIGHT_DECAYS}" ;;
    adafactor_came) echo "${CAME_WEIGHT_DECAYS}" ;;
    soap_adamw) echo "${SOAP_WEIGHT_DECAYS}" ;;
    rational_matrix_policy_onpolicy) echo "${MATRIX_POLICY_WEIGHT_DECAYS}" ;;
    *) echo "${WEIGHT_DECAYS}" ;;
  esac
}

task_spec() {
  local task="$1"
  case "${task}" in
    dclm)
      DATASET_NAME="mlfoundations/dclm-baseline-1.0"
      DATASET_CONFIG="none"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_TOKENS="${DCLM_VAL_SKIP_TOKENS:-110000000}"
      ;;
    fineweb_edu)
      DATASET_NAME="HuggingFaceFW/fineweb-edu"
      DATASET_CONFIG="sample-10BT"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_TOKENS="${FINEWEB_EDU_VAL_SKIP_TOKENS:-110000000}"
      ;;
    fineweb)
      DATASET_NAME="HuggingFaceFW/fineweb"
      DATASET_CONFIG="sample-10BT"
      TEXT_COLUMN="text"
      TRAIN_SPLIT="train"
      VAL_SPLIT="train"
      VAL_SKIP_TOKENS="${FINEWEB_VAL_SKIP_TOKENS:-110000000}"
      ;;
    *)
      echo "Unknown TASK '${task}'. Valid: dclm fineweb_edu fineweb" >&2
      exit 2
      ;;
  esac
}

jsonl_complete() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  "${PYTHON}" - "${path}" "${STEPS}" <<'PYJSON'
import json
import sys
path = sys.argv[1]
steps = int(sys.argv[2])
summary_complete = False
early_stopped = False
last_eval_step = None
with open(path) as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "summary":
            if bool(record.get("stopped_early")):
                early_stopped = True
            if int(record.get("steps", -1)) == steps and int(record.get("completed_steps", record.get("steps", -1))) == steps:
                summary_complete = True
        if record.get("event") == "stopped_early":
            early_stopped = True
        if record.get("event") == "eval":
            last_eval_step = record.get("step")
sys.exit(0 if early_stopped or (summary_complete and int(last_eval_step or -1) == steps) else 1)
PYJSON
}

archive_incomplete_jsonl() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    return 0
  fi
  local stamp
  stamp="$(date +%Y%m%d%H%M%S)"
  local archive_path="${path}.incomplete_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}_${stamp}"
  echo "=== archiving incomplete ${path} -> ${archive_path} ==="
  mv "${path}" "${archive_path}"
}

run_config() {
  local task="$1"
  local tag="$2"
  local optimizer="$3"
  local activations="$4"
  local lr="$5"
  local wd="$6"
  local extra_args="${7:-}"
  local run_name="${task}_${tag}_lr${lr}_wd${wd}_${RUN_SUFFIX}"
  local run_dir="${OUTPUT_ROOT}/${task}/${run_name}"
  local task_cache_dir="${TOKEN_CACHE_DIR}/${task}"
  local pending=""
  local activation

  check_repo_size
  for activation in ${activations}; do
    if jsonl_complete "${run_dir}/${activation}.jsonl"; then
      echo "=== ${run_name}/${activation} already complete; skipping ==="
    else
      archive_incomplete_jsonl "${run_dir}/${activation}.jsonl"
      pending="${pending} ${activation}"
    fi
  done
  pending="${pending# }"
  if [[ -z "${pending}" ]]; then
    return 0
  fi

  echo "=== task=${task}; optimizer=${optimizer}; activations=${pending}; lr=${lr}; wd=${wd}; tag=${tag} ==="
  RUN_NAME="${run_name}" \
  STEPS="${STEPS}" \
  SEEDS="${SEEDS}" \
  OPTIMIZERS="${optimizer}" \
  ACTIVATIONS="${pending}" \
  EVAL_INTERVAL="${EVAL_INTERVAL}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  SKIP_BUILD_EXT="1" \
  EXTRA_ARGS="--dataset-name ${DATASET_NAME} --dataset-config ${DATASET_CONFIG} --dataset-streaming --dataset-text-column ${TEXT_COLUMN} --train-split ${TRAIN_SPLIT} --validation-split ${VAL_SPLIT} --validation-skip-tokens ${VAL_SKIP_TOKENS} --cache-dir ${task_cache_dir} --output-dir ${OUTPUT_ROOT}/${task} --max-train-tokens ${MAX_TRAIN_TOKENS} --max-val-tokens ${MAX_VAL_TOKENS} --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} --lr ${lr} --weight-decay ${wd} --probe-batch-size 1 --matrix-spectrum-interval 250 --early-stop-min-step ${EARLY_STOP_MIN_STEP:-250} --early-stop-max-val-loss ${EARLY_STOP_MAX_VAL_LOSS:-20.0} --early-stop-loss-increase ${EARLY_STOP_LOSS_INCREASE:-1.0} ${COMMON_EXTRA_ARGS} ${extra_args}" \
  bash training/run_wikitext103_optimizer_sweep.sbatch
}

planned_configs() {
  local count=0
  local task family lr wd value value2 lrs weight_decays
  for task in ${TASKS}; do
    for family in ${OPTIMIZER_FAMILIES}; do
      lrs="$(family_lrs "${family}")"
      weight_decays="$(family_weight_decays "${family}")"
      for lr in ${lrs}; do
        for wd in ${weight_decays}; do
          case "${family}" in
            adamw|lion)
              count=$((count + 1))
              ;;
            muon)
              for value in ${MUON_MOMENTA}; do count=$((count + 1)); done
              ;;
            ademamix)
              for value in ${ADEMAMIX_ALPHAS}; do for value2 in ${ADEMAMIX_BETA3S}; do count=$((count + 1)); done; done
              ;;
            schedule_free_adamw)
              for value in ${SCHEDULE_FREE_BETA1S}; do count=$((count + 1)); done
              ;;
            adafactor_came)
              for value in ${CAME_CONFIDENCE_SCALES}; do count=$((count + 1)); done
              ;;
            soap_adamw)
              for value in ${SOAP_FREQS}; do for value2 in ${SOAP_ONE_SIDED_VALUES}; do count=$((count + 1)); done; done
              ;;
            rational_matrix_policy_onpolicy)
              for value in ${MATRIX_ADAM_LR_SCALES}; do for value2 in ${MATRIX_GROUP_GAINS}; do count=$((count + 1)); done; done
              ;;
          esac
        done
      done
    done
  done
  echo "${count}"
}

should_run_config() {
  CONFIGS_SEEN=$((CONFIGS_SEEN + 1))
  local zero_based=$((CONFIGS_SEEN - 1))
  if (( zero_based < CONFIG_START )); then
    return 1
  fi
  if (( CONFIG_LIMIT > 0 && CONFIGS_STARTED >= CONFIG_LIMIT )); then
    echo "=== CONFIG_LIMIT=${CONFIG_LIMIT} reached after CONFIG_START=${CONFIG_START}; stopping cleanly ==="
    "${PYTHON}" experiments/scripts/summarize_iclr_phase1_protocol_lock.py --run-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/summary" || true
    exit 0
  fi
  CONFIGS_STARTED=$((CONFIGS_STARTED + 1))
  return 0
}

if [[ "${CONFIRM_ICLR_PHASE1}" != "1" ]]; then
  echo "Refusing to start Phase 1 protocol lock without CONFIRM_ICLR_PHASE1=1."
  echo "Planned configs: $(planned_configs). One job uses 4 A6000s; submit at most two active jobs for the 8-GPU cap."
  echo "Use OPTIMIZER_FAMILIES, family-specific *_LRS/*_WEIGHT_DECAYS, and CONFIG_START/CONFIG_LIMIT for bounded chunks."
  exit 2
fi

mkdir -p experiments/runs/logs
check_eval_density
check_repo_size
if [[ "${BUILD_EXT}" == "1" ]]; then
  BUILD_EXT_ROOT="${OUTPUT_ROOT}/_build/${SLURM_JOB_ID:-manual}"
  mkdir -p "${BUILD_EXT_ROOT}/temp" "${BUILD_EXT_ROOT}/lib"
  "${PYTHON}" setup.py build_ext --inplace --build-temp "${BUILD_EXT_ROOT}/temp" --build-lib "${BUILD_EXT_ROOT}/lib"
fi
CONFIGS_SEEN=0
CONFIGS_STARTED=0

echo "=== ICLR Phase 1 protocol lock ${SLURM_JOB_ID:-manual}; stage=${PROTOCOL_STAGE}; planned=$(planned_configs); chunk_start=${CONFIG_START}; chunk_limit=${CONFIG_LIMIT}; one job uses 4 A6000s ==="
for task in ${TASKS}; do
  task_spec "${task}"
  for family in ${OPTIMIZER_FAMILIES}; do
    FAMILY_LRS="$(family_lrs "${family}")"
    FAMILY_WEIGHT_DECAYS="$(family_weight_decays "${family}")"
    for lr in ${FAMILY_LRS}; do
      for wd in ${FAMILY_WEIGHT_DECAYS}; do
        case "${family}" in
          adamw|lion)
            should_run_config || continue
            run_config "${task}" "${family}" "${family}" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}"
            ;;
          muon)
            for momentum in ${MUON_MOMENTA}; do
              should_run_config || continue
              run_config "${task}" "muon_m${momentum}" "muon" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}" "--muon-momentum ${momentum}"
            done
            ;;
          ademamix)
            for alpha in ${ADEMAMIX_ALPHAS}; do
              for beta3 in ${ADEMAMIX_BETA3S}; do
                should_run_config || continue
                run_config "${task}" "ademamix_b2${ADEMAMIX_BETA2}_a${alpha}_b3${beta3}" "ademamix" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}" "--beta2 ${ADEMAMIX_BETA2} --ademamix-alpha ${alpha} --ademamix-beta3 ${beta3}"
              done
            done
            ;;
          schedule_free_adamw)
            for beta1 in ${SCHEDULE_FREE_BETA1S}; do
              should_run_config || continue
              run_config "${task}" "schedule_free_b1${beta1}" "schedule_free_adamw" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}" "--schedule-free-beta1 ${beta1} --schedule-free-warmup-steps 0"
            done
            ;;
          adafactor_came)
            for confidence in ${CAME_CONFIDENCE_SCALES}; do
              should_run_config || continue
              run_config "${task}" "adafactor_came_c${confidence}" "adafactor_came" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}" "--came-confidence-scale ${confidence}"
            done
            ;;
          soap_adamw)
            for freq in ${SOAP_FREQS}; do
              for one_sided in ${SOAP_ONE_SIDED_VALUES}; do
                should_run_config || continue
                flag="--soap-precondition-frequency ${freq}"
                if [[ "${one_sided}" == "true" ]]; then
                  flag="${flag} --soap-one-sided"
                else
                  flag="${flag} --no-soap-one-sided"
                fi
                run_config "${task}" "soap_f${freq}_one${one_sided}" "soap_adamw" "silu ${RLB_ACTIVATION}" "${lr}" "${wd}" "${flag}"
              done
            done
            ;;
          rational_matrix_policy_onpolicy)
            for adam_scale in ${MATRIX_ADAM_LR_SCALES}; do
              for group_gain in ${MATRIX_GROUP_GAINS}; do
                should_run_config || continue
                run_config \
                  "${task}" \
                  "matrix_policy_as${adam_scale}_gg${group_gain}" \
                  "rational_matrix_policy_onpolicy" \
                  "${RLB_ACTIVATION}" \
                  "${lr}" \
                  "${wd}" \
                  "--rational-matrix-policy-backbone-optimizer adamw --rational-matrix-policy-adam-lr-scale ${adam_scale} --rational-matrix-policy-group-gain-strength ${group_gain} --rational-matrix-policy-group-pressure-strength 0.10 --rational-matrix-policy-group-activity-damping 0.20 --rational-matrix-policy-group-start 0.02 --rational-matrix-policy-group-end 0.30 --rational-matrix-policy-group-min-scale 0.75 --rational-matrix-policy-group-max-scale 1.35"
              done
            done
            ;;
          *)
            echo "Unknown optimizer family '${family}'" >&2
            exit 2
            ;;
        esac
      done
    done
  done
done

"${PYTHON}" experiments/scripts/summarize_iclr_phase1_protocol_lock.py --run-root "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}/summary"
