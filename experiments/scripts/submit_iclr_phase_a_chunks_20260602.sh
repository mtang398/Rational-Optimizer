#!/bin/bash
set -euo pipefail

cd /home/mt872/rationalOPT

SCRIPT="experiments/scripts/run_iclr_phase_a_hpo_20260602.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/runs/iclr_phase_a_hpo_20260602_ref}"
CHUNK_TIME="${CHUNK_TIME:-12:00:00}"
CORE_CHUNK="${CORE_CHUNK:-8}"
ADAPTIVE_CHUNK="${ADAPTIVE_CHUNK:-6}"
CORE_TOTAL="${CORE_TOTAL:-72}"
ADAPTIVE_TOTAL="${ADAPTIVE_TOTAL:-120}"
CONFIRM_ICLR_PHASE_A="${CONFIRM_ICLR_PHASE_A:-0}"

if [[ "${CONFIRM_ICLR_PHASE_A}" != "1" ]]; then
  echo "Refusing to submit chunks without CONFIRM_ICLR_PHASE_A=1." >&2
  echo "This submits dependency chains; each job uses 4 A6000 GPUs and ${CHUNK_TIME} wall time." >&2
  exit 2
fi

submit_core_chunk() {
  local task="$1"
  local start="$2"
  local dep="$3"
  local args=(--time="${CHUNK_TIME}")
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  local output
  output=$(CONFIRM_ICLR_PHASE_A=1 \
    TASKS="${task}" \
    RUN_SUFFIX="phase_a_surface_core_ref_20260602" \
    HPO_FAMILIES="adamw muon lion soap_adamw" \
    ADAMW_LRS="0.0001 0.0002 0.0003 0.0005" \
    MUON_LRS="0.0001 0.0002 0.0003 0.0005" \
    LION_LRS="0.00003 0.00006 0.0001 0.0002" \
    SOAP_LRS="0.0001 0.0002 0.0003 0.0005" \
    WEIGHT_DECAYS="0.03 0.10 0.20" \
    MUON_MOMENTA="0.90 0.95" \
    SOAP_FREQS="50" \
    SOAP_ONE_SIDED_VALUES="false true" \
    CONFIG_START="${start}" \
    CONFIG_LIMIT="${CORE_CHUNK}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    sbatch "${args[@]}" "${SCRIPT}")
  echo "${output}" >&2
  echo "${output}" | awk '{print $4}'
}

submit_adaptive_chunk() {
  local task="$1"
  local start="$2"
  local dep="$3"
  local args=(--time="${CHUNK_TIME}")
  if [[ -n "${dep}" ]]; then
    args+=(--dependency="afterok:${dep}")
  fi
  local output
  output=$(CONFIRM_ICLR_PHASE_A=1 \
    TASKS="${task}" \
    RUN_SUFFIX="phase_a_surface_adaptive_ref_20260602" \
    HPO_FAMILIES="ademamix schedule_free_adamw adafactor_came rational_matrix_policy_onpolicy" \
    ADEMAMIX_LRS="0.0001 0.0002 0.0003 0.0005" \
    SCHEDULE_FREE_LRS="0.0001 0.0002 0.0003 0.0005" \
    CAME_LRS="0.0001 0.0002 0.0003 0.0005" \
    MATRIX_POLICY_LRS="0.0001 0.0002 0.0003 0.0005" \
    WEIGHT_DECAYS="0.03 0.10 0.20" \
    ADEMAMIX_ALPHAS="2.0 5.0" \
    ADEMAMIX_BETA3S="0.999 0.9999" \
    SCHEDULE_FREE_BETA1S="0.90" \
    CAME_CONFIDENCE_SCALES="1.0" \
    MATRIX_ADAM_LR_SCALES="2.0 3.0" \
    MATRIX_GROUP_GAINS="0.0 0.20" \
    CONFIG_START="${start}" \
    CONFIG_LIMIT="${ADAPTIVE_CHUNK}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    sbatch "${args[@]}" "${SCRIPT}")
  echo "${output}" >&2
  echo "${output}" | awk '{print $4}'
}

submit_lane() {
  local task="$1"
  local lane="$2"
  local total="$3"
  local chunk="$4"
  local dep="${5:-}"
  local start job
  for ((start=0; start<total; start+=chunk)); do
    if [[ "${lane}" == "core" ]]; then
      job=$(submit_core_chunk "${task}" "${start}" "${dep}")
    else
      job=$(submit_adaptive_chunk "${task}" "${start}" "${dep}")
    fi
    echo "${lane} ${task} start=${start} limit=${chunk} job=${job} dep=${dep:-none}"
    dep="${job}"
  done
  LAST_JOB="${dep}"
}

submit_lane "fineweb_edu" "core" "${CORE_TOTAL}" "${CORE_CHUNK}"
CORE_EDU_LAST="${LAST_JOB}"
submit_lane "fineweb" "core" "${CORE_TOTAL}" "${CORE_CHUNK}" "${CORE_EDU_LAST}"
CORE_LAST="${LAST_JOB}"

submit_lane "fineweb_edu" "adaptive" "${ADAPTIVE_TOTAL}" "${ADAPTIVE_CHUNK}"
ADAPTIVE_EDU_LAST="${LAST_JOB}"
submit_lane "fineweb" "adaptive" "${ADAPTIVE_TOTAL}" "${ADAPTIVE_CHUNK}" "${ADAPTIVE_EDU_LAST}"
ADAPTIVE_LAST="${LAST_JOB}"

echo "Submitted chunked Phase A chains. Final core job=${CORE_LAST}; final adaptive job=${ADAPTIVE_LAST}."
