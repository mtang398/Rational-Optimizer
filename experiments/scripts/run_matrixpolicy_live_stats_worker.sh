#!/bin/bash

set -euo pipefail

ROOT="/home/mt872/rationalOPT"
CAMPAIGN_ROOT="${ROOT}/experiments/corrections/matrixpolicy_live_stats_20260712"
RUNTIME_ROOT="${CAMPAIGN_ROOT}/runtime"
ROW_LAUNCHER="${MATRIXPOLICY_WORKER_ROW_LAUNCHER:-${RUNTIME_ROOT}/experiments/scripts/run_matrixpolicy_live_stats_correction_job.sh}"
PYTHON="${ROOT}/.venv-cu128/bin/python"

: "${MANIFEST:?MANIFEST is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${ROW_INDICES:?ROW_INDICES is required}"
: "${CAMPAIGN_FREEZE_SHA256:?CAMPAIGN_FREEZE_SHA256 is required}"
: "${CAMPAIGN_RUNTIME_SHA256:?CAMPAIGN_RUNTIME_SHA256 is required}"
: "${CAMPAIGN_MANIFEST_SHA256:?CAMPAIGN_MANIFEST_SHA256 is required}"

MAX_PREEMPT_REQUEUES="${MAX_PREEMPT_REQUEUES:-8}"
CURRENT_CHILD=""
CURRENT_ROW=""

if (( ${SLURM_RESTART_COUNT:-0} > MAX_PREEMPT_REQUEUES )); then
  echo "Worker preemption retry cap exceeded: ${SLURM_RESTART_COUNT}" >&2
  exit 89
fi

request_requeue() {
  echo "=== worker received USR1 at $(date -Is); row=${CURRENT_ROW:-between_rows}; restart=${SLURM_RESTART_COUNT:-0} ==="
  if [[ -n "${CURRENT_CHILD}" ]]; then
    kill -USR1 "${CURRENT_CHILD}" 2>/dev/null || true
    wait "${CURRENT_CHILD}" 2>/dev/null || true
  else
    scontrol requeue "${SLURM_JOB_ID}"
  fi
  exit 0
}
trap request_requeue USR1

export RATIONALOPT_WORKSPACE_ROOT="${ROOT}"
export PYTHONDONTWRITEBYTECODE=1
"${PYTHON}" "${RUNTIME_ROOT}/experiments/scripts/build_matrixpolicy_live_stats_correction.py" verify-runtime >/dev/null

IFS=':' read -r -a rows <<< "${ROW_INDICES}"
if (( ${#rows[@]} == 0 )); then
  echo "Worker received no row indices" >&2
  exit 2
fi

for row in "${rows[@]}"; do
  if [[ ! "${row}" =~ ^[0-9]+$ ]]; then
    echo "Invalid worker row index: ${row}" >&2
    exit 2
  fi
  CURRENT_ROW="${row}"
  echo "=== worker starting row ${row} at $(date -Is) ==="
  env ROW_START="${row}" bash "${ROW_LAUNCHER}" &
  CURRENT_CHILD=$!
  set +e
  wait "${CURRENT_CHILD}"
  status=$?
  set -e
  CURRENT_CHILD=""
  if (( status != 0 )); then
    echo "Worker row ${row} failed with status ${status}" >&2
    exit "${status}"
  fi
  echo "=== worker finished row ${row} at $(date -Is) ==="
done

CURRENT_ROW=""
echo "=== worker completed all rows at $(date -Is) ==="
