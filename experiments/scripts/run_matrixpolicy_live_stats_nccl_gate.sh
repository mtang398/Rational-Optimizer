#!/bin/bash
#SBATCH --job-name=mp-sync-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append

set -euo pipefail

ROOT="/home/mt872/rationalOPT"
CAMPAIGN_ROOT="${ROOT}/experiments/corrections/matrixpolicy_live_stats_20260712"
RUNTIME_ROOT="${CAMPAIGN_ROOT}/runtime"
PYTHON="${ROOT}/.venv-cu128/bin/python"
RESULT_PATH="${CAMPAIGN_ROOT}/validation/nccl_gate.json"
NODE_DENYLIST="${NODE_DENYLIST:-sablab-gpu-12 seo-compute-01}"
MAX_PREEMPT_REQUEUES="${MAX_PREEMPT_REQUEUES:-8}"
CHILD_PID=""

cd "${RUNTIME_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export RATIONALOPT_WORKSPACE_ROOT="${ROOT}"
export PYTHONPATH="${RUNTIME_ROOT}/activation:${RUNTIME_ROOT}:${PYTHONPATH:-}"
export PATH="${ROOT}/.venv-cu128/bin:${PATH}"
export RATIONAL_OPT_TORCH_FALLBACK=0
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS=8

if (( ${SLURM_RESTART_COUNT:-0} > MAX_PREEMPT_REQUEUES )); then
  echo "Preemption retry cap exceeded before NCCL gate start: ${SLURM_RESTART_COUNT}" >&2
  exit 89
fi

request_requeue() {
  if [[ -n "${CHILD_PID}" ]]; then
    kill -TERM -- "-${CHILD_PID}" 2>/dev/null || true
  fi
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap request_requeue USR1

"${PYTHON}" "${RUNTIME_ROOT}/experiments/scripts/build_matrixpolicy_live_stats_correction.py" verify-runtime >/dev/null

node="${SLURMD_NODENAME:-${SLURM_NODELIST:-}}"
for denied in ${NODE_DENYLIST//,/ }; do
  if [[ "${node}" == "${denied}" ]]; then
    echo "Refusing denylisted node ${node}" >&2
    exit 87
  fi
done
node_info="$(scontrol show node "${node}")"
if [[ "${node_info}" != *"nvlink"* ]]; then
  echo "NCCL gate requires an NVLink node" >&2
  exit 87
fi
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#gpu_names[@]}" -ne 4 ]]; then
  echo "NCCL gate expected four GPUs, found ${#gpu_names[@]}" >&2
  exit 86
fi
for name in "${gpu_names[@]}"; do
  if [[ "${name}" != *"A6000"* ]]; then
    echo "NCCL gate expected RTX A6000, found ${name}" >&2
    exit 86
  fi
done
nvidia-smi topo -m

mkdir -p "$(dirname "${RESULT_PATH}")"
tmp_result="${RESULT_PATH}.tmp.${SLURM_JOB_ID}"
set +e
setsid "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  "${RUNTIME_ROOT}/tests/matrixpolicy_live_stats_nccl_preflight.py" \
  >"${tmp_result}" 2>&1 &
CHILD_PID=$!
wait "${CHILD_PID}"
status=$?
CHILD_PID=""
set -e
cat "${tmp_result}"
if (( status != 0 )); then
  mv "${tmp_result}" "${RESULT_PATH}.failed.${SLURM_JOB_ID}"
  exit "${status}"
fi
"${PYTHON}" - "${tmp_result}" "${RESULT_PATH}" "${CAMPAIGN_FREEZE_SHA256:?}" "${CAMPAIGN_RUNTIME_SHA256:?}" <<'PYRESULT'
import json
import os
import sys
import time

raw_path, result_path, freeze_sha, runtime_sha = sys.argv[1:5]
record = None
with open(raw_path) as handle:
    for line in handle:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("event") == "matrixpolicy_live_stats_nccl_preflight":
            record = candidate
if record is None or record.get("status") != "pass":
    raise SystemExit("NCCL preflight did not emit a passing result")
record.update(
    {
        "campaign_freeze_sha256": freeze_sha,
        "campaign_runtime_sha256": runtime_sha,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("SLURM_NODELIST"),
        "validated_unix_time": time.time(),
    }
)
with open(result_path, "w") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PYRESULT
rm -f "${tmp_result}"
echo "=== four-GPU NCCL correction gate passed ==="
