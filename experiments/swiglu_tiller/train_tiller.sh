#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" -m experiments.swiglu_tiller.train --condition swiglu_tiller "$@"
