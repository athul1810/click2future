#!/bin/bash -l
#SBATCH --job-name=c2f-coverage
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=coverage-%j.out
#SBATCH --error=coverage-%j.err
set -euo pipefail
: "${C2F_PROJECT:?}" "${C2F_PYTHON:?}" "${C2F_DATA_DIR:?}" "${C2F_CACHE:?}"
: "${C2F_WARM_START:?}" "${C2F_COVERAGE_RESULTS:?}" "${C2F_SOURCE_REVISION:?}"
cd "$C2F_PROJECT"
export PYTHONPATH="$C2F_PROJECT/src"
export OMP_NUM_THREADS=4
extra=()
if [[ -n "${C2F_PREPARED_COVERAGE:-}" ]]; then
  extra+=(--coverage "$C2F_PREPARED_COVERAGE")
fi
exec "$C2F_PYTHON" -u submission/coverage_comparison.py \
  --dataset "$C2F_DATA_DIR" --cache "$C2F_CACHE" \
  --warm-start "$C2F_WARM_START" --output "$C2F_COVERAGE_RESULTS" "${extra[@]}"
