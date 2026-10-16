#!/usr/bin/env bash
# Supply the assigned QoS explicitly: sbatch --qos="$TC2_QOS" cluster/inventory.sbatch
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=c2f-inventory
#SBATCH --output=tc2-inventory-%j.out
#SBATCH --error=tc2-inventory-%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Run this through sbatch on an allocated compute node.' >&2
    exit 2
fi
printf 'job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
uname -srmo
getconf GNU_LIBC_VERSION
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if type module >/dev/null 2>&1; then
    module avail 2>&1
fi
