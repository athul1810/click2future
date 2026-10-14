#!/usr/bin/env bash
# Submit only after verifying the assigned QoS and selecting a compatible torch build.
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=c2f-probe
#SBATCH --output=tc2-probe-%j.out
#SBATCH --error=tc2-probe-%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Run this through sbatch on an allocated compute node.' >&2
    exit 2
fi
: "${C2F_PROJECT:?Set the uploaded repository absolute path}"
cd -- "$C2F_PROJECT"
python_path="${C2F_PYTHON:-$C2F_PROJECT/.venv/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
printf 'job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$python_path" -m click2future.env_check --device cuda \
    --output "outputs/tc2-device-${SLURM_JOB_ID}.json"
"$python_path" -m click2future.profile_pilot --device cuda --require-slurm \
    --output "outputs/tc2-pilot-${SLURM_JOB_ID}.json"
