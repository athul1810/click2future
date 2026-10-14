#!/usr/bin/env bash
# C2F_PROJECT and C2F_UV must be absolute paths in your own allocation.
# Select CPU first; select cu128 only after checking the allocated GPU driver.
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=c2f-bootstrap
#SBATCH --output=tc2-bootstrap-%j.out
#SBATCH --error=tc2-bootstrap-%j.err
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Run this through sbatch on an allocated compute node.' >&2
    exit 2
fi
: "${C2F_PROJECT:?Set the uploaded repository absolute path}"
: "${C2F_UV:?Set the absolute uv executable path from your own bootstrap environment}"
backend="${C2F_BACKEND:-cpu}"
if [[ "$backend" != cpu && "$backend" != cu128 ]]; then
    printf '%s\n' 'C2F_BACKEND must be cpu or cu128.' >&2
    exit 2
fi
cd -- "$C2F_PROJECT"
"$C2F_UV" --version
"$C2F_UV" sync --locked --extra "$backend"
device=cpu
if [[ "$backend" == cu128 ]]; then device=cuda; fi
.venv/bin/python -m click2future.env_check --device "$device" \
    --output "outputs/tc2-bootstrap-${SLURM_JOB_ID}.json"
"$C2F_UV" pip check --python .venv/bin/python
