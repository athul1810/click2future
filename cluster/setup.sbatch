#!/bin/bash -l
# Installs only into the user's home; application execution stays in Slurm.
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=c2f-setup
#SBATCH --output=tc2-setup-%j.out
#SBATCH --error=tc2-setup-%j.err
set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    printf '%s\n' 'Submit with sbatch; application code must not run on the head node.' >&2
    exit 2
fi
: "${C2F_PROJECT:?Set the uploaded source directory}"
cd -- "$C2F_PROJECT"
mkdir -p outputs
printf 'job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
uname -srmo
getconf GNU_LIBC_VERSION
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "outputs/tc2-hardware-$SLURM_JOB_ID.csv"

# A conservative full CUDA 12.8 gate, rather than assuming minor-version compatibility.
# The locked Linux torch wheels require glibc 2.28. Never change cluster drivers.
glibc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
if [[ "$(printf '%s\n' 2.28 "$glibc_version" | sort -V | head -n 1)" != 2.28 ]]; then
    printf 'glibc %s is below 2.28; prepare a separately reviewed compatible lock.\n' "$glibc_version" >&2
    exit 3
fi
if [[ "$(uname -m)" != x86_64 && "$(uname -m)" != aarch64 ]]; then
    printf '%s\n' 'No locked torch wheel for this architecture.' >&2
    exit 3
fi
while IFS= read -r driver; do
    if [[ "$(printf '%s\n' 570.26 "$driver" | sort -V | head -n 1)" != 570.26 ]]; then
        printf 'Driver %s is below our conservative CUDA 12.8 gate (570.26).\n' "$driver" >&2
        printf '%s\n' 'Keep the hardware report and choose a compatible environment.' >&2
        exit 3
    fi
done < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)

if ! type module >/dev/null 2>&1; then
    printf '%s\n' 'Environment Modules is unavailable in this login job shell; contact TC2 support.' >&2
    exit 4
fi
module load anaconda
eval "$(conda shell.bash hook)"
bootstrap="$HOME/.conda/envs/c2f-bootstrap"
if [[ ! -x "$bootstrap/bin/uv" ]]; then
    conda create --yes --prefix "$bootstrap" --override-channels --channel conda-forge \
        --strict-channel-priority python=3.12.15 uv=0.13.0
fi
uv_path="$bootstrap/bin/uv"
"$uv_path" --version
export UV_CACHE_DIR="$HOME/.cache/click2future/uv"
export UV_PROJECT_ENVIRONMENT="$C2F_PROJECT/.venv"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Prebuilt torch wheels supply CUDA runtime libraries; avoid mixing toolkit modules.
"$uv_path" sync --locked --extra cu128
"$uv_path" pip check --python .venv/bin/python
.venv/bin/python -m click2future.env_check --device cuda \
    --output "outputs/tc2-device-$SLURM_JOB_ID.json"
.venv/bin/python -m pytest -q --junitxml="outputs/tc2-tests-$SLURM_JOB_ID.xml"
.venv/bin/python -m click2future.profile_pilot --device cuda --require-slurm \
    --batch-size 1 --future-frames 31 --warmup 3 --steps 5 \
    --output "outputs/tc2-pilot-$SLURM_JOB_ID.json"
printf '{"status":"passed","job_id":"%s","scope":"environment, tests and synthetic 31-frame denoiser; no trained model"}\n' \
    "$SLURM_JOB_ID" > "outputs/tc2-ready-$SLURM_JOB_ID.json"
