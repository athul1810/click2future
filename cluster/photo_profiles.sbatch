#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-photo-profiles
#SBATCH --output=photo-profiles-%j.out
#SBATCH --error=photo-profiles-%j.err
set -euo pipefail
umask 077
: "${C2F_PHOTO_PROJECT:?Set the committed project}"
: "${C2F_PHOTO_ROOT:?Set the shared photo workspace}"
node_root="/tmp/$USER/c2f-force-photo-20261104"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$node_root/hf-cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"$node_root/.venv/bin/python" "$C2F_PHOTO_PROJECT/scripts/run_photo_profiles.py" \
    --repo "$node_root/repo" --gallery "$C2F_PHOTO_ROOT/streamforce" \
    --output "$node_root/profiles-$SLURM_JOB_ID"
mkdir -p "$C2F_PHOTO_ROOT/profiles-$SLURM_JOB_ID"
cp "$node_root/profiles-$SLURM_JOB_ID/"*.json "$C2F_PHOTO_ROOT/profiles-$SLURM_JOB_ID/"
