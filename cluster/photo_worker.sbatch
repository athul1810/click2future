#!/bin/bash -l
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --job-name=c2f-photo-worker
#SBATCH --output=photo-worker-%j.out
#SBATCH --error=photo-worker-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${C2F_PHOTO_PROJECT:?Set the committed uploaded project}"
: "${C2F_PHOTO_NODE_ROOT:?Set the verified node-local model workspace}"
: "${C2F_PHOTO_SPOOL:?Set the shared photo spool}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$C2F_PHOTO_NODE_ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
test -s "$C2F_PHOTO_NODE_ROOT/repo/ARTIFACTS.json"
test -x "$C2F_PHOTO_NODE_ROOT/.venv/bin/python"
"$C2F_PHOTO_NODE_ROOT/.venv/bin/python" "$C2F_PHOTO_PROJECT/cluster/photo_worker.py" \
    --repo "$C2F_PHOTO_NODE_ROOT/repo" --spool "$C2F_PHOTO_SPOOL" \
    --raw-root "$C2F_PHOTO_NODE_ROOT/raw" --idle-seconds 1800
