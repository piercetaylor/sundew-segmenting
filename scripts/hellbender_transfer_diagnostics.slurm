#!/bin/bash
#SBATCH --job-name=sundew-transfer-diagnostics
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Pre-registered transfer-set diagnostics of KDw-100c+T (docs/species-classifier-plan.md,
# "Transfer set"): the teacher ensemble and student seed 17 over the 12,126 transfer
# photos, by source and training-count bin. See scripts/diagnose_transfer_set.py.
#   sbatch scripts/hellbender_transfer_diagnostics.slurm
# Writes reports/species-distill/t-transfer-diagnostics.json and
# models/species-110/finetune/dinov2-s-kdw-e100-t-c576/full/seed-17/transfer-diagnostics.npz.
# The held-out test split is not read.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src:scripts TORCH_HOME="$BASE/sundew-cache/torch" HF_HOME="$BASE/sundew-cache/huggingface" HF_HUB_OFFLINE=1

echo "=== node $(hostname) | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python scripts/diagnose_transfer_set.py
echo "=== done $(date -Is) ==="
