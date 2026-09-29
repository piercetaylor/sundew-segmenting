#!/bin/bash
#SBATCH --job-name=sundew-species-export
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Steps 2 and 3 of "Test protocol for the shipped student"
# (docs/species-classifier-plan.md): int8 ONNX export of the shipped student,
# its accuracy change against fp32, its temperature and its latency, all on
# validation. CPU only; latency is timed on one thread of this node.
#   sbatch scripts/hellbender_species_export.slurm
# CKPT and OUT default to seed 17 of the carried-forward arm, KDw-100c+T.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
TAG=dinov2-s-kdw-e100-t-c576
CKPT="${CKPT:-models/species-110/finetune/$TAG/full/seed-17/classifier-last.pt}"
OUT="${OUT:-models/species-110/release/$TAG-seed17}"

cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"
export HF_HUB_OFFLINE=1

echo "=== node $(hostname) | $CKPT -> $OUT | started $(date -Is) ==="
python scripts/export_species_onnx.py --checkpoint "$CKPT" --output "$OUT" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" --workers 6
echo "=== done | finished $(date -Is) ==="
