#!/bin/bash
#SBATCH --job-name=sundew-species-compare
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Step 2 of docs/crop-readiness-plan.md: does the segmentation crop earn its
# place in a crop-then-classify species pipeline?
#
# Indices 0-4 are the full-frame arm, 5-9 the crop arm, in seed order. The two
# arms differ only in which image file is read. Paired across the same five
# seeds used throughout this project, because field-compare.md showed a
# three-seed run could not clear p=0.05 on an effect of the size expected here.
#
# If the arms land within noise, the crop failure rate measured by step 1 stops
# being a risk to the species project and segmentation reverts to being about
# projected area alone.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
SET="$BASE/sundew-species-set"
SEEDS=(17 101 202 303 404)
ARMS=(full crop)

ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"

cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | arm=$ARM seed=$SEED ==="

python scripts/train_species_classifier.py \
  --records "$SET/species-records.jsonl" \
  --labels "$SET/labels.json" \
  --arm "$ARM" \
  --crop-dir "$SET/crops/crops" \
  --output "models/species-compare/$ARM/seed-$SEED" \
  --epochs 25 --batch-size 32 --image-size 224 \
  --learning-rate 3e-4 --patience 8 --workers 7 --seed "$SEED"

echo "=== done: arm=$ARM seed=$SEED ==="
