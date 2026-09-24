#!/bin/bash
#SBATCH --job-name=sundew-species-110-crops
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Predicted crop boxes for all 17,678 corpus images, so the crop arm of
# hellbender_species_110.slurm has something to read.
#
# render_crop_review.py never moves the model to a device, so this is CPU work:
# measured at 1.99 s/image on the ten-species set (1,753 images in 58 minutes,
# 8 CPUs). 17,678 images is about 9.8 hours in one process, so it is sharded
# eight ways over pre-built symlink directories -- roughly 75 minutes each.
# Sharding needs no code change because the script globs a flat directory.
#
# The checkpoint, threshold and pad MUST match the ones that produced the
# ten-species crops. If they drift, the crop arm is no longer measuring the
# same thing that scored +0.0246 in reports/species-crop-comparison.md, and
# the comparison against 0.7542 becomes meaningless. threshold 0.6 and pad
# 0.10 are the script's own defaults and are passed explicitly here so a
# future change to those defaults cannot silently alter the corpus.
#
# seed-101 is the field-trained SegFormer-B0 from reports/field-compare.md,
# the +0.061 IoU model, and is the checkpoint the ten-species crops used.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
K="$SLURM_ARRAY_TASK_ID"

cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | shard $K | started $(date -Is) ==="
echo "shard images: $(ls "$CORPUS/_shard-$K" | wc -l)"

python scripts/render_crop_review.py \
  --images "$CORPUS/_shard-$K" \
  --checkpoint models/field-compare/field/seed-101/segformer-b0-best.pt \
  --output "$CORPUS/crops-shard-$K" \
  --threshold 0.6 --pad 0.10 \
  --save-crops --no-sheets

echo "=== done: shard $K | crops $(ls "$CORPUS/crops-shard-$K/crops" 2>/dev/null | wc -l) | finished $(date -Is) ==="
