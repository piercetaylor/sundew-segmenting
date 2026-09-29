#!/bin/bash
#SBATCH --job-name=sundew-species-scrape
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Step 2 of docs/crop-readiness-plan.md: the data for the crop-vs-full-frame
# test. Ten best-represented species, 200 images each, observer-capped at 5 so
# the classifier split can be grouped by observer like the segmentation splits.
#
# This exists to answer whether the crop is needed at all. If a classifier
# trained on full frames matches one trained on segmentation crops, the crop
# failure rate stops being a risk to the species project.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src

echo "=== node $(hostname) ==="
python scripts/acquire_species_set.py \
  --output "$BASE/sundew-species-set" \
  --per-species 200 \
  --max-per-observer 5 \
  --seed 20260921 \
  --species \
    "Drosera rotundifolia" "Drosera intermedia" "Drosera capillaris" \
    "Drosera spatulata" "Drosera anglica" "Drosera auriculata" \
    "Drosera brevifolia" "Drosera aberrans" "Drosera glanduligera" \
    "Drosera arcturi"
echo "=== species scrape done ==="
