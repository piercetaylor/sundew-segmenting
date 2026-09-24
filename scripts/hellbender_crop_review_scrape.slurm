#!/bin/bash
#SBATCH --job-name=sundew-crop-scrape
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Step 1 of docs/crop-readiness-plan.md: 500 fresh uncurated images whose crops
# a human reviews by eye, to pin down the crop-failure rate. The field-eval
# estimate rests on one failing image in 17, 95% CI [1.0%, 27.0%] -- too wide to
# decide whether a 10-20k scrape is safe.
#
# Diversity caps are deliberately loose (species 25, observer 3). The point is
# to sample what an actual large scrape would hit, not a balanced set; capping
# hard would bias the failure rate away from the common species that dominate
# the real pool.

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
python scripts/acquire_inaturalist.py \
  --count 500 \
  --output "$BASE/sundew-crop-review/raw" \
  --max-per-species 25 \
  --max-per-observer 3 \
  --seed 20260921
echo "=== scrape done ==="
ls "$BASE/sundew-crop-review/raw/images" | wc -l
