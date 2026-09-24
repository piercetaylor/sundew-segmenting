#!/bin/bash
#SBATCH --job-name=sundew-species-corpus
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# The scaled species corpus, authorised by both gates in
# docs/crop-readiness-plan.md: the crop failure rate is 2.43% (below the
# pre-registered 5%) and the crop is worth +0.0246 balanced accuracy.
#
#   110 species with >= 50 available observations, capped at 300 each
#   ~21,600 images, ~45 GB, roughly 4 hours
#   observer cap 5, so the classifier split can be grouped by observer
#   licences CC0 / CC-BY / CC-BY-NC -- see docs/licence-policy.md
#
# Restartable. The per-species selection is written to selection.jsonl before
# any download, and download_candidates skips photo ids already in the species
# manifest, so re-running after a walltime kill resumes the same corpus rather
# than drawing a new sample. Walltime is 12h against an estimated 4 so that
# should not be needed.
#
# This does NOT reuse sundew-species-set/. That set is CC0/CC-BY at 200 images
# across 10 species and backs reports/species-crop-comparison.md; leaving it
# untouched keeps that result reproducible, at the cost of re-downloading about
# 1,750 photographs here.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
cd "$BASE/sundew-segmenting"
mkdir -p logs

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src

echo "=== node $(hostname), started $(date -Is) ==="
df -h "$BASE" | tail -1

python scripts/acquire_species_corpus.py \
  --output "$BASE/sundew-species-corpus" \
  --min-observations 50 \
  --per-species 300 \
  --max-per-observer 5 \
  --licenses cc0,cc-by,cc-by-nc \
  --seed 20260921 \
  --download-delay 0.25 \
  --exclude-manifest \
    "$BASE/sundew-data/sundew-segmenting-reviewed-v0.3.0/metadata/training-records.jsonl" \
    "$BASE/field-probe-v1/field-probe-records.jsonl"

echo "=== finished $(date -Is) ==="
df -h "$BASE" | tail -1
