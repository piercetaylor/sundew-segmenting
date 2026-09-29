#!/bin/bash
#SBATCH --job-name=sundew-transfer-set
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Download and deduplicate the distillation transfer set, pre-registered in
# docs/species-classifier-plan.md, "Transfer set". Approved 2026-09-28.
#
#   12,179 photos selected by the metadata-only dry run (selection.jsonl,
#   observation-id ceiling 404031620); about 25 GB, roughly 3 hours
#   licences CC0 / CC-BY / CC-BY-NC -- see docs/licence-policy.md
#
# Restartable: --download reads the recorded selection and download_candidates
# skips photo ids already in each species manifest, so a rerun resumes rather
# than drawing a new sample. --dedup then drops sha256 duplicates and dHash
# distance <= 6 against every split (train, validation and test) and within the
# set, and writes transfer-records.jsonl and transfer-dedup.json. The kept photos
# are then cached at short side 576 (make_full_cache.py, as the corpus's
# full-s576), into $OUT/full-s576. Every selected original is >= 576 px.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
OUT="$BASE/sundew-transfer-set"
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

ARGS=(--corpus "$BASE/sundew-species-corpus"
      --records "$BASE/sundew-species-corpus/split-110-test/species-records.jsonl"
      --output "$OUT")
python scripts/acquire_transfer_set.py "${ARGS[@]}" --download
python scripts/acquire_transfer_set.py "${ARGS[@]}" --dedup
python scripts/make_full_cache.py --records "$OUT/transfer-records.jsonl" --splits transfer \
  --out "$OUT/full-s576" --short-side 576 --workers 8

du -sh "$OUT"
echo "=== done $(date -Is) ==="
