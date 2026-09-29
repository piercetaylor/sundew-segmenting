#!/bin/bash
#SBATCH --job-name=sundew-full-cache
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Build the short-side-576 full-frame cache for split-110-test, train and
# validation only (scripts/make_full_cache.py). Rerunnable: finished files are skipped.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
python scripts/make_full_cache.py --records "$CORPUS/split-110-test/species-records.jsonl" \
  --out "$CORPUS/full-s576" --short-side 576 --workers 16
ls "$CORPUS/full-s576" | grep -c '^inat_' ; du -sh "$CORPUS/full-s576"
