#!/bin/bash
#SBATCH --job-name=sundew-backbone-screen-summary
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Tabulates the frozen-backbone screen into reports/species-backbone-screen/.
# Submitted with --dependency=afterok on hellbender_backbone_screen.slurm; safe
# to rerun by hand at any time, it only reads the per-model outputs.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

python scripts/summarize_backbone_screen.py \
  --screen-dir models/species-110/screen \
  --out-md reports/species-backbone-screen/results.md \
  --out-json reports/species-backbone-screen/summary.json
