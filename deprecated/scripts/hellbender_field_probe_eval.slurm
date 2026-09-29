#!/bin/bash
#SBATCH --job-name=sundew-field-eval
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Step 1 of docs/field-probe-return.md: score the checkpoint that ran the scrape
# probe against the held-out field masks, before any retraining.
# evaluate_checkpoint.py keeps the model on CPU, so this asks for no GPU.

set -euo pipefail

REPO_DIR=/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting
PROBE_ROOT=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CKPT=models/recipe/combined/seed-17/segformer-b0-best.pt

cd "$REPO_DIR"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$PROBE_ROOT/sundew-cache/torch"
export HF_HOME="$PROBE_ROOT/sundew-cache/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== node: $(hostname) ==="
echo "=== checkpoint: $CKPT (combined recipe, val IoU 0.6838 — the one that ran the probe) ==="

# field-eval is the number the retrain gets compared against.
# field-train is descriptive only, and is meaningless once those 30 are trained on.
for split in field-eval field-train; do
  echo
  echo "############ $split ############"
  python scripts/evaluate_checkpoint.py \
    --checkpoint "$CKPT" \
    --split "$split" \
    --images "$PROBE_ROOT/field-probe-eval/images" \
    --masks "$PROBE_ROOT/field-probe-eval/masks" \
    --metadata "$PROBE_ROOT/field-probe-v1/field-probe-records.jsonl" \
    --output data/reports/field-probe-evaluation
done

echo
echo "=== done ==="
