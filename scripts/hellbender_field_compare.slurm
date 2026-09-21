#!/bin/bash
#SBATCH --job-name=sundew-field-compare
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Step 2 of docs/field-probe-return.md: does adding 30 field masks to the 144
# curated ones close any of the 0.184 IoU field gap measured in
# reports/field-probe-evaluation.md?
#
# Two arms, identical flags, differing only in dataset root. Paired across three
# seeds rather than run once, because baseline-seed-sweep.md measured a
# seed-to-seed SD of 0.0176 for SegFormer on this dataset -- a single seed per
# arm could not distinguish anything smaller than ~0.07 IoU.
#
# Flags are variant 5 ("combined") of hellbender_recipe_sweep.slurm, NOT
# hellbender_train.slurm, which is the 768 px architecture comparison.
# --no-balanced-growth-forms is required on BOTH arms: 9 of the 30 field images
# carry growth_form 'unknown', which inverse-frequency weighting would promote
# to its own upweighted bucket.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
REPO="$BASE/sundew-segmenting"
SEEDS=(17 101 202)
ARMS=(frozen field)

ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"

case "$ARM" in
  frozen) ROOT="$BASE/sundew-data/sundew-segmenting-reviewed-v0.3.0"
          META="$ROOT/metadata/training-records.jsonl" ;;
  field)  ROOT="$BASE/field-train-root"
          META="$ROOT/training-records.jsonl" ;;
esac

cd "$REPO"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | arm=$ARM seed=$SEED ==="
echo "=== root $ROOT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_baseline.py --model segformer-b0 \
  --images "$ROOT/images" --masks "$ROOT/masks" --metadata "$META" \
  --output "models/field-compare/$ARM/seed-$SEED" \
  --image-size 1024 --epochs 80 --patience 15 --learning-rate 3e-4 \
  --loss bce-tversky --batch-size 8 --workers 7 --seed "$SEED" \
  --no-allow-partial --no-balanced-growth-forms

echo "=== done: arm=$ARM seed=$SEED ==="
