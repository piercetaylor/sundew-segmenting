#!/bin/bash
#SBATCH --job-name=sundew-field-unet
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# U-Net / ResNet-34 with the field recipe, for the plant outline on the site
# (docs/reports/segmentation-site-prereg.md). Identical to the `field` arm of
# hellbender_field_compare.slurm except --model: SegFormer-B0's NVIDIA licence
# allows research or evaluation use only, so the site can't use it.
#
# REPO is the checkout whose code runs; MODELS is where checkpoints go (the
# shared, git-ignored models/ folder by default, next to the SegFormer runs).
#   sbatch scripts/hellbender_field_unet.slurm
#   sbatch --export=ALL,REPO=<worktree> scripts/hellbender_field_unet.slurm

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
REPO="${REPO:-$BASE/sundew-segmenting}"
MODELS="${MODELS:-$BASE/sundew-segmenting/models}"
SEEDS=(17 101 202 303 404)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
ROOT="$BASE/field-train-root"

cd "$REPO"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | unet-resnet34 field seed=$SEED | repo $REPO ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_baseline.py --model unet-resnet34 \
  --images "$ROOT/images" --masks "$ROOT/masks" --metadata "$ROOT/training-records.jsonl" \
  --output "$MODELS/field-compare/field-unet/seed-$SEED" \
  --image-size 1024 --epochs 80 --patience 15 --learning-rate 3e-4 \
  --loss bce-tversky --batch-size 8 --workers 7 --seed "$SEED" \
  --no-allow-partial --no-balanced-growth-forms

echo "=== done: unet-resnet34 field seed=$SEED ==="
