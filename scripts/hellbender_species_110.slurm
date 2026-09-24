#!/bin/bash
#SBATCH --job-name=sundew-species-110
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# The 110-species classifier on the scaled corpus (17,678 images), scaling up
# reports/species-crop-comparison.md from ten species to the full class set.
#
# Indices 0-4 are the full-frame arm, 5-9 the crop arm, in seed order. Submit
# 0-4 now; 5-9 only once every crop exists, because the trainer refuses to run
# when an image is missing from an arm and that guard is the thing keeping the
# two arms comparable.
#
# The recipe is deliberately byte-identical to the ten-species run -- ResNet-18
# at 224, batch 32, lr 3e-4, 25 epochs, patience 8, seeds 17/101/202/303/404 --
# because the only reason the new number can be read against 0.7295 (full) and
# 0.7542 (crop) is that nothing but the class set and the data volume moved.
# Backbone and resolution are the next experiments, one variable at a time.
#
# Split: split-110, built with --max-class-val-fraction 0.28. Without that
# ceiling the multi-species observers flood validation and starve thin classes
# (D. nitidula got 14 training images against 31 validation); see the split
# note in the report. 13,719 train / 3,959 validation, 22.4%, every class
# between 19.7% and 28.0%, zero shared observers.
#
# Walltime: the ten-species full arm took 6 minutes for 1,362 training images.
# 13,719 is about 10x that, so roughly an hour, and 3h is margin. gpu_requeue
# is preemptible and the trainer has no resume, so a preempted task restarts
# from scratch; pass --partition=gpu instead if that becomes a problem.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
# Rerun on the test-carved split without touching the original checkpoints:
#   sbatch --export=ALL,SPLIT_NAME=split-110-test,OUT_ROOT=models/species-110-test/resnet18 \
#     scripts/hellbender_species_110.slurm
SPLIT="$CORPUS/${SPLIT_NAME:-split-110}"
OUT_ROOT="${OUT_ROOT:-models/species-110}"
SEEDS=(17 101 202 303 404)
ARMS=(full crop)

ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"

cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | split=$SPLIT arm=$ARM seed=$SEED | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/train_species_classifier.py \
  --records "$SPLIT/species-records.jsonl" \
  --labels "$SPLIT/labels.json" \
  --arm "$ARM" \
  --crop-dir "$CORPUS/crops/crops" \
  --output "$OUT_ROOT/$ARM/seed-$SEED" \
  --epochs 25 --batch-size 32 --image-size 224 \
  --learning-rate 3e-4 --patience 8 --workers 7 --seed "$SEED"

echo "=== done: arm=$ARM seed=$SEED | finished $(date -Is) ==="
