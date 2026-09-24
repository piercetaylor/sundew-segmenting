#!/bin/bash
#SBATCH --job-name=sundew-species-finetune
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Fine-tune one screened backbone on the 110-species classifier, stage 2 of
# docs/species-classifier-plan.md. One submission per backbone:
#   sbatch --export=ALL,MODEL=dinov2-l-reg scripts/hellbender_species_finetune.slurm
# Indices 0-4 are the full-frame arm, 5-9 the crop arm, 10-14 full-square (only
# if the plan's full-square rule fires: add --array=10-14), seeds in order.
# The full arm is in every submission by the 2026-09-23 amendment to the plan.
#
# Seeds are the ResNet-18 baseline's. The split is split-110-test: the same
# validation, but 2,601 training images moved to a held-out test split
# (scripts/carve_test_split.py). Training is therefore ~19% smaller than the
# baseline's, so compare against a ResNet-18 rerun on this split, not 0.546.
#
# Per-size recipe: 24-block models get a gentler layer decay, or the bottom
# blocks would sit at ~0.75^25 = 1e-3 of the top lr and never move. Drop-path
# follows the usual 0.1 (base) / 0.2 (large); open_clip models ignore it.
# These are set once here, before any fine-tune result, and are not tuned on
# validation.
#
# Walltime: the ResNet-18 full arm took 45-56 min, bound by decoding ~2048 px
# JPEGs; a ViT-L forward/backward adds roughly as much again. 5h is margin.
# On preemption (gpu_requeue) the task is requeued and resumes from last.pt.

set -euo pipefail

: "${MODEL:?pass --export=ALL,MODEL=<tag from screen_species_backbones.MODELS>}"

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110-test"  # train minus the held-out test split; validation unchanged
SEEDS=(17 101 202 303 404)
ARMS=(full crop full-square)

ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"

case "$MODEL" in
  dinov2-l-reg|siglip2-so400m|bioclip-2) LAYER_DECAY=0.85; DROP_PATH=0.2 ;;
  *)                                     LAYER_DECAY=0.75; DROP_PATH=0.1 ;;
esac

cd "$BASE/sundew-segmenting"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"
export HF_HUB_OFFLINE=1

echo "=== node $(hostname) | model=$MODEL arm=$ARM seed=$SEED | restart ${SLURM_RESTART_COUNT:-0} | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/finetune_species_backbone.py \
  --model "$MODEL" --arm "$ARM" --seed "$SEED" \
  --records "$SPLIT/species-records.jsonl" \
  --labels "$SPLIT/labels.json" \
  --sections data/species-110-sections.json \
  --crop-dir "$CORPUS/crops/crops" \
  --output "models/species-110/finetune/$MODEL/$ARM/seed-$SEED" \
  --image-size 224 --epochs 25 --warmup-epochs 2 --batch-size 64 \
  --learning-rate 5e-5 --layer-decay "$LAYER_DECAY" --head-lr-mult 10 \
  --weight-decay 0.05 --drop-path "$DROP_PATH" --patience 8 --workers 7

echo "=== done: model=$MODEL arm=$ARM seed=$SEED | finished $(date -Is) ==="
