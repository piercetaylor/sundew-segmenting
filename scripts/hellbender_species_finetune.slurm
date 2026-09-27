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
#
# Distillation (docs/species-classifier-plan.md, "Distillation into DINOv2-S"):
#   KD=1        target = mean softmax of the five dinov2-l-reg full best-epoch
#               checkpoints; output tag <model>-kd
#   EPOCHS=100  longer schedule; output tag <model>[-kd]-e100; pass --time=08:00:00
#   CACHE=1     read full frames from the short-side-576 cache (make_full_cache.py);
#               output tag <model>[-kd][-eN]-c576
# Any run with KD=1 or EPOCHS other than 25 has patience off (= epochs), so the
# last epoch is always the end of the schedule.
#   sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1 scripts/hellbender_species_finetune.slurm

set -euo pipefail

: "${MODEL:?pass --export=ALL,MODEL=<tag from screen_species_backbones.MODELS>}"

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110-test"  # train minus the held-out test split; validation unchanged
SEEDS=(17 101 202 303 404)
ARMS=(full crop full-square)

ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}"

KD="${KD:-0}"
EPOCHS="${EPOCHS:-25}"
TAG="$MODEL"; PATIENCE=8; KD_ARGS=()
if [ "$KD" = 1 ]; then
  TAG="$TAG-kd"
  for s in "${SEEDS[@]}"; do
    KD_ARGS+=("models/species-110/finetune/dinov2-l-reg/full/seed-$s/classifier-best.pt")
  done
  KD_ARGS=(--teacher-checkpoints "${KD_ARGS[@]}" --kd-tau 1.0)
fi
[ "$EPOCHS" != 25 ] && TAG="$TAG-e$EPOCHS"
{ [ "$KD" = 1 ] || [ "$EPOCHS" != 25 ]; } && PATIENCE="$EPOCHS"
CACHE_ARGS=()
if [ "${CACHE:-0}" = 1 ]; then
  TAG="$TAG-c576"; CACHE_ARGS=(--full-dir "$CORPUS/full-s576")
fi

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

echo "=== node $(hostname) | model=$MODEL tag=$TAG kd=$KD epochs=$EPOCHS patience=$PATIENCE arm=$ARM seed=$SEED | restart ${SLURM_RESTART_COUNT:-0} | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/finetune_species_backbone.py \
  --model "$MODEL" --arm "$ARM" --seed "$SEED" \
  --records "$SPLIT/species-records.jsonl" \
  --labels "$SPLIT/labels.json" \
  --sections data/species-110-sections.json \
  --crop-dir "$CORPUS/crops/crops" \
  --output "models/species-110/finetune/$TAG/$ARM/seed-$SEED" \
  --image-size 224 --epochs "$EPOCHS" --warmup-epochs 2 --batch-size 64 \
  --learning-rate 5e-5 --layer-decay "$LAYER_DECAY" --head-lr-mult 10 \
  --weight-decay 0.05 --drop-path "$DROP_PATH" --patience "$PATIENCE" --workers 7 \
  "${KD_ARGS[@]}" "${CACHE_ARGS[@]}"

echo "=== done: model=$MODEL arm=$ARM seed=$SEED | finished $(date -Is) ==="
