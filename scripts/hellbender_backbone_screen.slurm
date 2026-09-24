#!/bin/bash
#SBATCH --job-name=sundew-backbone-screen
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Frozen-backbone screen for the 110-species classifier: one array task per
# pretrained backbone, each scoring a linear probe, k-NN and (CLIP models)
# zero-shot on the crop, full and full-square arms. Design, prior and decision
# rule: docs/species-classifier-plan.md.
#
# Submit together with the summary, which runs only if every task succeeds:
#   jid=$(sbatch --parsable scripts/hellbender_backbone_screen.slurm)
#   sbatch --dependency=afterok:$jid scripts/hellbender_backbone_screen_summary.slurm
#
# Weights were pre-fetched to $HF_HOME on the login node, so tasks run offline.
# Features are cached per arm under models/species-110/screen/<model>/, so a
# task preempted on gpu_requeue picks up where it stopped.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110"
# MODEL_SET=small (array 0-10) screens the on-device student candidates:
#   jid=$(sbatch --parsable --array=0-10 --export=ALL,MODEL_SET=small scripts/hellbender_backbone_screen.slurm)
#   sbatch --dependency=afterok:$jid --export=ALL,OUT=reports/species-backbone-screen-small scripts/hellbender_backbone_screen_summary.slurm
case "${MODEL_SET:-main}" in
  main)  MODELS=(resnet18-in1k convnext-b-in22k dinov2-b dinov2-l-reg dinov3-vit-b dinov3-convnext-b siglip2-so400m bioclip bioclip-2) ;;
  small) MODELS=(dinov2-s dinov2-s-reg dinov3-vit-s dinov3-vit-s-plus tinyvit-21m-in22k convnext-nano-in12k convnext-t-in22k
                 mobilenetv4-conv-m-in12k mobilenetv4-hybrid-m-in12k efficientnetv2-s-in21k mobileclip2-s2) ;;
  *) echo "unknown MODEL_SET=$MODEL_SET"; exit 1 ;;
esac
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

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

echo "=== node $(hostname) | model=$MODEL | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/screen_species_backbones.py \
  --model "$MODEL" \
  --records "$SPLIT/species-records.jsonl" \
  --labels "$SPLIT/labels.json" \
  --sections data/species-110-sections.json \
  --crop-dir "$CORPUS/crops/crops" \
  --output "models/species-110/screen/$MODEL" \
  --workers 7

echo "=== done: model=$MODEL | finished $(date -Is) ==="
