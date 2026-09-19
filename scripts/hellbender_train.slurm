#!/bin/bash
#SBATCH --job-name=sundew-baselines
#SBATCH --array=0-1
#SBATCH --partition=requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_DIR="${1:?Pass the sundew-segmenting repository directory as argument 1}"
DATASET_DIR="${2:?Pass the extracted reviewed dataset directory as argument 2}"

cd "$REPO_DIR"
mkdir -p logs models/hellbender
source .venv/bin/activate
export PYTHONPATH=src

MODELS=(unet-resnet34 segformer-b0)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

python scripts/train_baseline.py \
  --model "$MODEL" \
  --images "$DATASET_DIR/images" \
  --masks "$DATASET_DIR/masks" \
  --metadata "$DATASET_DIR/metadata/training-records.jsonl" \
  --output "models/hellbender/$MODEL" \
  --epochs 30 \
  --batch-size 8 \
  --image-size 768 \
  --workers 8 \
  --no-allow-partial
