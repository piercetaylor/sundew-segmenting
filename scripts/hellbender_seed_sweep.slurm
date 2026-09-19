#!/bin/bash
#SBATCH --job-name=sundew-sweep
#SBATCH --array=0-9%5
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Multi-seed U-Net vs SegFormer comparison.
#
#   sbatch scripts/hellbender_seed_sweep.slurm
#
# A single run cannot separate these two models: the validation split is 19
# images and the headline IoU gap was ~0.02. This repeats both models over five
# seeds so the gap can be read against its own spread.
#
# The array is 2 models x 5 seeds = 10 tasks, throttled to 5 concurrent (%5) so
# the sweep does not occupy ten A100s at once for a ~3 minute job each.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting}"
DATASET_DIR="${DATASET_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data/sundew-segmenting-reviewed-v0.3.0}"
CACHE_ROOT="${CACHE_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache}"
ENV_NAME="${ENV_NAME:-sundew-seg}"
MODEL_ROOT="${MODEL_ROOT:-models/sweep}"

MODELS=(unet-resnet34 segformer-b0)
SEEDS=(17 101 202 303 404)

MODEL="${MODELS[$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))]}"
OUT_DIR="$MODEL_ROOT/$MODEL/seed-$SEED"

cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

echo "=== node=$(hostname) model=$MODEL seed=$SEED ==="
nvidia-smi --query-gpu=name --format=csv,noheader

python scripts/train_baseline.py \
  --model "$MODEL" \
  --images "$DATASET_DIR/images" \
  --masks "$DATASET_DIR/masks" \
  --metadata "$DATASET_DIR/metadata/training-records.jsonl" \
  --output "$OUT_DIR" \
  --epochs "${EPOCHS:-30}" \
  --batch-size 8 \
  --image-size 768 \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --no-allow-partial
