#!/bin/bash
#SBATCH --job-name=sundew-recipe
#SBATCH --array=0-11%6
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Recipe sweep: is the flat tail of the data-scaling curve a data ceiling or a
# recipe ceiling?
#
#   sbatch scripts/hellbender_recipe_sweep.slurm
#
# Six variants x two seeds, all on the full 144-image train split and the same
# 19 validation images. Every variant uses flags that already exist, so nothing
# here changes training behaviour for other callers.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting}"
DATASET_DIR="${DATASET_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data/sundew-segmenting-reviewed-v0.3.0}"
CACHE_ROOT="${CACHE_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache}"
ENV_NAME="${ENV_NAME:-sundew-seg}"
MODEL_ROOT="${MODEL_ROOT:-models/recipe}"

# name|extra flags
VARIANTS=(
  "baseline|--image-size 768 --epochs 30 --patience 5 --learning-rate 1e-3 --loss bce-tversky"
  "res1024|--image-size 1024 --epochs 30 --patience 5 --learning-rate 1e-3 --loss bce-tversky"
  "long|--image-size 768 --epochs 80 --patience 15 --learning-rate 1e-3 --loss bce-tversky"
  "dice|--image-size 768 --epochs 30 --patience 5 --learning-rate 1e-3 --loss bce-dice"
  "lowlr-long|--image-size 768 --epochs 80 --patience 15 --learning-rate 3e-4 --loss bce-tversky"
  "combined|--image-size 1024 --epochs 80 --patience 15 --learning-rate 3e-4 --loss bce-tversky"
)
SEEDS=(17 101)

ENTRY="${VARIANTS[$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))]}"
NAME="${ENTRY%%|*}"
FLAGS="${ENTRY#*|}"
SEED="${SEEDS[$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))]}"
OUT_DIR="$MODEL_ROOT/$NAME/seed-$SEED"

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

echo "=== node=$(hostname) variant=$NAME seed=$SEED flags=$FLAGS ==="

python scripts/train_baseline.py \
  --model segformer-b0 \
  --images "$DATASET_DIR/images" \
  --masks "$DATASET_DIR/masks" \
  --metadata "$DATASET_DIR/metadata/training-records.jsonl" \
  --output "$OUT_DIR" \
  --batch-size 8 \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --no-allow-partial \
  $FLAGS
