#!/bin/bash
#SBATCH --job-name=sundew-curve
#SBATCH --array=0-11%6
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Data-scaling curve: does validation IoU still climb at 144 training images?
#
#   sbatch scripts/hellbender_data_curve.slurm
#
# Trains SegFormer-B0 (the better baseline) on 25/50/75/100% of the train
# split, three seeds each. If the curve is still rising at 100%, more annotated
# images buy accuracy; if it has flattened, the bottleneck is elsewhere.
#
# Validation is never subsampled, so all twelve runs are scored on the same 19
# images and the points are comparable.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting}"
DATASET_DIR="${DATASET_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data/sundew-segmenting-reviewed-v0.3.0}"
CACHE_ROOT="${CACHE_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache}"
ENV_NAME="${ENV_NAME:-sundew-seg}"
MODEL_ROOT="${MODEL_ROOT:-models/curve}"
MODEL="${MODEL:-segformer-b0}"

FRACTIONS=(0.25 0.50 0.75 1.00)
SEEDS=(17 101 202)

FRACTION="${FRACTIONS[$(( SLURM_ARRAY_TASK_ID / ${#SEEDS[@]} ))]}"
SEED="${SEEDS[$(( SLURM_ARRAY_TASK_ID % ${#SEEDS[@]} ))]}"
OUT_DIR="$MODEL_ROOT/frac-$FRACTION/seed-$SEED"

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

echo "=== node=$(hostname) model=$MODEL fraction=$FRACTION seed=$SEED ==="

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
  --train-fraction "$FRACTION" \
  --no-allow-partial
