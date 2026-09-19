#!/bin/bash
#SBATCH --job-name=sundew-baselines
#SBATCH --array=0-1
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Controlled U-Net vs SegFormer comparison on Hellbender.
#
#   sbatch scripts/hellbender_train.slurm [REPO_DIR] [DATASET_DIR]
#
# `logs/` must exist before submitting: Slurm opens the output files itself and
# will fail the job if the directory is missing.
#
# Partition: GPU work is rejected by the `requeue` partition ("GPU jobs are not
# allowed in the 'requeue' partition"), so this uses `gpu_requeue`, which is
# preemptible but schedules immediately. `gpu` is non-preemptible but currently
# backlogged by more than a day; switch with `--partition=gpu` when a run must
# not be interrupted. Both satisfy the documented acceptable use for the GPU
# partitions, since these jobs use the GPU for the majority of the run.
#
# Walltime is sized from a measured run: ~6.3 s/epoch on an A100, so 30
# epochs take about 3 minutes. 30 minutes leaves ample headroom, and a short
# request both schedules sooner and makes preemption unlikely to matter.

set -euo pipefail

REPO_DIR="${1:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting}"
DATASET_DIR="${2:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data/sundew-segmenting-reviewed-v0.3.0}"
CACHE_ROOT="${CACHE_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache}"
ENV_NAME="${ENV_NAME:-sundew-seg}"
# Overridable so a short smoke run (EPOCHS=2 MODEL_ROOT=models/smoke) cannot
# leave a half-trained checkpoint where the real comparison is read from.
MODEL_ROOT="${MODEL_ROOT:-models/hellbender}"

cd "$REPO_DIR"
mkdir -p logs "$MODEL_ROOT"

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
set -u

# Use the caches warmed by hellbender_setup.slurm so the run does not refetch
# pretrained encoders, and keep thread pools inside the CPU allocation.
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Leave one core for the main process; the rest feed the DataLoader.
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

MODELS=(unet-resnet34 segformer-b0)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "=== node=$(hostname) model=$MODEL workers=$WORKERS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_baseline.py \
  --model "$MODEL" \
  --images "$DATASET_DIR/images" \
  --masks "$DATASET_DIR/masks" \
  --metadata "$DATASET_DIR/metadata/training-records.jsonl" \
  --output "$MODEL_ROOT/$MODEL" \
  --epochs "${EPOCHS:-30}" \
  --batch-size 8 \
  --image-size 768 \
  --workers "$WORKERS" \
  --no-allow-partial
