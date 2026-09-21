#!/bin/bash
#SBATCH --job-name=sundew-species-control
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out

# Control arm for docs/crop-readiness-plan.md step 2.
#
# The crop arm beat full frames by +0.025 balanced accuracy, but the advantage
# did not grow as the predicted box got tighter (r = -0.025), which is not what
# a cropping benefit looks like. The suspected cause is preprocessing: crops
# were saved as 768px LANCZOS thumbnails while full frames kept up to 1600px,
# so the crop arm got a two-stage downsample before the 224px resize.
#
# This arm is UNCROPPED frames put through the identical thumbnail call.
#   crop vs full768  isolates cropping
#   full768 vs full  isolates resampling
#
# It reuses --arm crop because that path reads by photo_id from --crop-dir;
# only the directory differs. The output directory records which it was.

set -euo pipefail
BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
SET="$BASE/sundew-species-set"
SEEDS=(17 101 202 303 404)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"

echo "=== node $(hostname) | arm=full768 (uncropped, 768px) seed=$SEED ==="
python scripts/train_species_classifier.py \
  --records "$SET/species-records.jsonl" --labels "$SET/labels.json" \
  --arm crop --crop-dir "$SET/full768" \
  --output "models/species-compare/full768/seed-$SEED" \
  --epochs 25 --batch-size 32 --image-size 224 \
  --learning-rate 3e-4 --patience 8 --workers 7 --seed "$SEED"
echo "=== done: full768 seed=$SEED ==="
