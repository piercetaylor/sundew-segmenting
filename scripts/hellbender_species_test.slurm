#!/bin/bash
#SBATCH --job-name=sundew-species-test
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Score the species classifiers, scripts/score_species_test.py, under the test
# protocol of docs/species-classifier-plan.md (2026-09-26, amended 2026-09-27
# and 2026-09-29). SPLIT picks the pass:
#
#   SPLIT=val   validation only, reproduces the stored and reported numbers;
#               never opens a test image. SKIP_SHIPPED=1 leaves out the ONNX
#               model when its release.json does not exist yet.
#     sbatch --export=ALL,SPLIT=val scripts/hellbender_species_test.slurm
#   SPLIT=test  THE SINGLE READING of the held-out test split. Runs the
#               validation pass first and stops, test unread, if it fails.
#               Needs models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/release.json.
#     sbatch --export=ALL,SPLIT=test scripts/hellbender_species_test.slurm
#
# Read-once guard: with SPLIT=test the script refuses to start when
# reports/species-test/test-results.json exists. That file is written last and
# atomically, at the very end, so a preempted and requeued task has written
# nothing and reruns the same computation; once it exists nothing reruns.

set -euo pipefail

: "${SPLIT:?pass --export=ALL,SPLIT=val or SPLIT=test}"
EXTRA=()
if [ "${SKIP_SHIPPED:-0}" = 1 ]; then
  [ "$SPLIT" = val ] || { echo "SKIP_SHIPPED=1 is for SPLIT=val only"; exit 1; }
  EXTRA=(--skip-shipped)
fi

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
cd "$BASE/sundew-segmenting"

if [ "$SPLIT" = test ] && [ -e reports/species-test/test-results.json ]; then
  echo "reports/species-test/test-results.json exists: the test split was already scored. Not rerunning."
  exit 0
fi

module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u

export PYTHONPATH=src
export TORCH_HOME="$BASE/sundew-cache/torch"
export HF_HOME="$BASE/sundew-cache/huggingface"
export HF_HUB_OFFLINE=1

echo "=== node $(hostname) | split=$SPLIT ${EXTRA[*]:-} | restart ${SLURM_RESTART_COUNT:-0} | started $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/score_species_test.py --split "$SPLIT" --workers 14 --onnx-threads 8 "${EXTRA[@]}"

echo "=== done: split=$SPLIT | finished $(date -Is) ==="
