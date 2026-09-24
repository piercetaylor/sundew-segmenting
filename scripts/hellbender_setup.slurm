#!/bin/bash
#SBATCH --job-name=sundew-setup
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# One-time environment + data setup for the sundew baselines.
# Runs on a compute node because Hellbender policy forbids running work on the
# login node, and because a full torch install is heavy.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting}"
DATA_ROOT="${DATA_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-data}"
CACHE_ROOT="${CACHE_ROOT:-/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-cache}"
ENV_NAME="${ENV_NAME:-sundew-seg}"

cd "$REPO_DIR"
mkdir -p "$DATA_ROOT" "$CACHE_ROOT"

echo "=== node: $(hostname) ==="

# The mamba module only puts conda/mamba on PATH; the shell hook defines
# `conda activate`. It touches unset variables, so relax `-u` around it.
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
set -u

# --- environment -----------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "=== updating existing env '$ENV_NAME' ==="
  mamba env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "=== creating env '$ENV_NAME' ==="
  mamba env create -f environment.yml
fi

set +u
conda activate "$ENV_NAME"
set -u
echo "python: $(which python) ($(python -V 2>&1))"

# Install the repo package itself without touching the resolved dependency set.
pip install --no-deps -e .

# --- dataset ---------------------------------------------------------------
echo "=== dataset ==="
if [ ! -d "$DATA_ROOT/sundew-segmenting-reviewed-v0.3.0" ]; then
  # The checksum file was written on Windows; strip CR before checking.
  ( cd "$REPO_DIR/dist" && tr -d '\r' < sundew-segmenting-reviewed-v0.3.0.zip.sha256 | sha256sum --check - )
  unzip -q -d "$DATA_ROOT" "$REPO_DIR/dist/sundew-segmenting-reviewed-v0.3.0.zip"
fi
for split in train validation test; do
  echo "  $split: $(ls "$DATA_ROOT/sundew-segmenting-reviewed-v0.3.0/images/$split" | wc -l) images, $(ls "$DATA_ROOT/sundew-segmenting-reviewed-v0.3.0/masks/$split" | wc -l) masks"
done

# --- pretrained weights ----------------------------------------------------
# Warm the caches here so the training job does not depend on the network.
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
mkdir -p "$TORCH_HOME" "$HF_HOME"

echo "=== warming pretrained weight caches ==="
python - <<'PY'
import segmentation_models_pytorch as smp
import torch
print("torch", torch.__version__, "| smp", smp.__version__)
for arch, encoder in (("Unet", "resnet34"), ("Segformer", "mit_b0")):
    getattr(smp, arch)(encoder_name=encoder, encoder_weights="imagenet", classes=1, activation=None)
    print(f"  cached {arch}/{encoder}")
PY

echo "=== SETUP_DONE ==="
