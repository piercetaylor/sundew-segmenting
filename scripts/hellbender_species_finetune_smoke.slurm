#!/bin/bash
#SBATCH --job-name=sundew-finetune-smoke
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Smoke test for finetune_species_backbone.py before the real array: every
# code path (timm ConvNeXt, timm ViT, open_clip ViT; all three arms) on 512
# images per split, then a resume check: kill a run after its first epoch,
# restart it, and require that it picks up at epoch 2. Writes only to
# models/species-110/finetune-smoke/, which is safe to delete.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110"
OUT=models/species-110/finetune-smoke

cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src TORCH_HOME="$BASE/sundew-cache/torch" HF_HOME="$BASE/sundew-cache/huggingface" HF_HUB_OFFLINE=1
rm -rf "$OUT"

run() {  # model arm epochs
  python scripts/finetune_species_backbone.py --model "$1" --arm "$2" --epochs "$3" --warmup-epochs 1 \
    --records "$SPLIT/species-records.jsonl" --labels "$SPLIT/labels.json" \
    --sections data/species-110-sections.json --crop-dir "$CORPUS/crops/crops" \
    --output "$OUT/$1-$2" --limit 512 --batch-size 64 --workers 7
}

for spec in "convnext-b-in22k full" "dinov2-l-reg full-square" "bioclip-2 crop"; do
  set -- $spec
  echo "--- $1 $2"
  run "$1" "$2" 2
done

echo "--- resume check"
LOG="$OUT/resume.log"
run dinov2-b crop 3 > "$LOG" 2>&1 &
PID=$!
# Kill only once epoch 1's state is on disk; the epoch line prints before it is written.
until [ -e "$OUT/dinov2-b-crop/last-smoke.pt" ]; do kill -0 $PID 2>/dev/null || { cat "$LOG"; exit 1; }; sleep 2; done
# $PID is the backgrounded shell; kill its python child too, then make sure nothing is left.
pkill -TERM -P $PID || true; kill $PID 2>/dev/null || true; wait $PID || true
sleep 5; pgrep -f "finetune_species_backbone.py --model dinov2-b" && { echo "first run still alive"; exit 1; }
run dinov2-b crop 3 | tee -a "$LOG"
grep -q "resumed after epoch 1" "$LOG" && grep -q '"epoch": 3' "$LOG" && echo "RESUME OK" || { echo "RESUME FAILED"; exit 1; }
test ! -e "$OUT/dinov2-b-crop/last-smoke.pt" && echo "last.pt cleaned up"
echo "=== SMOKE PASSED ==="
