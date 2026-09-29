#!/bin/bash
#SBATCH --job-name=sundew-distill-smoke
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Smoke test for the distillation path of finetune_species_backbone.py, before
# the KD-25 array (docs/species-classifier-plan.md, "Distillation into
# DINOv2-S"). On 512 images per split of split-110-test, dinov2-s full, seed 17:
#   1. a label-trained run and a KD run start from the same student weights;
#   2. the rebuilt teacher ensemble matches the saved best-epoch predictions;
#   3. the KD loss falls, and last-epoch weights are written;
#   4. a KD run killed after epoch 1 resumes at epoch 2.
# Writes only to models/species-110/finetune-smoke-kd/, which is safe to delete.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110-test"
OUT=models/species-110/finetune-smoke-kd
TEACHERS=()
for s in 17 101 202 303 404; do TEACHERS+=("models/species-110/finetune/dinov2-l-reg/full/seed-$s/classifier-best.pt"); done

cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src TORCH_HOME="$BASE/sundew-cache/torch" HF_HOME="$BASE/sundew-cache/huggingface" HF_HUB_OFFLINE=1
rm -rf "$OUT"; mkdir -p "$OUT"

run() {  # name epochs [extra args]
  local name=$1 epochs=$2; shift 2
  python scripts/finetune_species_backbone.py --model dinov2-s --arm full --seed 17 --epochs "$epochs" \
    --warmup-epochs 1 --records "$SPLIT/species-records.jsonl" --labels "$SPLIT/labels.json" \
    --sections data/species-110-sections.json --crop-dir "$CORPUS/crops/crops" \
    --output "$OUT/$name" --limit 512 --batch-size 64 --layer-decay 0.75 --drop-path 0.1 \
    --patience "$epochs" --workers 7 "$@"
}

echo "--- label-trained"
run ce 2 | tee "$OUT/ce.log"
echo "--- KD"
run kd 2 --teacher-checkpoints "${TEACHERS[@]}" --kd-tau 1.0 | tee "$OUT/kd.log"

a=$(grep "student init sha256" "$OUT/ce.log"); b=$(grep "student init sha256" "$OUT/kd.log")
[ -n "$a" ] && [ "$a" = "$b" ] && echo "INIT OK: $a" || { echo "INIT DIFFERS: $a / $b"; exit 1; }

python - "$OUT" <<'PY'
import json, sys, numpy as np
out = sys.argv[1]
t = np.load(f"{out}/kd/teacher-val-smoke.npz")
ids = t["val_photo_id"]
def sm(s): e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
saved = []
for s in (17, 101, 202, 303, 404):
    z = np.load(f"models/species-110/finetune/dinov2-l-reg/full/seed-{s}/predictions-best.npz")
    pos = {p: i for i, p in enumerate(z["val_photo_id"])}
    saved.append(sm(z["scores"][[pos[p] for p in ids]].astype(np.float64)))
saved = np.mean(saved, 0)
agree = float(np.mean(saved.argmax(1) == t["probs"].argmax(1)))
maxdiff = float(np.abs(saved - t["probs"]).max())
print(f"teacher rebuilt vs saved predictions, {len(ids)} images: top-1 agreement {agree:.4f}, max prob diff {maxdiff:.4f}")
m = json.load(open(f"{out}/kd/classifier-metrics-smoke.json"))
h = m["history"]
print("KD history:", [(x["epoch"], round(x["train_loss"], 4), round(x["val_teacher_agreement"], 3)) for x in h])
ok = agree >= 0.99 and m["loss"] == "kd" and h[-1]["train_loss"] < h[0]["train_loss"]
import os
ok &= os.path.exists(f"{out}/kd/classifier-last-smoke.pt")
print("TEACHER/LOSS OK" if ok else "TEACHER/LOSS FAILED")
sys.exit(0 if ok else 1)
PY

echo "--- resume check (KD)"
LOG="$OUT/resume.log"
run kd-resume 3 --teacher-checkpoints "${TEACHERS[@]}" > "$LOG" 2>&1 &
PID=$!
until [ -e "$OUT/kd-resume/last-smoke.pt" ]; do kill -0 $PID 2>/dev/null || { cat "$LOG"; exit 1; }; sleep 2; done
pkill -TERM -P $PID || true; kill $PID 2>/dev/null || true; wait $PID || true
sleep 5; pgrep -f "finetune_species_backbone.py --model dinov2-s .*kd-resume" && { echo "first run still alive"; exit 1; }
run kd-resume 3 --teacher-checkpoints "${TEACHERS[@]}" | tee -a "$LOG"
grep -q "resumed after epoch 1" "$LOG" && grep -q '"epoch": 3' "$LOG" && echo "RESUME OK" || { echo "RESUME FAILED"; exit 1; }
echo "=== DISTILL SMOKE PASSED ==="
