#!/bin/bash
#SBATCH --job-name=sundew-transfer-smoke
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Smoke test for --transfer-records and --steps in finetune_species_backbone.py,
# before the KDw-100c+T arm (docs/species-classifier-plan.md, "Transfer set").
#   PART=A (no transfer photos needed)
#     1. without transfer records nothing changed: a replay of the KDw smoke of
#        job 18014924 has the same student init, the same weight diagnostics and
#        the same loss to 1e-3;
#     2. --steps ends mid-epoch: 12 steps at 8 per epoch give epochs ending at
#        steps 8 and 12.
#   PART=B (after hellbender_transfer_set.slurm: real records and cache)
#     3. a KDw run on train + transfer records: rows without labels enter the
#        batches (share near their share of the union), the loss is finite and
#        falls, and the metrics record the transfer set;
#     4. preempted after epoch 1 and resumed, it replays the same batches: the
#        per-epoch transfer share and weight diagnostics equal an uninterrupted run.
# Writes only to models/species-110/finetune-smoke-transfer/, which is safe to delete.

set -euo pipefail
: "${PART:?pass --export=ALL,PART=A or PART=B}"

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110-test"
TSET="$BASE/sundew-transfer-set"
OUT=models/species-110/finetune-smoke-transfer/part-$PART
INIT_SHA=a56b2c8c3a551083c6b287992c170fd081c6752cb56815ae4b71848126aff526  # seed 17, job 18012907
TEACHERS=()
for s in 17 101 202 303 404; do TEACHERS+=("models/species-110/finetune/dinov2-l-reg/full/seed-$s/classifier-best.pt"); done

cd "$BASE/sundew-segmenting"
module load mamba/v2.4.0
set +u
eval "$(conda shell.bash hook)"
conda activate sundew-seg
set -u
export PYTHONPATH=src:scripts TORCH_HOME="$BASE/sundew-cache/torch" HF_HOME="$BASE/sundew-cache/huggingface" HF_HUB_OFFLINE=1
rm -rf "$OUT"; mkdir -p "$OUT"

run() {  # name [extra args]
  local name=$1; shift
  # exec: every call runs in a pipeline or background subshell, so the pid is python's
  exec python scripts/finetune_species_backbone.py --model dinov2-s --arm full --seed 17 \
    --warmup-epochs 1 --records "$SPLIT/species-records.jsonl" --labels "$SPLIT/labels.json" \
    --sections data/species-110-sections.json --crop-dir "$CORPUS/crops/crops" --full-dir "$CORPUS/full-s576" \
    --output "$OUT/$name" --limit 512 --batch-size 64 --layer-decay 0.75 --drop-path 0.1 \
    --workers 7 --teacher-checkpoints "${TEACHERS[@]}" --kd-tau 1.0 --kd-class-weight label "$@"
}

if [ "$PART" = A ]; then
  echo "--- 1. KDw replay of job 18014924"
  run replay --epochs 2 --patience 2 | tee "$OUT/replay.log"
  grep -q "student init sha256 $INIT_SHA" "$OUT/replay.log" && echo "INIT OK" || { echo "INIT DIFFERS"; exit 1; }
  echo "--- 2. --steps 12"
  run steps --steps 12 --patience 99 | tee "$OUT/steps.log"
  python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
h = json.load(open(f"{out}/replay/classifier-metrics-smoke.json"))["history"]
ref = [(5.250886499881744, 0.13299305271357298), (4.760348081588745, 0.13335878029465675)]  # job 18014924
print("replay:", [(x["train_loss"], x["train_weight_absdiff"]) for x in h])
ok1 = len(h) == 2 and all(abs(x["train_loss"] - l) < 1e-3 and abs(x["train_weight_absdiff"] - w) < 1e-4
                          for x, (l, w) in zip(h, ref))
m = json.load(open(f"{out}/steps/classifier-metrics-smoke.json")); s = m["history"]
print("steps:", [(x["epoch"], x["step"]) for x in s], "recipe steps", m["recipe"]["steps"])
ok2 = [(x["epoch"], x["step"]) for x in s] == [(1, 8), (2, 12)] and m["recipe"]["steps"] == 12
print("REPLAY OK" if ok1 else "REPLAY DIFFERS"); print("STEPS OK" if ok2 else "STEPS FAILED")
sys.exit(0 if ok1 and ok2 else 1)
PY
  echo "=== TRANSFER SMOKE PART A PASSED ==="
  exit 0
fi

echo "--- 3. KDw + transfer, uninterrupted"
TARGS=(--transfer-records "$TSET/transfer-records.jsonl" --transfer-full-dir "$TSET/full-s576" --steps 24 --patience 99)
run whole "${TARGS[@]}" | tee "$OUT/whole.log"
grep -q "student init sha256 $INIT_SHA" "$OUT/whole.log" && echo "INIT OK" || { echo "INIT DIFFERS"; exit 1; }
echo "--- 4. the same, killed after epoch 1, then resumed"
run resumed "${TARGS[@]}" > "$OUT/resumed-1.log" 2>&1 &
pid=$!
until [ -f "$OUT/resumed/last-smoke.pt" ]; do
  kill -0 $pid 2>/dev/null || { echo "run ended before its first checkpoint"; cat "$OUT/resumed-1.log"; exit 1; }
  sleep 1
done
kill $pid; wait $pid || true
run resumed "${TARGS[@]}" | tee "$OUT/resumed-2.log"
grep -q "resumed after epoch" "$OUT/resumed-2.log" && echo "RESUMED" || { echo "DID NOT RESUME"; exit 1; }
python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
a = json.load(open(f"{out}/whole/classifier-metrics-smoke.json"))
b = json.load(open(f"{out}/resumed/classifier-metrics-smoke.json"))
key = lambda h: [(x["epoch"], x["step"], x["train_transfer_share"], round(x["train_weight_absdiff"], 5)) for x in h]
print("whole:  ", key(a["history"])); print("resumed:", key(b["history"]))
print("losses: ", [round(x["train_loss"], 4) for x in a["history"]])
share = sum(x["train_transfer_share"] for x in a["history"]) / len(a["history"])
ok = (a["transfer"]["samples"] == 512 and a["recipe"]["steps"] == 24 and key(a["history"]) == key(b["history"])
      and [x["step"] for x in a["history"]] == [8, 16, 24] and 0.35 < share < 0.65
      and a["history"][-1]["train_loss"] < a["history"][0]["train_loss"])
print(f"mean transfer share {share:.3f} (union share 0.5)")
print("TRANSFER OK" if ok else "TRANSFER FAILED"); sys.exit(0 if ok else 1)
PY
echo "=== TRANSFER SMOKE PART B PASSED ==="
