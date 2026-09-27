#!/bin/bash
#SBATCH --job-name=sundew-kdw-smoke
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Smoke test for class-weighted KD on the short-side-576 cache, before the
# cache arms (docs/species-classifier-plan.md, "Class-weighted KD"):
#   1. the teacher ensemble scores 0.834 +- 0.002 on the full cached validation;
#   2. a KDw run (label weights) starts from the same student weights as every
#      earlier seed-17 run, its loss falls, and it records its weight rule;
#   3. a KDwe run (expected weights, the rule for unlabeled photos) runs, and
#      the weight diagnostics are logged.
# Writes only to models/species-110/finetune-smoke-kdw/, which is safe to delete.

set -euo pipefail

BASE=/cluster/VAST/mendozacozatld-lab/PierceTaylor
CORPUS="$BASE/sundew-species-corpus"
SPLIT="$CORPUS/split-110-test"
OUT=models/species-110/finetune-smoke-kdw
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

echo "--- teacher ensemble on the cached validation frames"
python - "$SPLIT" "$CORPUS/full-s576" "${TEACHERS[@]}" <<'PY'
import json, sys, pathlib, numpy as np, torch
from PIL import Image
from torchvision import transforms
from finetune_species_backbone import build
from screen_species_backbones import balanced_accuracy
split, cache, paths = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
labels = json.loads((split / "labels.json").read_text()); idx = {l: i for i, l in enumerate(labels)}
val = [json.loads(l) for l in open(split / "species-records.jsonl") if l.strip()]
val = [r for r in val if r["split"] == "validation"]
y = np.array([idx[r["label"]] for r in val])
nets = []
for p in paths:
    st = torch.load(p, map_location="cpu", weights_only=False)
    net, mean, std, _ = build(st["model"], len(labels), 0.0, 224, "cuda")
    net.load_state_dict(st["model_state"]); nets.append(net.eval())
tf = transforms.Compose([transforms.Resize(int(224 * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
class DS(torch.utils.data.Dataset):
    def __len__(self): return len(val)
    def __getitem__(self, i):
        with Image.open(cache / f"inat_{val[i]['photo_id']}.jpg") as im:
            return tf(im.convert("RGB"))
probs = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for x in torch.utils.data.DataLoader(DS(), batch_size=128, num_workers=7):
        x = x.cuda()
        probs.append(torch.stack([torch.softmax(n(x).float(), -1) for n in nets]).mean(0).cpu().numpy())
b = balanced_accuracy(y, np.concatenate(probs).argmax(1), len(labels))
print(f"teacher ensemble, cached validation ({len(val)} images): balanced accuracy {b:.4f}")
sys.exit(0 if abs(b - 0.834) <= 0.002 else 1)
PY
echo "TEACHER ON CACHE OK"

run() {  # name epochs [extra args]
  local name=$1 epochs=$2; shift 2
  python scripts/finetune_species_backbone.py --model dinov2-s --arm full --seed 17 --epochs "$epochs" \
    --warmup-epochs 1 --records "$SPLIT/species-records.jsonl" --labels "$SPLIT/labels.json" \
    --sections data/species-110-sections.json --crop-dir "$CORPUS/crops/crops" --full-dir "$CORPUS/full-s576" \
    --output "$OUT/$name" --limit 512 --batch-size 64 --layer-decay 0.75 --drop-path 0.1 \
    --patience "$epochs" --workers 7 --teacher-checkpoints "${TEACHERS[@]}" --kd-tau 1.0 "$@"
}

echo "--- KDw (label weights)"
run kdw 2 --kd-class-weight label | tee "$OUT/kdw.log"
grep -q "student init sha256 $INIT_SHA" "$OUT/kdw.log" && echo "INIT OK" || { echo "INIT DIFFERS"; exit 1; }
echo "--- KDwe (expected weights)"
run kdwe 1 --kd-class-weight expected | tee "$OUT/kdwe.log"

python - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
m = json.load(open(f"{out}/kdw/classifier-metrics-smoke.json")); h = m["history"]
e = json.load(open(f"{out}/kdwe/classifier-metrics-smoke.json"))
print("KDw history:", [(x["epoch"], round(x["train_loss"], 4), round(x["train_weight_absdiff"], 3),
                        round(x["train_teacher_label_disagree"], 3)) for x in h])
ok = (m["kd"]["class_weight"] == "label" and e["kd"]["class_weight"] == "expected"
      and m["full_dir"] and h[-1]["train_loss"] < h[0]["train_loss"]
      and all(k in h[-1] for k in ("train_weight_absdiff", "val_teacher_agreement")))
print("KDW OK" if ok else "KDW FAILED"); sys.exit(0 if ok else 1)
PY
echo "=== KDW SMOKE PASSED ==="
