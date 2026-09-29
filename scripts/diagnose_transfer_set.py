"""Pre-registered transfer-set diagnostics of KDw-100c+T (docs/species-classifier-plan.md, "Transfer set").

Usage: PYTHONPATH=src:scripts python scripts/diagnose_transfer_set.py [--out-json J] [--out-npz N]

Runs the teacher (mean softmax of the five dinov2-l-reg full best-epoch
checkpoints, tau 1) and the shipped student (dinov2-s-kdw-e100-t-c576, seed 17,
last epoch) over the 12,126 transfer photos, with the validation transform of
finetune_species_backbone.py (224 px, Resize 255 + CenterCrop, bf16 autocast).
By source ((a) wild research grade, (b) captive, (c) needs-ID or casual, all)
and by the training count of the iNaturalist-ID species (under 40 / 40-79 /
80-129 / 130 and over): n, teacher mean max-probability, teacher top-1
agreement with the iNaturalist ID, student-teacher top-1 agreement, student
agreement with the iNaturalist ID, and mean KL(teacher || student); also the
share of the set by source x bin. The iNaturalist ID is the record's taxon_name;
infraspecific names map to their species (the taxon the photo was queried
under) and are counted in the output. The captive row of the decision rule is
teacher agreement on source (b) against 0.6.

As a check that the checkpoints and preprocessing match training, the teacher
is also run on validation and compared with the teacher-val.npz stored by the
+T seed-17 run. The held-out test split is not read: its rows are dropped from
the records before anything else.
"""
import argparse, json, pathlib, sys
from collections import Counter
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0, "scripts")
from finetune_species_backbone import build
from screen_species_backbones import metrics
from analyze_species_teacher import CORPUS

BASE = pathlib.Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor")
TSET = BASE / "sundew-transfer-set"
FT = pathlib.Path("models/species-110/finetune")
TEACHERS = [FT / "dinov2-l-reg" / "full" / f"seed-{s}" / "classifier-best.pt" for s in (17, 101, 202, 303, 404)]
STUDENT = FT / "dinov2-s-kdw-e100-t-c576" / "full" / "seed-17"
BINS = [(0, 40, "under 40"), (40, 80, "40-79"), (80, 130, "80-129"), (130, 10**9, "130 and over")]
SOURCES = {"a": "wild, research grade", "b": "captive", "c": "needs-ID or casual"}

p = argparse.ArgumentParser()
p.add_argument("--out-json", type=pathlib.Path, default=pathlib.Path("reports/species-distill/t-transfer-diagnostics.json"))
p.add_argument("--out-npz", type=pathlib.Path, default=STUDENT / "transfer-diagnostics.npz")
p.add_argument("--batch-size", type=int, default=128)
args = p.parse_args()

labels = json.loads((CORPUS / "labels.json").read_text()); n = len(labels)
index = {l: i for i, l in enumerate(labels)}
rows = [r for r in map(json.loads, filter(str.strip, open(CORPUS / "species-records.jsonl"))) if r["split"] != "test"]
tc = Counter(r["label"] for r in rows if r["split"] == "train")
train = np.array([tc[l] for l in labels])
val = [r for r in rows if r["split"] == "validation"]

T = [json.loads(l) for l in open(TSET / "transfer-records.jsonl") if l.strip()]
remapped = Counter()
def species(name):
    if name in index:
        return index[name]
    sp = " ".join(name.split()[:2])
    if sp not in index:
        raise SystemExit(f"transfer taxon {name!r} maps to no label")
    remapped[f"{name} -> {sp}"] += 1
    return index[sp]
ty = np.array([species(r["taxon_name"]) for r in T])
src = np.array([r["source"] for r in T])
tbin = np.digitize(train[ty], [40, 80, 130])
print(f"transfer {len(T)}  sources {dict(Counter(r['source'] for r in T))}  infraspecific remapped {sum(remapped.values())}", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
def load(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    assert st["labels"] == labels and st["image_size"] == 224, path
    net, mean, std, _ = build(st["model"], n, 0.0, 224, device)
    net.load_state_dict(st["model_state"]); net.eval().requires_grad_(False)
    return net, (tuple(mean), tuple(std))
teachers = [load(c) for c in TEACHERS]
student, norm = load(STUDENT / "classifier-last.pt")
assert all(t[1] == norm for t in teachers), "normalisation differs between teacher and student"
teachers = [t[0] for t in teachers]
tf = transforms.Compose([transforms.Resize(int(224 * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
                         transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(*norm)])

class DS(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return tf(im.convert("RGB"))

def run(paths, with_student=True):
    tp, sl = [], []
    dl = torch.utils.data.DataLoader(DS(paths), batch_size=args.batch_size, num_workers=7, pin_memory=True)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for x in dl:
            x = x.to(device, non_blocking=True)
            tp.append(torch.stack([torch.softmax(t(x).float(), -1) for t in teachers]).mean(0).cpu().numpy())
            if with_student:
                sl.append(student(x).float().cpu().numpy())
    return np.concatenate(tp), (np.concatenate(sl) if with_student else None)

# Check: the teacher on validation (cached frames, as the +T run read them) against its stored teacher-val.npz.
ref = np.load(STUDENT / "teacher-val.npz")
assert np.array_equal(ref["val_photo_id"], [r["photo_id"] for r in val])
vt, _ = run([str(CORPUS.parent / "full-s576" / f"inat_{r['photo_id']}.jpg") for r in val], with_student=False)
vy = np.array([index[r["label"]] for r in val])
check = {"val_balanced_accuracy": metrics(vy, np.log(np.clip(vt, 1e-12, None)), n, np.zeros(n))["balanced_accuracy"],
         "max_abs_prob_diff_vs_teacher_val_npz": float(np.abs(vt - ref["probs"]).max()),
         "top1_match_vs_teacher_val_npz": float(np.mean(vt.argmax(1) == ref["probs"].argmax(1)))}
print("teacher check", check, flush=True)

tprob, slogit = run([str(TSET / "full-s576" / f"inat_{r['photo_id']}.jpg") for r in T])
slogp = slogit - slogit.max(1, keepdims=True); slogp -= np.log(np.exp(slogp).sum(1, keepdims=True))
kl = np.sum(tprob * (np.log(np.clip(tprob, 1e-12, None)) - slogp), 1)
t1, s1 = tprob.argmax(1), slogit.argmax(1)
np.savez_compressed(args.out_npz, photo_id=np.array([r["photo_id"] for r in T]), source=src, inat_label=ty,
                    teacher_top1=t1, teacher_maxprob=tprob.max(1).astype(np.float32), teacher_probs=tprob.astype(np.float32),
                    student_top1=s1, student_maxprob=np.exp(slogp.max(1)).astype(np.float32),
                    student_logits=slogit.astype(np.float32), kl=kl.astype(np.float32))

def stats(m):
    if not m.any():
        return {"n": 0}
    return {"n": int(m.sum()), "teacher_maxprob": float(tprob.max(1)[m].mean()), "teacher_inat_agree": float(np.mean(t1[m] == ty[m])),
            "student_teacher_agree": float(np.mean(s1[m] == t1[m])), "student_inat_agree": float(np.mean(s1[m] == ty[m])),
            "kl": float(kl[m].mean()), "species": int(len(np.unique(ty[m])))}
groups = {**{s: src == s for s in SOURCES}, "all": np.ones(len(T), bool)}
out = {"teacher_check": check, "n": len(T), "sources": SOURCES, "bins": [b[2] for b in BINS],
       "infraspecific_remapped": dict(remapped),
       "by_source": {g: stats(m) for g, m in groups.items()},
       "by_bin": {b[2]: stats(tbin == j) for j, b in enumerate(BINS)},
       "by_source_bin": {g: {b[2]: stats(m & (tbin == j)) for j, b in enumerate(BINS)} for g, m in groups.items()},
       "share_by_source_bin": {g: [float(np.mean(m & (tbin == j))) for j in range(len(BINS))] for g, m in groups.items()},
       "per_species": {labels[c]: {"train": int(train[c]), **{s: int(np.sum((ty == c) & (src == s))) for s in SOURCES}}
                       for c in range(n)},
       "captive_rule": {"threshold": 0.6, "teacher_inat_agree": stats(src == "b")["teacher_inat_agree"],
                        "fires": bool(stats(src == "b")["teacher_inat_agree"] < 0.6)}}
args.out_json.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("by_source", "by_bin", "captive_rule")}, indent=1))
