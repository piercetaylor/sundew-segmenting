"""Pre-registered diagnostics of the distillation arms, on validation, last epoch.

Usage: python scripts/diagnose_species_distill.py <model tag> ... > out.json

For each tag (arm `full`, five seeds), per-seed values averaged over seeds:
single-model ECE (15 bins) before and after one temperature fitted on
validation NLL, mean confidence, plain accuracy, top-5, the share of
predictions going to species with 130+ training images, and fidelity to the
teacher (top-1 agreement and KL(teacher || student), overall and per
training-count bin) where a KD run stored `teacher-val.npz`. Per-species
accuracy is averaged over seeds and binned by training count; every pair of
tags gets a per-bin paired difference with an observer-grouped bootstrap
interval (every seed rescored on the same resample). The held-out test split
is not read.
"""
import json, pathlib, sys
from collections import Counter
import numpy as np
sys.path.insert(0, "scripts")
from analyze_species_teacher import softmax, ece, tempered, CORPUS

FT = pathlib.Path("models/species-110/finetune")
SEEDS = (17, 101, 202, 303, 404)
BINS = [(0, 40, "under 40"), (40, 80, "40-79"), (80, 130, "80-129"), (130, 10**9, "130 and over")]
REPS = 1000

labels = json.loads((CORPUS / "labels.json").read_text())
n = len(labels)
rows = [json.loads(l) for l in open(CORPUS / "species-records.jsonl") if l.strip()]
tc = Counter(r["label"] for r in rows if r["split"] == "train")
train = np.array([tc[l] for l in labels])
common = train >= 130

teacher = None
def load(tag):
    global teacher
    runs = [np.load(FT / tag / "full" / f"seed-{s}" / "predictions-last.npz") for s in SEEDS]
    tv = FT / tag / "full" / "seed-17" / "teacher-val.npz"
    if tv.exists() and teacher is None:
        teacher = np.load(tv)["probs"]
    loss = [json.load(open(FT / tag / "full" / f"seed-{s}" / "classifier-metrics.json"))["last"]["train_loss"] for s in SEEDS]
    return [softmax(z["scores"].astype(np.float64)) for z in runs], runs[0]["val_label"], runs[0]["val_observer"], loss

def per_class_acc(pred, y, w):
    tot = np.bincount(y, weights=w, minlength=n)
    ok = np.bincount(y, weights=w * (pred == y), minlength=n)
    return np.where(tot > 0, ok / np.maximum(tot, 1e-12), np.nan)

tags = sys.argv[1:]
data = {t: load(t) for t in tags}
y, obs = data[tags[0]][1], data[tags[0]][2]
assert all(np.array_equal(d[1], y) for d in data.values())
uobs, oidx = np.unique(obs, return_inverse=True)
grid = np.round(np.arange(0.30, 3.001, 0.01), 2)

out = {}
for t in tags:
    probs, _, _, loss = data[t]
    r = {"ece": [], "ece_t": [], "temp": [], "conf": [], "acc": [], "top5": [], "share": [], "agree": [], "kl": [], "kl_bin": []}
    for p in probs:
        pred = p.argmax(1); logp = np.log(np.clip(p, 1e-12, None))
        nll = [-np.mean(np.log(np.clip(tempered(logp, g)[np.arange(len(y)), y], 1e-12, None))) for g in grid]
        tb = grid[int(np.argmin(nll))]
        r["ece"].append(ece(p, y)); r["ece_t"].append(ece(tempered(logp, tb), y)); r["temp"].append(tb)
        r["conf"].append(p.max(1).mean()); r["acc"].append(np.mean(pred == y)); r["top5"].append(np.mean((np.argsort(-p, 1)[:, :5] == y[:, None]).any(1))); r["share"].append(np.mean(common[pred]))
        if teacher is not None:
            r["agree"].append(np.mean(pred == teacher.argmax(1)))
            kl = np.sum(teacher * (np.log(np.clip(teacher, 1e-12, None)) - logp), 1)
            r["kl"].append(kl.mean())
            r["kl_bin"].append([kl[(train[y] >= lo) & (train[y] < hi)].mean() for lo, hi, _ in BINS])
    pc = np.mean([per_class_acc(p.argmax(1), y, np.ones(len(y))) for p in probs], 0)
    out[t] = {k: (np.mean(v, 0).tolist() if v else None) for k, v in r.items()}
    out[t]["true_share"] = float(np.mean(common[y]))
    out[t]["train_loss"] = float(np.mean(loss))
    out[t]["bins"] = [float(np.nanmean(pc[(train >= lo) & (train < hi)])) for lo, hi, _ in BINS]
    out[t]["bin_species"] = [int(((train >= lo) & (train < hi)).sum()) for lo, hi, _ in BINS]

# Paired per-bin differences, observer bootstrap, every seed rescored on the same resample.
rng = np.random.default_rng(20260928)
preds = {t: [p.argmax(1) for p in data[t][0]] for t in tags}
boot = {t: [] for t in tags}
for _ in range(REPS):
    k = np.bincount(rng.integers(0, len(uobs), len(uobs)), minlength=len(uobs))[oidx].astype(float)
    for t in tags:
        pc = np.nanmean([per_class_acc(pr, y, k) for pr in preds[t]], 0)
        boot[t].append([np.nanmean(pc[(train >= lo) & (train < hi)]) for lo, hi, _ in BINS])
diffs = {}
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        d = np.array(boot[a]) - np.array(boot[b])
        pt = np.array(out[a]["bins"]) - np.array(out[b]["bins"])
        diffs[f"{a} - {b}"] = [[float(pt[j]), *np.nanpercentile(d[:, j], [2.5, 97.5]).tolist()] for j in range(len(BINS))]
json.dump({"arms": out, "bin_diffs": diffs, "bins": [b[2] for b in BINS], "reps": REPS}, sys.stdout, indent=1)
