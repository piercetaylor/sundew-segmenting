"""Score the held-out test split, once, as pre-registered.

Protocol (docs/species-classifier-plan.md): "Test protocol, fixed 2026-09-26
before any test image is read", its "Amendment to the test protocol,
2026-09-27" (the teacher is scored as its best-epoch ensemble,
classifier-best.pt, because last-epoch teacher weights were never kept), and
"Test protocol for the shipped student (written 2026-09-29 ...)". The test
split (the `split: test` rows of split-110-test/species-records.jsonl, 2,601
images from 1,095 observers) is read exactly once, by `--split test`, after
the ONNX export of the shipped model has written its release.json. Nothing is
retrained or re-tuned afterwards; a test number outside its validation
interval is reported, not corrected. The test is reported with the release
and is not a gate: the release floor stays on validation.

Scored, all at the one reading:

1. teacher: DINOv2-L (`dinov2-l-reg`, `full`), classifier-best.pt, the 5-seed
   ensemble (mean softmax) and the five single seeds. Original JPEGs, as it
   was trained. The ensemble's temperature is the recorded T = 0.72 (224 px,
   reports/species-finetune/teacher-analysis.json);
2. anchor: ResNet-18 on split-110-test (models/species-110-test/resnet18),
   `full` (original JPEGs) and `crop` (sundew-species-corpus/crops/crops),
   five seeds each. Only classifier-best.pt exists (train_species_classifier.py
   keeps no last-epoch weights), so the anchor is scored at its best epoch, as
   its validation numbers (0.4985 / 0.5156) always were;
3. the carried-forward student KDw-100c+T (`dinov2-s-kdw-e100-t-c576`),
   classifier-last.pt: five single seeds and the 5-seed ensemble;
4. CE-100c (`dinov2-s-e100-c576`), classifier-last.pt, five seeds; and the
   student - CE-100c difference of the seed mean, paired by seed;
5. the shipped model: seed 17 of KDw-100c+T as exported
   (models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/release.json:
   int8 or fp32 ONNX, run with onnxruntime on CPU, batch 1 as shipped), with
   its validation-fitted temperature from release.json.

Preprocessing is each model's own validation path. The students (and the
shipped model) were trained and validated on full frames pre-shrunk to short
side 576 px (scripts/make_full_cache.py: LANCZOS, JPEG quality 95, 4:4:4).
The test photos are not in that cache (it was built for train and validation
only), so every full frame for a student is rebuilt here in memory with the
cache's exact code path. On validation the rebuilt JPEG bytes are compared
with the cached files, byte for byte.

Metrics per scored item, on validation and on test: balanced accuracy
(primary), plain accuracy, top-5, section-level balanced accuracy (all as
screen_species_backbones.metrics), ECE (15 bins, analyze_species_teacher.ece)
before and after a temperature fitted on validation (single models: per seed,
grid 0.30-3.00 on NLL, as diagnose_species_distill.py; the teacher ensemble:
the recorded 0.72; the shipped model: release.json), and per-species accuracy
averaged within training-count bins (under 40 / 40-79 / 80-129 / 130+), as
diagnose_species_distill.py. Intervals: observer-grouped bootstrap, 2,000
resamples; a seed mean is rescored seed by seed on the same resample
(summarize_species_finetune.py). Each item's test balanced accuracy is set
against its validation 95% interval and flagged when outside it.

`--split val` scores everything on validation only and checks the recomputed
predictions against the stored ones (predictions-{best,last}.npz, the anchor's
classifier-metrics.json, the export's val-logits.npz) and the reported
numbers; it never opens a test image. `--split test` first does the same
validation pass and stops before opening any test image if a check fails;
then scores the test. It refuses to run if the final output
(reports/species-test/test-results.json) already exists, and needs
release.json. Every output is written only at the end, the JSON last and
atomically, so a job preempted and requeued before that point has written
nothing and simply reruns the same pre-registered computation.

Usage:
  python scripts/score_species_test.py --split val [--skip-shipped]
  python scripts/score_species_test.py --split test
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from analyze_species_teacher import ece, softmax, tempered  # noqa: E402
from screen_species_backbones import metrics  # noqa: E402
from summarize_species_finetune import balanced  # noqa: E402

BASE = pathlib.Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor")
CORPUS = BASE / "sundew-species-corpus"
SPLIT_DIR = CORPUS / "split-110-test"
FT = pathlib.Path("models/species-110/finetune")
ANCHOR_DIR = pathlib.Path("models/species-110-test/resnet18")
RELEASE = pathlib.Path("models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17")
OUT = pathlib.Path("reports/species-test")
LOGITS = pathlib.Path("models/species-110/test-scoring")
SEEDS = (17, 101, 202, 303, 404)
TEACHER, STUDENT, CE = "dinov2-l-reg", "dinov2-s-kdw-e100-t-c576", "dinov2-s-e100-c576"
BINS = [(0, 40, "under 40"), (40, 80, "40-79"), (80, 130, "80-129"), (130, 10**9, "130 and over")]
GRID = np.round(np.arange(0.30, 3.001, 0.01), 2)  # diagnose_species_distill.py
# Teacher ensemble temperature fixed by the 2026-09-26 protocol ("T = 0.72 at 224 px"),
# fitted by analyze_species_teacher.py on the validation ensemble; checked against its JSON below.
TEACHER_T = 0.72
TEACHER_T_SOURCE = pathlib.Path("reports/species-finetune/teacher-analysis.json")
REPS, BOOT_SEED = 2000, 20260926
CACHE_SHORT, CACHE_QUALITY = 576, 95  # make_full_cache.py defaults, as used for full-s576
TOL = 1e-3

# Reported validation numbers the val pass must reproduce (to TOL).
KNOWN = {
    # reports/species-finetune/results.md (best epoch per seed); the best-epoch ensemble is
    # 0.834 in the 2026-09-27 amendment ("0.834 against 0.835" for the last-epoch one)
    "teacher/seed-17": 0.8230, "teacher/seed-101": 0.8229, "teacher/seed-202": 0.8261,
    "teacher/seed-303": 0.8245, "teacher/seed-404": 0.8216, "teacher/ensemble": 0.834,
    # reports/species-110-baseline.md, rerun on split-110-test, best epoch
    "anchor-full/seed-mean": 0.4985, "anchor-crop/seed-mean": 0.5156,
    # reports/species-distill/t-results.md, last epoch
    "student/seed-mean": 0.7684, "student/ensemble": 0.7797, "student/seed-17": 0.7634,
    "ce100c/seed-mean": 0.7189,
}
# reports/species-distill/t-diagnostics.json (diagnose_species_distill.py): seed means.
KNOWN_DIAG = {
    "student": {"ece": 0.10112, "ece_t": 0.02421, "temp": 0.758, "acc": 0.80333, "top5": 0.95893,
                "bins": [0.67679, 0.71912, 0.79151, 0.84200]},
    "ce100c": {"ece": 0.16001, "ece_t": 0.03550, "temp": 0.716, "acc": 0.75812, "top5": 0.92397,
               "bins": [0.61035, 0.66930, 0.75947, 0.79511]},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--skip-shipped", action="store_true",
                   help="val only: leave out the shipped ONNX model (release.json not written yet).")
    p.add_argument("--out-dir", type=pathlib.Path, default=OUT)
    p.add_argument("--logits-dir", type=pathlib.Path, default=LOGITS)
    p.add_argument("--release-dir", type=pathlib.Path, default=RELEASE)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--onnx-threads", type=int, default=8)
    p.add_argument("--reps", type=int, default=REPS)
    return p.parse_args()


# ---------------------------------------------------------------- inference

def load_models(device, release, labels):
    """Every scored network, each with the image view it reads."""
    import torch
    from torch import nn
    from torchvision import models as tvm
    from finetune_species_backbone import build

    n = len(labels)
    nets = []  # (name, view, kind, module-or-session)
    norms = {}

    def timm_net(path, view):
        st = torch.load(path, map_location="cpu", weights_only=False)
        if st["labels"] != labels or st["image_size"] != 224 or st["arm"] != "full":
            raise SystemExit(f"{path}: labels, image size or arm differ")
        net, mean, std, _ = build(st["model"], n, 0.0, 224, device)
        net.load_state_dict(st["model_state"])
        net.eval().requires_grad_(False)
        if norms.setdefault(view, (tuple(mean), tuple(std))) != (tuple(mean), tuple(std)):
            raise SystemExit(f"{path}: normalisation differs within view {view}")
        return net, st["model"]

    for s in SEEDS:
        net, tag = timm_net(FT / TEACHER / "full" / f"seed-{s}" / "classifier-best.pt", "orig")
        assert tag == TEACHER
        nets.append((f"teacher/seed-{s}", "orig", "timm", net))
    for tag, name in ((STUDENT, "student"), (CE, "ce100c")):
        for s in SEEDS:
            d = FT / tag / "full" / f"seed-{s}"
            full_dir = json.loads((d / "classifier-metrics.json").read_text())["full_dir"]
            if full_dir != str(CORPUS / "full-s576"):
                raise SystemExit(f"{d}: trained on {full_dir}, not the 576 px cache")
            net, _ = timm_net(d / "classifier-last.pt", "c576")
            nets.append((f"{name}/seed-{s}", "c576", "timm", net))
    for arm in ("full", "crop"):
        for s in SEEDS:
            st = torch.load(ANCHOR_DIR / arm / f"seed-{s}" / "classifier-best.pt", map_location="cpu", weights_only=False)
            if st["labels"] != labels or st["arm"] != arm:
                raise SystemExit(f"anchor {arm}/{s}: labels or arm differ")
            m = tvm.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, n)
            m.load_state_dict(st["model_state"])
            nets.append((f"anchor-{arm}/seed-{s}", f"anchor_{arm}", "resnet", m.to(device).eval().requires_grad_(False)))
    if release is not None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = release["_threads"], 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(str(release["_onnx"]), so, providers=["CPUExecutionProvider"])
        nets.append(("shipped", "c576_raw", "onnx", sess))
    return nets, norms


class Views:
    """One decode of each photo, every view the scored models read.

    orig      original JPEG, the DINOv2 validation transform (teacher)
    c576      the full-s576 cache rebuilt in memory, same transform (students)
    c576_raw  the same pixels before Normalize (the ONNX graph normalises itself)
    anchor_*  train_species_classifier.py's eval transform on the original / the crop
    """

    def __init__(self, rows, norms, check_cache):
        from torchvision import transforms
        try:
            from finetune_species_backbone import eval_transform
        except ImportError:  # identical geometry, kept for an older checkout
            eval_transform = None
        self.rows, self.check_cache = rows, check_cache
        s = 224
        bicubic = transforms.InterpolationMode.BICUBIC
        if eval_transform is not None:
            t = eval_transform("full", s, *norms["orig"])
            self.geom = transforms.Compose(t.transforms[:-1])  # Resize, CenterCrop, ToTensor
            self.norm_orig = t.transforms[-1]
            self.norm_c576 = eval_transform("full", s, *norms["c576"]).transforms[-1]
        else:
            self.geom = transforms.Compose([transforms.Resize(int(s * 1.14), interpolation=bicubic),
                                            transforms.CenterCrop(s), transforms.ToTensor()])
            self.norm_orig = transforms.Normalize(*norms["orig"])
            self.norm_c576 = transforms.Normalize(*norms["c576"])
        # scripts/train_species_classifier.py eval_tf (default bilinear Resize)
        self.anchor = transforms.Compose([
            transforms.Resize(int(s * 1.14)), transforms.CenterCrop(s), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        r = self.rows[i]
        with Image.open(r["image"]) as im:
            im = im.convert("RGB")
            orig = self.norm_orig(self.geom(im))
            anchor_full = self.anchor(im)
            # make_full_cache.one(), in memory
            w, h = im.size
            if min(w, h) < CACHE_SHORT:
                raise ValueError(f"{r['image']}: short side {min(w, h)} below {CACHE_SHORT}")
            scale = CACHE_SHORT / min(w, h)
            small = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=CACHE_QUALITY, subsampling=0)
            data = buf.getvalue()
        match = -1
        if self.check_cache:
            cached = CORPUS / "full-s576" / f"inat_{r['photo_id']}.jpg"
            match = int(cached.read_bytes() == data) if cached.exists() else -1
        with Image.open(io.BytesIO(data)) as im:
            raw = self.geom(im.convert("RGB"))
        with Image.open(CORPUS / "crops" / "crops" / f"inat_{r['photo_id']}.jpg") as im:
            anchor_crop = self.anchor(im.convert("RGB"))
        return {"orig": orig, "c576": self.norm_c576(raw), "c576_raw": raw,
                "anchor_full": anchor_full, "anchor_crop": anchor_crop, "cache_match": match}


def run(rows, nets, norms, device, workers, check_cache, label):
    import torch
    from torch.utils.data import DataLoader

    # Batch 128 in record order, as finetune_species_backbone.py's validation loader
    # (batch_size * 2); the anchor in chunks of 32 with cuDNN deterministic, as
    # train_species_classifier.py evaluated it.
    dl = DataLoader(Views(rows, norms, check_cache), batch_size=128, shuffle=False, num_workers=workers,
                    pin_memory=True)
    out = {name: [] for name, *_ in nets}
    matches = []
    t0 = time.time()
    for k, b in enumerate(dl):
        matches.append(b["cache_match"].numpy())
        for name, view, kind, net in nets:
            x = b[view]
            if kind == "timm":
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    out[name].append(net(x.to(device, non_blocking=True)).float().cpu().numpy())
            elif kind == "resnet":
                torch.backends.cudnn.deterministic = True
                with torch.no_grad():
                    for j in range(0, len(x), 32):
                        out[name].append(net(x[j:j + 32].to(device)).float().cpu().numpy())
                torch.backends.cudnn.deterministic = False
            else:  # onnx, batch 1 as shipped
                xs = x.numpy()
                out[name].append(np.concatenate([net.run(None, {"pixels": xs[i:i + 1]})[0] for i in range(len(xs))]))
        if k % 5 == 0:
            print(f"  [{label}] {min((k + 1) * 128, len(rows))}/{len(rows)} images, {time.time() - t0:.0f} s", flush=True)
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}, np.concatenate(matches)


# ---------------------------------------------------------------- metrics

class Scorer:
    """Point metrics and observer-bootstrap replicates on one split."""

    def __init__(self, y, observers, n, section_of, train_count, reps, seed):
        self.y, self.n, self.section_of = y, n, section_of
        self.train = train_count
        self.bin_of = [(train_count >= lo) & (train_count < hi) for lo, hi, _ in BINS]
        uniq, idx = np.unique(observers, return_inverse=True)
        self.observers = len(uniq)
        rng = np.random.default_rng(seed)  # as summarize_species_finetune.py
        self.w = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[idx]
                           for _ in range(reps)]).astype(float)

    def per_class(self, hit, w):
        # diagnose_species_distill.per_class_acc: NaN for a class absent from the resample
        tot = np.bincount(self.y, weights=w, minlength=self.n)
        ok = np.bincount(self.y, weights=w * hit, minlength=self.n)
        return np.where(tot > 0, ok / np.maximum(tot, 1e-12), np.nan)

    def hits(self, scores):
        top5 = np.argsort(-scores, axis=1, kind="stable")[:, :5]  # as metrics()
        pred = top5[:, 0]
        return {"pred": pred, "hit": (pred == self.y).astype(float),
                "top5": (top5 == self.y[:, None]).any(1).astype(float),
                "sec": (self.section_of[pred] == self.section_of[self.y]).astype(float)}

    def boot(self, h):
        """(reps, 8): balanced, accuracy, top-5, section balanced, then the four bins."""
        res = np.empty((len(self.w), 8))
        for i, w in enumerate(self.w):
            pc = self.per_class(h["hit"], w)
            res[i, 0] = balanced(self.y, h["pred"], w, self.n)
            res[i, 1] = np.sum(w * h["hit"]) / w.sum()
            res[i, 2] = np.sum(w * h["top5"]) / w.sum()
            res[i, 3] = np.nanmean(self.per_class(h["sec"], w))
            res[i, 4:] = [np.nanmean(pc[m]) for m in self.bin_of]
        return res

    def point(self, scores, prob, logp, temp):
        m = metrics(self.y, scores, self.n, self.section_of)
        h = self.hits(scores)
        pc = self.per_class(h["hit"], np.ones(len(self.y)))
        cal = tempered(logp, temp)
        return {**m, "ece_before": ece(prob, self.y), "ece_after": ece(cal, self.y), "temperature": float(temp),
                "mean_confidence": float(prob.max(1).mean()),
                "bins": [float(np.nanmean(pc[b])) for b in self.bin_of]}


def single_prob(scores):
    """diagnose_species_distill.py: softmax in float64, log-probabilities clipped."""
    p = softmax(scores.astype(np.float64))
    return p, np.log(np.clip(p, 1e-12, None))


def ensemble_prob(score_list):
    """summarize_species_finetune.py / analyze_species_teacher.py: mean of float32 softmax."""
    p = np.mean([softmax(s) for s in score_list], axis=0)
    return p, np.log(np.clip(p, 1e-12, None))


def fit_temperature(logp, y):
    nll = [-np.mean(np.log(np.clip(tempered(logp, g)[np.arange(len(y)), y], 1e-12, None))) for g in GRID]
    return float(GRID[int(np.argmin(nll))])


KEYS = ["balanced_accuracy", "accuracy", "top5_accuracy", "section_balanced_accuracy"]


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def evaluate(sc, logits, temps, release, members):
    """Score every item on one split. temps: per single model, fitted on validation."""
    items, boots = {}, {}
    singles = {}
    for name, s in logits.items():
        prob, logp = single_prob(s)
        t = temps[name]
        singles[name] = sc.point(s, prob, logp, t)
        boots[name] = sc.boot(sc.hits(s))
    for name in logits:
        items[name] = {"kind": "single", **singles[name], "boot": boots[name]}
    for group, names in members.items():
        if group.endswith("/ensemble"):
            prob, logp = ensemble_prob([logits[k] for k in names])
            t = temps[group]
            items[group] = {"kind": "ensemble", "members": names, **sc.point(logp, prob, logp, t),
                            "boot": sc.boot(sc.hits(logp))}
        else:  # seed mean
            vals = [singles[k] for k in names]
            items[group] = {"kind": "seed_mean", "members": names,
                            **{k: float(np.mean([v[k] for v in vals])) for k in
                               KEYS + ["ece_before", "ece_after", "temperature", "mean_confidence"]},
                            "bins": np.mean([v["bins"] for v in vals], 0).tolist(),
                            "seed_sd": float(np.std([v["balanced_accuracy"] for v in vals], ddof=1)),
                            "boot": np.mean([boots[k] for k in names], 0)}
    for it in items.values():
        b = it.pop("boot")
        it["ci95"] = {k: ci(b[:, j]) for j, k in enumerate(KEYS)}
        it["bins_ci95"] = [ci(b[:, 4 + j]) for j in range(len(BINS))]
        it["_boot"] = b
    return items


def paired(items, a, b, per_seed):
    """a - b, seed means paired by seed, observer bootstrap on shared resamples."""
    d = items[a]["_boot"] - items[b]["_boot"]
    point_bins = np.array(items[a]["bins"]) - np.array(items[b]["bins"])
    diffs = [per_seed[0][k] - per_seed[1][k] for k in range(len(SEEDS))]
    return {"delta": items[a]["balanced_accuracy"] - items[b]["balanced_accuracy"], "ci95": ci(d[:, 0]),
            "p_le_0": float(np.mean(d[:, 0] <= 0)), "per_seed": diffs, "wins": int(sum(x > 0 for x in diffs)),
            "bins": [[float(point_bins[j]), *ci(d[:, 4 + j])] for j in range(len(BINS))]}


# ---------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    import torch

    os.chdir(pathlib.Path(__file__).resolve().parent.parent)
    final_json = args.out_dir / ("test-results.json" if args.split == "test" else "val-check.json")
    if args.split == "test":
        if final_json.exists():
            raise SystemExit(f"{final_json} exists: the test split has been scored and is read only once. Refusing.")
        if args.skip_shipped:
            raise SystemExit("--skip-shipped is for the validation check only; the test scores the shipped model")
    release = None
    rel_path = args.release_dir / "release.json"
    if not args.skip_shipped:
        if not rel_path.exists():
            raise SystemExit(f"{rel_path} missing: export the shipped model first (or --skip-shipped for --split val)")
        release = json.loads(rel_path.read_text())
        release["_onnx"] = args.release_dir / release["shipped_file"]
        release["_threads"] = args.onnx_threads
        release["_dir"] = args.release_dir
        if release.get("seed") != 17 or release.get("model") != "dinov2-s" or STUDENT not in release.get("checkpoint", ""):
            raise SystemExit(f"{rel_path}: not seed 17 of {STUDENT}")
        if release["input"]["name"] != "pixels" or not release["input"]["normalisation_in_graph"] \
                or release["input"]["shape"] != [1, 3, 224, 224]:
            raise SystemExit(f"{rel_path}: input contract differs from what this script feeds")
    rec_t = json.loads(TEACHER_T_SOURCE.read_text())["calibration"]["temperature"]
    if rec_t != TEACHER_T:
        raise SystemExit(f"recorded teacher temperature {rec_t} != {TEACHER_T}")

    labels = json.loads((SPLIT_DIR / "labels.json").read_text())
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    sec_map = json.loads(pathlib.Path("data/species-110-sections.json").read_text())["species"]
    section_of = np.array([sec_map[lab]["section"] for lab in labels])
    rows = [json.loads(l) for l in open(SPLIT_DIR / "species-records.jsonl", encoding="utf-8") if l.strip()]
    tc = Counter(r["label"] for r in rows if r["split"] == "train")
    train_count = np.array([tc[lab] for lab in labels])
    val = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"] if args.split == "test" else None
    del rows
    for r in val + (test or []):  # existence only; nothing is decoded here
        for pth in (pathlib.Path(r["image"]), CORPUS / "crops" / "crops" / f"inat_{r['photo_id']}.jpg"):
            if not pth.exists():
                raise SystemExit(f"{pth} missing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    nets, norms = load_models(device, release, labels)
    names = [nm for nm, *_ in nets]
    members = {"teacher/ensemble": [f"teacher/seed-{s}" for s in SEEDS],
               "teacher/seed-mean": [f"teacher/seed-{s}" for s in SEEDS],
               "anchor-full/seed-mean": [f"anchor-full/seed-{s}" for s in SEEDS],
               "anchor-crop/seed-mean": [f"anchor-crop/seed-{s}" for s in SEEDS],
               "student/seed-mean": [f"student/seed-{s}" for s in SEEDS],
               "student/ensemble": [f"student/seed-{s}" for s in SEEDS],
               "ce100c/seed-mean": [f"ce100c/seed-{s}" for s in SEEDS]}
    print(f"split={args.split} device={device} models={len(nets)} shipped={'yes' if release else 'skipped'} "
          f"validation={len(val)} test={'not read' if test is None else len(test)}", flush=True)

    # ---- validation pass: temperatures, validation intervals, reproduction checks
    vy = np.array([index[r["label"]] for r in val])
    v_ids = np.array([r["photo_id"] for r in val])
    v_obs = np.array([r["observer_login"] for r in val])
    v_logits, v_match = run(val, nets, norms, device, args.workers, True, "validation")
    temps = {}
    for nm in names:
        temps[nm] = fit_temperature(single_prob(v_logits[nm])[1], vy)
    if release is not None:
        temps["shipped"] = float(release["temperature"])
    temps["teacher/ensemble"] = TEACHER_T
    temps["student/ensemble"] = fit_temperature(ensemble_prob([v_logits[k] for k in members["student/ensemble"]])[1], vy)
    v_sc = Scorer(vy, v_obs, n, section_of, train_count, args.reps, BOOT_SEED)
    v_items = evaluate(v_sc, v_logits, temps, release, members)

    checks, ok = reproduce(v_logits, v_ids, vy, v_items, v_match, release, members, section_of, n, train_count)
    print(json.dumps({"checks_passed": ok}), flush=True)
    for c in checks:
        print(f"  {'ok ' if c['pass'] else 'FAIL'} {c['what']}: {c['got']} vs {c['want']} {c.get('note', '')}", flush=True)
    platform = cpu_info()
    if release is not None:
        # Dynamic-int8 ONNX Runtime kernels differ by CPU instruction set (AVX2 u8s8 saturates in
        # int16; AVX512-VNNI accumulates in int32), so int8 logits are not bit-identical across CPU
        # families. fp32 is. Both splits of this job run on the same CPU, and the temperature is
        # release.json's; its refit on this CPU's validation logits is recorded beside it.
        platform["release_export_cpu"] = release.get("latency", {}).get("cpu")
        platform["shipped_temperature_refit_here"] = fit_temperature(single_prob(v_logits["shipped"])[1], vy)
    refit_teacher_t = fit_temperature(ensemble_prob([v_logits[k] for k in members["teacher/ensemble"]])[1], vy)

    summary = {
        "protocol": "docs/species-classifier-plan.md: test protocol 2026-09-26, amendment 2026-09-27, "
                    "shipped-student protocol 2026-09-29",
        "split": args.split, "bootstrap_reps": args.reps, "bootstrap_seed": BOOT_SEED,
        "bins": [b[2] for b in BINS], "bin_species": [int(m.sum()) for m in v_sc.bin_of],
        "validation": {"images": len(vy), "observers": v_sc.observers,
                       "cache_bytes_identical": int((v_match == 1).sum()), "cache_bytes_differ": int((v_match == 0).sum()),
                       "cache_missing": int((v_match == -1).sum())},
        "temperatures": temps, "teacher_temperature_source": str(TEACHER_T_SOURCE),
        "teacher_ensemble_temperature_refit_on_best_epoch_validation": refit_teacher_t,
        "release": {k: v for k, v in release.items() if not k.startswith("_")} if release else None,
        "onnx_platform": platform,
        "checks": checks, "checks_passed": ok,
    }
    if args.split == "test" and not ok:
        raise SystemExit("validation reproduction failed; the test split was NOT read. Fix before scoring.")

    v_pairs = paired(v_items, "student/seed-mean", "ce100c/seed-mean",
                     ([v_items[f"student/seed-{s}"]["balanced_accuracy"] for s in SEEDS],
                      [v_items[f"ce100c/seed-{s}"]["balanced_accuracy"] for s in SEEDS]))
    t_items, t_pairs, t_logits = None, None, None
    if args.split == "test":
        # ---- the single reading of the test split
        ty = np.array([index[r["label"]] for r in test])
        t_obs = np.array([r["observer_login"] for r in test])
        t_logits, _ = run(test, nets, norms, device, args.workers, False, "test")
        t_sc = Scorer(ty, t_obs, n, section_of, train_count, args.reps, BOOT_SEED)
        t_items = evaluate(t_sc, t_logits, temps, release, members)
        t_pairs = paired(t_items, "student/seed-mean", "ce100c/seed-mean",
                         ([t_items[f"student/seed-{s}"]["balanced_accuracy"] for s in SEEDS],
                          [t_items[f"ce100c/seed-{s}"]["balanced_accuracy"] for s in SEEDS]))
        summary["test"] = {"images": len(ty), "observers": t_sc.observers,
                           "labels": int(len(np.unique(ty)))}

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    summary["items"] = {}
    for nm, v in v_items.items():
        e = {"validation": strip(v)}
        if t_items is not None:
            t = strip(t_items[nm])
            lo, hi = v["ci95"]["balanced_accuracy"]
            e["test"] = t
            e["test_outside_validation_ci95"] = bool(not lo <= t["balanced_accuracy"] <= hi)
        summary["items"][nm] = e
    summary["student_minus_ce100c"] = {"validation": v_pairs, **({"test": t_pairs} if t_pairs else {})}

    md = render(summary, args.split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.logits_dir.mkdir(parents=True, exist_ok=True)
    stem = "test" if args.split == "test" else "val-check"

    def atomic(path, write):
        tmp = path.with_name(path.name + ".tmp")
        write(tmp)
        tmp.replace(path)

    npz = {"val_photo_id": v_ids, "val_label": vy, "val_observer": v_obs,
           **{f"val/{k}": v for k, v in v_logits.items()}}
    if t_logits is not None:
        npz.update({"test_photo_id": np.array([r["photo_id"] for r in test]), "test_label": ty,
                    "test_observer": t_obs, **{f"test/{k}": v for k, v in t_logits.items()}})
    def save_npz(p):
        with open(p, "wb") as f:
            np.savez_compressed(f, **{k.replace("/", "__"): v for k, v in npz.items()})
    atomic(args.logits_dir / f"{stem}-logits.npz", save_npz)
    atomic(args.out_dir / f"{stem}-results.md" if args.split == "test" else args.out_dir / "val-check.md",
           lambda p: p.write_text(md))
    atomic(final_json, lambda p: p.write_text(json.dumps(summary, indent=2) + "\n"))  # last: the read-once guard
    print(f"wrote {final_json}", flush=True)
    return 0


def cpu_info():
    info = {"cpu": None, "avx512_vnni": None, "avx2": None}
    try:
        txt = open("/proc/cpuinfo").read()
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in txt.splitlines() if l.startswith("model name"))
        flags = next(l for l in txt.splitlines() if l.startswith("flags")).split()
        info["avx512_vnni"], info["avx2"] = "avx512_vnni" in flags, "avx2" in flags
    except (OSError, StopIteration):
        pass
    import onnxruntime as ort
    info["onnxruntime"] = ort.__version__
    return info


def reproduce(v_logits, v_ids, vy, items, v_match, release, members, section_of, n, train_count):
    """Recomputed validation against stored predictions and reported numbers."""
    checks = []

    def add(what, got, want, tol, note=""):
        checks.append({"what": what, "got": round(float(got), 6), "want": round(float(want), 6),
                       "pass": bool(abs(got - want) <= tol), "note": note})

    stored = {}
    for tag, name, which in ((TEACHER, "teacher", "best"), (STUDENT, "student", "last"), (CE, "ce100c", "last")):
        for s in SEEDS:
            z = np.load(FT / tag / "full" / f"seed-{s}" / f"predictions-{which}.npz")
            if not (np.array_equal(z["val_photo_id"], v_ids) and np.array_equal(z["val_label"], vy)):
                raise SystemExit(f"{tag}/{s}: stored validation order differs")
            stored[f"{name}/seed-{s}"] = z["scores"]
    if release is not None:
        z = np.load(release["_dir"] / "val-logits.npz")
        if not np.array_equal(z["val_photo_id"], v_ids):
            raise SystemExit("val-logits.npz: validation order differs")
        stored["shipped"] = z[f"scores_onnx_{release['shipped']}"]  # the export's own ONNX Runtime logits
    for nm, ref in stored.items():
        got = v_logits[nm]
        agree = float(np.mean(got.argmax(1) == ref.argmax(1)))
        b_ref = metrics(vy, ref, n, section_of)["balanced_accuracy"]
        add(f"{nm} balanced accuracy vs stored predictions", items[nm]["balanced_accuracy"], b_ref, TOL,
            f"(top-1 agreement {agree:.4f}, max |logit diff| {np.abs(got - ref).max():.2e}, "
            f"exact: {items[nm]['balanced_accuracy'] == b_ref}"
            + ("; int8 kernels differ by CPU, export ran on " + str(release.get("latency", {}).get("cpu"))
               if nm == "shipped" else "") + ")")
    for arm in ("full", "crop"):
        for s in SEEDS:
            want = json.loads((ANCHOR_DIR / arm / f"seed-{s}" / "classifier-metrics.json").read_text())["best_balanced_accuracy"]
            got = items[f"anchor-{arm}/seed-{s}"]["balanced_accuracy"]
            add(f"anchor-{arm}/seed-{s} balanced accuracy vs classifier-metrics.json", got, want, TOL,
                f"(exact to 1e-9: {abs(got - want) < 1e-9})")
    # Ensembles from the stored scores, through the same code.
    for group in ("teacher/ensemble", "student/ensemble"):
        p, lp = ensemble_prob([stored[k] for k in members[group]])
        add(f"{group} balanced accuracy vs stored predictions", items[group]["balanced_accuracy"],
            metrics(vy, lp, n, section_of)["balanced_accuracy"], TOL)
    for nm, want in KNOWN.items():
        add(f"{nm} balanced accuracy vs report", items[nm]["balanced_accuracy"], want,
            5e-4 if nm == "teacher/ensemble" else 5e-5 + 1e-6, "(report rounded)")
    for name, d in KNOWN_DIAG.items():
        it = items[f"{name}/seed-mean"]
        add(f"{name} seed-mean ECE before (diagnose)", it["ece_before"], d["ece"], TOL)
        add(f"{name} seed-mean ECE after (diagnose)", it["ece_after"], d["ece_t"], TOL)
        add(f"{name} seed-mean temperature (diagnose)", it["temperature"], d["temp"], TOL)
        add(f"{name} seed-mean accuracy (diagnose)", it["accuracy"], d["acc"], TOL)
        add(f"{name} seed-mean top-5 (diagnose)", it["top5_accuracy"], d["top5"], TOL)
        for j, b in enumerate(d["bins"]):
            add(f"{name} seed-mean bin {BINS[j][2]} (diagnose)", it["bins"][j], b, TOL)
    # Teacher calibration: the recorded T = 0.72 was fitted on the last-epoch ensemble;
    # reproduce that fit from predictions-last.npz as a check of the method.
    last = [np.load(FT / TEACHER / "full" / f"seed-{s}" / "predictions-last.npz")["scores"] for s in SEEDS]
    p, lp = ensemble_prob(last)
    ta = json.loads(TEACHER_T_SOURCE.read_text())
    grid2 = np.round(np.arange(0.30, 2.001, 0.01), 2)
    nll = [-np.mean(np.log(np.clip(tempered(lp, g)[np.arange(len(vy)), vy], 1e-12, None))) for g in grid2]
    add("teacher last-epoch ensemble temperature (analyze_species_teacher)", grid2[int(np.argmin(nll))],
        ta["calibration"]["temperature"], 1e-9)
    add("teacher last-epoch ensemble ECE before", ece(p, vy), ta["calibration"]["ece_before"], TOL)
    add("teacher last-epoch ensemble ECE after", ece(tempered(lp, TEACHER_T), vy), ta["calibration"]["ece_after"], TOL)
    add("teacher last-epoch ensemble balanced accuracy (0.834)", metrics(vy, lp, n, section_of)["balanced_accuracy"],
        ta["balanced_accuracy"], 1e-9)
    if release is not None:
        rv = release["validation"][f"onnx_{release['shipped']}"]
        it = items["shipped"]
        for k in KEYS:
            add(f"shipped {k} vs release.json", it[k], rv[k], TOL)
        add("shipped ECE before vs release.json", it["ece_before"], release["calibration"]["ece_before"], TOL)
        add("shipped ECE after vs release.json", it["ece_after"], release["calibration"]["ece_after"], TOL)
    add("full-s576 rebuilt in memory: validation JPEGs byte-identical to the cache", int((v_match == 1).sum()), len(vy), 0)
    return checks, all(c["pass"] for c in checks)


def render(s, split):
    f3 = lambda v: f"{v:.4f}"  # noqa: E731
    lines = [f"# Species classifier, {'held-out test' if split == 'test' else 'validation check'}: results table", "",
             "Generated by `scripts/score_species_test.py`; do not edit by hand.",
             "Protocol: `docs/species-classifier-plan.md` (test protocol 2026-09-26, amendment 2026-09-27, "
             "shipped-student protocol 2026-09-29).", ""]
    v = s["validation"]
    lines.append(f"Validation: {v['images']} images, {v['observers']} observers."
                 + (f" Test: {s['test']['images']} images, {s['test']['observers']} observers, "
                    f"{s['test']['labels']} species." if split == "test" else " The test split was not read.")
                 + f" Intervals: observer-grouped bootstrap, {s['bootstrap_reps']} resamples; seed means rescored per "
                   "seed on the same resample. Teacher: best epoch (amendment 2026-09-27); anchor: best epoch; "
                   "students: last epoch.")
    lines.append("")
    lines.append(f"Full-s576 cache rebuilt in memory: {v['cache_bytes_identical']}/{v['images']} validation JPEGs "
                 "byte-identical to the cached files.")
    if s["release"]:
        r = s["release"]
        lines.append(f"Shipped: `{r['shipped_file']}` ({r['shipped']}), seed {r['seed']}, epoch {r['epoch']}, "
                     f"temperature {r['temperature']:.2f}.")
    lines.append("")
    splits = ["validation"] + (["test"] if split == "test" else [])
    for sp in splits:
        lines += [f"## {sp.capitalize()}", "",
                  "| Item | Balanced acc [95% CI] | Accuracy | Top-5 | Section bal. acc | ECE before | T | ECE after |"
                  + (" Outside val CI |" if sp == "test" else ""),
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |" + (" --- |" if sp == "test" else "")]
        for nm, e in s["items"].items():
            it = e[sp]
            lo, hi = it["ci95"]["balanced_accuracy"]
            t = it["temperature"]
            row = (f"| `{nm}` | {f3(it['balanced_accuracy'])} [{lo:.3f}, {hi:.3f}] | {f3(it['accuracy'])} | "
                   f"{f3(it['top5_accuracy'])} | {f3(it['section_balanced_accuracy'])} | {f3(it['ece_before'])} | "
                   f"{t:.2f}{' (mean)' if it['kind'] == 'seed_mean' else ''} | {f3(it['ece_after'])} |")
            if sp == "test":
                row += f" {'**yes**' if e['test_outside_validation_ci95'] else 'no'} |"
            lines.append(row)
        lines += ["", f"Per-species accuracy averaged within training-count bins ({sp}):", "",
                  "| Item | " + " | ".join(f"{b} ({k})" for b, k in zip(s["bins"], s["bin_species"])) + " |",
                  "| --- |" + " ---: |" * len(s["bins"])]
        for nm, e in s["items"].items():
            it = e[sp]
            lines.append(f"| `{nm}` | " + " | ".join(
                f"{b:.3f} [{c[0]:.3f}, {c[1]:.3f}]" for b, c in zip(it["bins"], it["bins_ci95"])) + " |")
        d = s["student_minus_ce100c"][sp]
        lines += ["", f"Student (KDw-100c+T) - CE-100c, seed mean paired by seed ({sp}): "
                      f"{d['delta']:+.4f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], P(delta <= 0) {d['p_le_0']:.3f}, "
                      f"wins {d['wins']}/5; per seed " + ", ".join(f"{x:+.4f}" for x in d["per_seed"]) + ".",
                  "", "| Bin | Delta [95% CI] |", "| --- | --- |"]
        for b, (pt, lo, hi) in zip(s["bins"], d["bins"]):
            lines.append(f"| {b} | {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        lines.append("")
    lines += ["## Validation reproduction checks", "",
              f"All passed: {'yes' if s['checks_passed'] else '**NO**'}.", "",
              "| Check | Recomputed | Reference | Pass | Note |", "| --- | ---: | ---: | --- | --- |"]
    for c in s["checks"]:
        lines.append(f"| {c['what']} | {c['got']} | {c['want']} | {'yes' if c['pass'] else '**no**'} | {c['note']} |")
    pf = s.get("onnx_platform") or {}
    if s["release"]:
        lines += ["", f"Shipped ONNX run on {pf.get('cpu')} (AVX512-VNNI: {pf.get('avx512_vnni')}), onnxruntime "
                      f"{pf.get('onnxruntime')}; the export and its temperature ran on {pf.get('release_export_cpu')}. "
                      "Dynamic-int8 logits differ between CPU families; fp32 ones do not. Temperature used: "
                      f"{s['release']['temperature']:.2f} (release.json); refit on this CPU's validation logits: "
                      f"{pf.get('shipped_temperature_refit_here'):.2f} (not used)."]
    lines += ["", f"Teacher ensemble temperature: recorded {s['temperatures']['teacher/ensemble']:.2f} "
                  f"(fitted on the last-epoch ensemble); a refit on the best-epoch validation ensemble gives "
                  f"{s['teacher_ensemble_temperature_refit_on_best_epoch_validation']:.2f} (not used)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
