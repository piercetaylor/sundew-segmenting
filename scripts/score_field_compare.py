"""Score every field-compare checkpoint against field-eval, paired by image.

Preprocessing matches scripts/evaluate_checkpoint.py exactly so the numbers are
comparable to the 0.4994 baseline in reports/field-probe-evaluation.md.

Reports IoU and relative area error. Area is the project's deliverable and it
degraded proportionally worse than IoU in step 1, so a comparison on IoU alone
would miss the thing the model is for.
"""
import json, pathlib, sys, itertools
from dataclasses import replace
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, 'src')
from sundew_segmentation.baseline import SEGFORMER_B0, UNET_RESNET34, binary_metrics, resolve_model

BASE = pathlib.Path('/cluster/VAST/mendozacozatld-lab/PierceTaylor')
IMG = BASE / 'field-probe-eval/images/field-eval'
MSK = BASE / 'field-probe-v1/masks/field-eval'
SEEDS = (17, 101, 202, 303, 404)
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
mean = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
std = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    base = UNET_RESNET34 if ck['config']['name'] == 'unet-resnet34' else SEGFORMER_B0
    cfg = replace(base, **{k: ck['config'][k] for k in base.to_dict() if k in ck['config']})
    m = resolve_model(cfg, pretrained=False)
    m.load_state_dict(ck['model_state'])
    m.eval()
    return m, cfg


def score(model, cfg):
    """Per-image metrics at every threshold, keyed by image name."""
    out = {}
    for imp in sorted(IMG.glob('*.jpg')):
        with Image.open(imp) as s:
            im = s.convert('RGB').resize((cfg.image_size,) * 2, Image.Resampling.BILINEAR)
        with Image.open(MSK / (imp.stem + '.png')) as s:
            gt = np.asarray(s.convert('L').resize((cfg.image_size,) * 2, Image.Resampling.NEAREST)) > 0
        a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(((a - mean) / std).copy()).unsqueeze(0)))[0, 0].numpy()
        rec = {}
        for t in THRESHOLDS:
            pr = prob >= t
            m = binary_metrics(pr, gt)
            gt_a, pr_a = float(gt.mean()), float(pr.mean())
            m['area_rel'] = (pr_a - gt_a) / gt_a if gt_a > 0 else float('nan')
            rec[t] = m
        out[imp.name] = rec
    return out


runs = {}
for arm in ('frozen', 'field'):
    for seed in SEEDS:
        p = pathlib.Path(f'models/field-compare/{arm}/seed-{seed}/segformer-b0-best.pt')
        if not p.exists():
            print(f"!! missing {p}", flush=True); continue
        model, cfg = load(p)
        runs[(arm, seed)] = score(model, cfg)
        v = json.load(open(p.parent / 'segformer-b0-metrics.json'))['best_iou']
        m5 = np.mean([r[0.5]['iou'] for r in runs[(arm, seed)].values()])
        print(f"{arm:<7} seed {seed:<4} curated-val {v:.4f}   field-eval@0.5 {m5:.4f}", flush=True)

ref = pathlib.Path('models/recipe/combined/seed-17/segformer-b0-best.pt')
if ref.exists():
    model, cfg = load(ref)
    runs[('reference', 17)] = score(model, cfg)
    print(f"{'ref':<7} seed 17   (original combined, WITH balancing)  field-eval@0.5 "
          f"{np.mean([r[0.5]['iou'] for r in runs[('reference',17)].values()]):.4f}", flush=True)

names = sorted(next(iter(runs.values())).keys())
json.dump({f"{a}|{s}": {n: {str(t): v for t, v in r.items()} for n, r in d.items()}
           for (a, s), d in runs.items()},
          open('data/reports/field-probe-evaluation/field-compare-scores.json', 'w'), indent=2)


def arm_mean(arm, t, key='iou'):
    return {s: np.mean([runs[(arm, s)][n][t][key] for n in names]) for s in SEEDS if (arm, s) in runs}


print("\n" + "=" * 74)
print("ARM MEANS on field-eval (n=17), threshold 0.5")
print("=" * 74)
for arm in ('frozen', 'field'):
    got = arm_mean(arm, 0.5)
    if not got: continue
    vals = list(got.values())
    print(f"  {arm:<7} " + "  ".join(f"seed{s}={v:.4f}" for s, v in got.items())
          + f"   mean {np.mean(vals):.4f}  SD {np.std(vals, ddof=1) if len(vals)>1 else 0:.4f}")

if all((a, s) in runs for a in ('frozen', 'field') for s in SEEDS):
    print("\n" + "=" * 74)
    print("PAIRED COMPARISON  (field - frozen), threshold 0.5")
    print("=" * 74)
    # Pair by seed: same seed, same images, differing only in the 30 extra masks.
    per_seed = [arm_mean('field', 0.5)[s] - arm_mean('frozen', 0.5)[s] for s in SEEDS]
    for s, d in zip(SEEDS, per_seed):
        print(f"  seed {s:<4} delta {d:+.4f}")
    n = len(per_seed); m = float(np.mean(per_seed)); sd = float(np.std(per_seed, ddof=1))
    se = sd / np.sqrt(n)
    crit = 2.776  # t .05/2, 4 df
    print(f"\n  paired mean delta {m:+.4f}   SD {sd:.4f}   SE {se:.4f}")
    print(f"  95% CI [{m-crit*se:+.4f}, {m+crit*se:+.4f}]   t = {m/se if se else float('nan'):.2f} on {n-1} df (crit {crit})")
    print(f"  seeds where field wins: {sum(1 for d in per_seed if d>0)}/{n}")

    # Per-image paired difference, averaged over seeds: which images moved?
    print("\n  per-image change (mean over all seeds), worst-first by baseline:")
    base_iou = {n_: np.mean([runs[('frozen', s)][n_][0.5]['iou'] for s in SEEDS]) for n_ in names}
    for n_ in sorted(names, key=lambda x: base_iou[x]):
        f = np.mean([runs[('field', s)][n_][0.5]['iou'] for s in SEEDS])
        print(f"    {n_:<22} {base_iou[n_]:.3f} -> {f:.3f}   {f-base_iou[n_]:+.3f}")

print("\n" + "=" * 74)
print("AREA ERROR vs THRESHOLD  (median |relative area error|, and signed bias)")
print("=" * 74)
print(f"  {'thr':<6}" + "".join(f"{a+' med':>12}{a+' bias':>12}" for a in ('frozen', 'field')))
for t in THRESHOLDS:
    cells = ""
    for arm in ('frozen', 'field'):
        if (arm, SEEDS[0]) not in runs: continue
        vals = [runs[(arm, s)][n_][t]['area_rel'] for s in SEEDS if (arm, s) in runs for n_ in names]
        vals = [v for v in vals if np.isfinite(v)]
        cells += f"{np.median(np.abs(vals))*100:>11.1f}%{np.mean(vals)*100:>11.1f}%"
    print(f"  {t:<6}{cells}")

print("\n  IoU vs threshold:")
print(f"  {'thr':<6}{'frozen':>10}{'field':>10}")
for t in THRESHOLDS:
    row = f"  {t:<6}"
    for arm in ('frozen', 'field'):
        got = arm_mean(arm, t)
        row += f"{np.mean(list(got.values())):>10.4f}" if got else f"{'-':>10}"
    print(row)
