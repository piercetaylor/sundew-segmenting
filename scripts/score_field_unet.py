"""Decide whether the U-Net/ResNet-34 (field recipe) can draw the plant outline on the site.

Applies docs/reports/segmentation-site-prereg.md: scores the five U-Net seeds
and the five SegFormer-B0 field seeds on the 17 field-eval images with the
preprocessing of scripts/score_field_compare.py, then

  1. quality bar: U-Net five-seed mean IoU at 0.5 >= 0.593 (SegFormer 0.6129 - 0.02);
  2. seed: highest curated-validation IoU (best_iou), ties to the lower seed;
  3. threshold: lowest median |relative area error| on field-eval over the seeds.

Writes docs/reports/segmentation-site-scores.json and prints the tables for
docs/reports/segmentation-site.md.

Run from the repo root:
  PYTHONPATH=src python scripts/score_field_unet.py [--models <models dir>]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sundew_segmentation.baseline import SEGFORMER_B0, UNET_RESNET34, binary_metrics, resolve_model

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor")
IMG = BASE / "field-probe-eval/images/field-eval"
MSK = BASE / "field-probe-v1/masks/field-eval"
SEEDS = (17, 101, 202, 303, 404)
THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
SEGFORMER_FIELD_MEAN = 0.6129  # docs/reports/field-compare.md
BAR = 0.593
MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]


def load(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    base = UNET_RESNET34 if ck["config"]["name"] == "unet-resnet34" else SEGFORMER_B0
    cfg = replace(base, **{k: ck["config"][k] for k in base.to_dict() if k in ck["config"]})
    model = resolve_model(cfg, pretrained=False)
    model.load_state_dict(ck["model_state"])
    return model.eval(), cfg


def score(model, cfg) -> dict:
    """Per image and threshold: IoU and relative area error (as score_field_compare.py)."""
    out = {}
    for imp in sorted(IMG.glob("*.jpg")):
        with Image.open(imp) as s:
            im = s.convert("RGB").resize((cfg.image_size,) * 2, Image.Resampling.BILINEAR)
        with Image.open(MSK / (imp.stem + ".png")) as s:
            gt = np.asarray(s.convert("L").resize((cfg.image_size,) * 2, Image.Resampling.NEAREST)) > 0
        a = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(((a - MEAN) / STD).copy()).unsqueeze(0)))[0, 0].numpy()
        rec = {}
        for t in THRESHOLDS:
            pr = prob >= t
            m = binary_metrics(pr, gt)
            gt_a = float(gt.mean())
            rec[str(t)] = {"iou": float(m["iou"]), "area_rel": (float(pr.mean()) - gt_a) / gt_a if gt_a > 0 else float("nan")}
        out[imp.name] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", type=Path, default=BASE / "sundew-segmenting/models/field-compare")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/reports/segmentation-site-scores.json")
    args = ap.parse_args()
    torch.set_num_threads(4)

    arms = {"unet": ("field-unet", "unet-resnet34"), "segformer": ("field", "segformer-b0")}
    runs, val_iou = {}, {}
    for arm, (folder, name) in arms.items():
        for seed in SEEDS:
            d = args.models / folder / f"seed-{seed}"
            ck = d / f"{name}-best.pt"
            if not ck.exists():
                raise SystemExit(f"missing {ck}")
            runs[(arm, seed)] = score(*load(ck))
            val_iou[(arm, seed)] = json.loads((d / f"{name}-metrics.json").read_text())["best_iou"]
            m5 = np.mean([r["0.5"]["iou"] for r in runs[(arm, seed)].values()])
            print(f"{arm:<9} seed {seed:<4} curated-val {val_iou[(arm, seed)]:.4f}  field-eval@0.5 {m5:.4f}", flush=True)

    names = sorted(next(iter(runs.values())))
    field = {k: float(np.mean([v[n]["0.5"]["iou"] for n in names])) for k, v in runs.items()}
    unet_mean = float(np.mean([field[("unet", s)] for s in SEEDS]))
    seg_mean = float(np.mean([field[("segformer", s)] for s in SEEDS]))
    paired = {s: field[("unet", s)] - field[("segformer", s)] for s in SEEDS}
    ship_seed = max(SEEDS, key=lambda s: (val_iou[("unet", s)], -s))
    area = {}
    for t in THRESHOLDS:
        vals = [runs[("unet", s)][n][str(t)]["area_rel"] for s in SEEDS for n in names]
        vals = [v for v in vals if np.isfinite(v)]
        area[str(t)] = {"median_abs": float(np.median(np.abs(vals))), "bias": float(np.mean(vals)),
                        "iou": float(np.mean([runs[("unet", s)][n][str(t)]["iou"] for s in SEEDS for n in names]))}
    threshold = min(THRESHOLDS, key=lambda t: (area[str(t)]["median_abs"], t))
    passed = unet_mean >= BAR
    ship_iou = float(np.mean([runs[("unet", ship_seed)][n][str(threshold)]["iou"] for n in names]))
    result = {
        "pre_registration": "docs/reports/segmentation-site-prereg.md", "bar": BAR, "passed": passed,
        "unet_field_mean_at_0.5": unet_mean, "segformer_field_mean_at_0.5": seg_mean,
        "segformer_reference": SEGFORMER_FIELD_MEAN,
        "per_seed": {str(s): {"unet_val": val_iou[("unet", s)], "unet_field": field[("unet", s)],
                              "segformer_val": val_iou[("segformer", s)], "segformer_field": field[("segformer", s)],
                              "paired_diff": paired[s]} for s in SEEDS},
        "paired_mean_diff": float(np.mean(list(paired.values()))),
        "ship_seed": ship_seed, "threshold": threshold, "ship_seed_field_iou_at_threshold": ship_iou,
        "unet_area_by_threshold": area,
        "per_image": {f"{a}|{s}": v for (a, s), v in runs.items()},
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\nU-Net mean field-eval IoU@0.5 {unet_mean:.4f} (SegFormer {seg_mean:.4f}; bar {BAR}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    print("| Seed | U-Net val | U-Net field@0.5 | SegFormer field@0.5 | U-Net - SegFormer |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for s in SEEDS:
        print(f"| {s} | {val_iou[('unet', s)]:.4f} | {field[('unet', s)]:.4f} | {field[('segformer', s)]:.4f} | {paired[s]:+.4f} |")
    print(f"| mean | | **{unet_mean:.4f}** | {seg_mean:.4f} | {np.mean(list(paired.values())):+.4f} |")
    print("\n| Threshold | U-Net IoU | Median abs. area error | Mean signed area error |")
    print("| ---: | ---: | ---: | ---: |")
    for t in THRESHOLDS:
        a = area[str(t)]
        print(f"| {t}{' **(chosen)**' if t == threshold else ''} | {a['iou']:.4f} | {100 * a['median_abs']:.1f}% | {100 * a['bias']:+.1f}% |")
    print(f"\nship seed {ship_seed} (best curated-val IoU), threshold {threshold}: field-eval IoU {ship_iou:.4f}")


if __name__ == "__main__":
    main()
