"""Build the padded segmentation crops and render indexed sheets for eye review.

The crop is produced exactly as a crop-then-classify pipeline would consume it:
predicted foreground at the operating threshold, bounding box of all foreground,
padded 10% on each side, clipped to the frame. This is the same definition
scrape-pipeline-readiness.md used for crop coverage, so the reviewed failure
rate is comparable to the 99.6% measured there.

Writes the crops themselves as well as the sheets, so the same crops can feed
the classifier comparison rather than being regenerated with a second
implementation that might drift.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sundew_segmentation.baseline import (  # noqa: E402
    SEGFORMER_B0, UNET_RESNET34, resolve_model,
)

MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--pad", type=float, default=0.10)
    p.add_argument("--per-sheet", type=int, default=20)
    p.add_argument("--columns", type=int, default=5)
    p.add_argument("--tile", type=int, default=360)
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--no-sheets", action="store_true",
                   help="Write crops and manifest only. Used when generating classifier inputs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    base = UNET_RESNET34 if ck["config"]["name"] == "unet-resnet34" else SEGFORMER_B0
    cfg = replace(base, **{k: ck["config"][k] for k in base.to_dict() if k in ck["config"]})
    model = resolve_model(cfg, pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output / "crops"
    if args.save_crops:
        crop_dir.mkdir(exist_ok=True)

    paths = sorted(args.images.glob("*.jpg"))
    rows, tiles = [], []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as source:
            full = source.convert("RGB")
        w, h = full.size
        small = full.resize((cfg.image_size,) * 2, Image.Resampling.BILINEAR)
        arr = np.asarray(small, np.float32).transpose(2, 0, 1) / 255.0
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(((arr - MEAN) / STD).copy()).unsqueeze(0)))[0, 0].numpy()
        pred = prob >= args.threshold

        empty = not pred.any()
        if empty:
            box_px = (0, 0, w, h)
        else:
            r = np.where(pred.any(1))[0]
            c = np.where(pred.any(0))[0]
            y0, y1, x0, x1 = int(r[0]), int(r[-1]), int(c[0]), int(c[-1])
            ph = int(args.pad * (y1 - y0 + 1))
            pw = int(args.pad * (x1 - x0 + 1))
            y0, y1 = max(0, y0 - ph), min(cfg.image_size - 1, y1 + ph)
            x0, x1 = max(0, x0 - pw), min(cfg.image_size - 1, x1 + pw)
            sx, sy = w / cfg.image_size, h / cfg.image_size
            box_px = (int(x0 * sx), int(y0 * sy), int(min(w, (x1 + 1) * sx)), int(min(h, (y1 + 1) * sy)))

        crop = full.crop(box_px)
        if args.save_crops and crop.width > 0 and crop.height > 0:
            out = crop.copy()
            out.thumbnail((768, 768), Image.Resampling.LANCZOS)
            out.save(crop_dir / f"{path.stem}.jpg", quality=90)

        rows.append({
            "index": index, "name": path.name, "empty_prediction": bool(empty),
            "box_px": list(box_px), "frame_px": [w, h],
            "box_frac": round((box_px[2] - box_px[0]) * (box_px[3] - box_px[1]) / (w * h), 4),
            "fg_frac": round(float(pred.mean()), 4),
            "mean_conf_fg": round(float(prob[pred].mean()) if pred.any() else 0.0, 4),
        })

        # Tile: the crop as the classifier sees it, with a large index for review.
        tile = None
        if not args.no_sheets:
            tile = Image.new("RGB", (args.tile, args.tile + 34), "white")
            shown = crop.copy()
            shown.thumbnail((args.tile - 8, args.tile - 8), Image.Resampling.LANCZOS)
            tile.paste(shown, ((args.tile - shown.width) // 2, (args.tile - shown.height) // 2))
            draw = ImageDraw.Draw(tile)
            label = f"{index}" + ("  EMPTY PREDICTION" if empty else "")
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
            except OSError:
                font = ImageFont.load_default()
            draw.rectangle([0, args.tile, args.tile, args.tile + 34], fill="#222222")
            draw.text((8, args.tile + 4), label, fill="#ffdd33" if empty else "#ffffff", font=font)
        if not args.no_sheets:
            tiles.append(tile)
        if index % 50 == 0:
            print(f"  {index}/{len(paths)}", flush=True)

    cols = args.columns
    per = args.per_sheet
    sheets = 0
    for start in range(0, len(tiles), per):
        chunk = tiles[start:start + per]
        r = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * args.tile, r * (args.tile + 34)), "white")
        for i, t in enumerate(chunk):
            sheet.paste(t, ((i % cols) * args.tile, (i // cols) * (args.tile + 34)))
        sheets += 1
        sheet.save(args.output / f"crop-review-{sheets:02d}.jpg", quality=90)

    (args.output / "crop-review-manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    empties = sum(1 for r in rows if r["empty_prediction"])
    print(f"\n{len(rows)} crops, {sheets} sheets, indices 1-{len(rows)}")
    print(f"empty predictions (automatic failures): {empties}")
    print(f"median predicted box covers {np.median([r['box_frac'] for r in rows])*100:.1f}% of frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
