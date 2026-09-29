"""Run a dependency-light baseline smoke test on curated images.

This checks image loading and writes deterministic color-heuristic masks. It
does not claim model quality and intentionally works without annotations/GPU.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from sundew_segmentation.baseline import UNET_RESNET34, SEGFORMER_B0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/curated/images"))
    ap.add_argument("--output", type=Path, default=Path("data/reports/baseline-smoke.json"))
    ap.add_argument("--limit", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(args.images.rglob("*.jpg"))[:args.limit]
    rows = []
    for path in paths:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        # Stable sanity mask only: sundew foliage tends to be greener/redder than background.
        mask = ((rgb[..., 1] > rgb[..., 0] * 0.85) & (rgb[..., 1] > rgb[..., 2] * 0.8)).mean()
        rows.append({"image": path.as_posix(), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "heuristic_foreground_fraction": round(float(mask), 6)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"configs": [UNET_RESNET34.to_dict(), SEGFORMER_B0.to_dict()], "images_checked": len(rows), "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"checked {len(rows)} images; wrote {args.output}")

if __name__ == "__main__":
    main()
