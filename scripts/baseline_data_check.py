"""Validate the image/mask boundary before a baseline training run."""
from __future__ import annotations
import argparse
from pathlib import Path
from sundew_segmentation.baseline import paired_samples

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/curated/images"))
    ap.add_argument("--masks", type=Path, default=Path("data/curated/masks"))
    ap.add_argument("--split", action="append", default=["train", "validation"])
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    total = 0
    for split in dict.fromkeys(args.split):
        pairs = paired_samples(args.images, args.masks, split, require_masks=not args.allow_missing)
        print(f"{split}: {len(pairs)} image/mask pairs")
        total += len(pairs)
    if not total and not args.allow_missing:
        raise SystemExit("no reviewed masks found; refusing to train")

if __name__ == "__main__":
    main()
