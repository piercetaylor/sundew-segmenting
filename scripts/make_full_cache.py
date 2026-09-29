"""Cache full frames at a fixed short side, so training is not bound by JPEG decoding.

Every fine-tune so far spent ~105-120 s per epoch decoding the original
1,500-2,048 px JPEGs, whatever the model. This writes each frame once, resized
so its short side is --short-side px (LANCZOS, as the crops and the full768
control were), to <out>/inat_<photo_id>.jpg. Every original has a short side
of at least 600 px, so every image goes through the same single downscale.
576 px leaves room for 384 px training: RandomResizedCrop at scale 0.7 needs a
short side of about 459.

Kept identical to the training loader: Image.open + convert("RGB"), no EXIF
rotation (finetune_species_backbone.py applies none). JPEG quality 95 with
4:4:4 chroma, so pigment is not smeared by chroma subsampling.

Train and validation only by default: the held-out test split is not decoded
until it is scored. docs/species-classifier-plan.md, amendment of 2026-09-27.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from multiprocessing import Pool

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--short-side", type=int, default=576)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def one(job):
    src, dst, short, quality = job
    if pathlib.Path(dst).exists():
        return "skipped"
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if min(w, h) < short:
            raise SystemExit(f"{src}: short side {min(w, h)} below {short}; would need upsampling")
        scale = short / min(w, h)
        im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        tmp = pathlib.Path(dst).with_suffix(".tmp")
        im.save(tmp, format="JPEG", quality=quality, subsampling=0)
        tmp.replace(dst)  # atomic, so a killed job leaves no half-written file
    return "written"


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] in args.splits]
    jobs = [(r["image"], str(args.out / f"inat_{r['photo_id']}.jpg"), args.short_side, args.quality) for r in rows]
    counts = {"written": 0, "skipped": 0}
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs, chunksize=16), 1):
            counts[res] += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({
        "records": str(args.records), "splits": args.splits, "images": len(jobs),
        "short_side": args.short_side, "resample": "LANCZOS", "jpeg_quality": args.quality,
        "chroma_subsampling": "4:4:4", "exif_transpose": False, **counts,
    }, indent=2) + "\n")
    print(f"done: {counts} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
