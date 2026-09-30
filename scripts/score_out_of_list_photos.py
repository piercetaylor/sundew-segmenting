"""Spot check: what the species model says about photos outside its 110 species.

Downloads the photos listed in a manifest (Wikimedia Commons, CC0 / public
domain / CC BY), runs the shipped int8 model with the browser's preprocessing,
applies the site's decision rule, and writes a table. The photos are cached
locally and never committed; the manifest records their source and licence.

This is a sanity check of the wording on the site, not an evaluation: about 20
photos is far too few to tune the threshold on, and the threshold is not
changed from it. See docs/reports/species-out-of-list-check.md.

Preprocessing (as site/classify.js, method "pil"): EXIF rotation applied, RGB,
short side to 255 px (Pillow bicubic), centre crop 224, pixel / 255.

Run from the repo root:
  PYTHONPATH=src python scripts/score_out_of_list_photos.py --cache /tmp/oos-photos
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from sundew_segmentation.species_decision import P_ANSWER, decide, load_sections, softmax_t

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/species-v1.0.0"
UA = "sundew-segmenting out-of-list check (https://github.com/piercetaylor/sundew-segmenting)"


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    size = (255, int(255 * h / w)) if w <= h else (int(255 * w / h), 255)
    im = im.resize(size, Image.BICUBIC)
    left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
    im = im.crop((left, top, left + 224, top + 224))
    x = np.asarray(im, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())
    time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/out-of-list-check-photos.json")
    ap.add_argument("--cache", type=Path, required=True, help="local folder for the downloaded photos")
    ap.add_argument("--model", type=Path, default=RELEASE / "model-int8.onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/reports/species-out-of-list-check.json")
    args = ap.parse_args()

    import onnxruntime as ort

    labels = json.loads((RELEASE / "labels.json").read_text())
    release = json.loads((RELEASE / "release.json").read_text())
    sections = load_sections(ROOT / "data/species-110-sections.json", labels)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(args.model), so, providers=["CPUExecutionProvider"])

    photos = json.loads(args.manifest.read_text())["photos"]
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ph in enumerate(photos):
        dest = args.cache / f"{i:02d}.jpg"
        fetch(ph["url"], dest)
        logits = sess.run(None, {"pixels": preprocess(dest)})[0][0]
        d = decide(softmax_t(logits, release["temperature"]), labels, sections)
        rows.append({**{k: ph[k] for k in ("category", "title", "licence", "author", "page")},
                     "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                     "state": d["state"], "top1": d["top"][0]["label"], "p": round(d["p"], 4),
                     "section": d["section"], "section_p": None if d["section_p"] is None else round(d["section_p"], 4),
                     "top5": [[t["label"], round(t["p"], 4)] for t in d["top"]]})
        print(f"{ph['category']:<26} {d['state']:<9} p={d['p']:.3f} {d['top'][0]['label']:<28} {ph['title'][:50]}")
    confident = sum(r["state"] == "answer" for r in rows)
    summary = {"photos": len(rows), "confident_species_name": confident, "p_answer": P_ANSWER,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(), "rows": rows}
    args.out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(f"{confident} of {len(rows)} out-of-list photos got a confident species name (p >= {P_ANSWER})")


if __name__ == "__main__":
    main()
