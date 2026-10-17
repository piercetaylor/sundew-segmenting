#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pierce Taylor
"""Species data and photos for the site's species pages (cluster only; CI never needs the corpus).

Reads the species corpus manifest, the section table, the release labels, and the shipped
model's validation scores, and writes:

  data/site/species.json         one entry per species: names, taxonomy, training count,
                                 how the shipped model does on its validation photos,
                                 which species it mistakes it for, and the chosen photos
  site/img/species/<slug>/*.webp the chosen photos: one within 480 px, two within 320 px,
                                 EXIF-rotated, no metadata kept
  site/img/ATTRIBUTION.md        a credit line for every photo on the site

Photo choice per species (3 photos, all different observers):
  1. photos the shipped model names correctly and confidently on validation (a clear view
     of the plant), then the rest;
  2. CC0, then CC BY, then CC BY-NC (only these three are in the corpus);
  3. validation, then test, then train; larger photos first.
The 330 browser-parity fixture photos are never picked. data/site/photo-picks.json can pin
or reject photo ids per species:
  {"Drosera alba": {"pin": [123], "reject": [456]}}

Usage:
  PYTHONPATH=src <sundew-seg python> scripts/build_site_data.py [--contact-sheet out.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from sundew_segmentation.species_decision import E_ANSWER, P_ANSWER, load_sections, neg_energy, softmax_t

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT.parent / "sundew-species-corpus"
RECORDS = CORPUS / "split-110-test/species-records.jsonl"
SECTIONS = ROOT / "data/species-110-sections.json"
LABELS = ROOT / "release/species-v1.0.0/labels.json"
LOGITS = ROOT / "models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/val-logits.npz"
FIXTURE = ROOT / "handoff/browser-parity/fixture/expected.json"
PICKS = ROOT / "data/site/photo-picks.json"
OUT_JSON = ROOT / "data/site/species.json"
IMG_DIR = ROOT / "site/img/species"
ATTRIBUTION = ROOT / "site/img/ATTRIBUTION.md"

N_PHOTOS = 3
BOXES = (480, 320, 320)  # longest side, px
WEBP_QUALITY = 72
RARE_TRAIN = 40  # the "under 40" training-count band of docs/reports/species-test
LICENCE_RANK = {"cc0": 0, "cc-by": 1, "cc-by-nc": 2}
LICENCE_NAME = {"cc0": "CC0 1.0", "cc-by": "CC BY 4.0", "cc-by-nc": "CC BY-NC 4.0"}
SPLIT_RANK = {"validation": 0, "test": 1, "train": 2}


def slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def load_records(labels: list[str]) -> dict[str, list[dict]]:
    by = defaultdict(list)
    with RECORDS.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["label"] in labels:
                by[r["label"]].append(r)
    return by


def validation_stats(labels: list[str]):
    """Per species: validation n, top-1 right, answered, answered right, top wrong guesses; and the
    photo ids the model names correctly under the site's rule."""
    d = np.load(LOGITS, allow_pickle=False)
    logits = d["scores_onnx_int8"]  # the shipped int8 model, as the browser runs it
    y = d["val_label"].astype(int)
    probs = softmax_t(logits)
    top1 = probs.argmax(1)
    answered = (probs.max(1) >= P_ANSWER) & (neg_energy(logits) >= E_ANSWER)
    right = top1 == y
    stats, confident_right = {}, set(d["val_photo_id"][answered & right].tolist())
    for c, name in enumerate(labels):
        m = y == c
        wrong = Counter(top1[m & ~right].tolist())
        stats[name] = {
            "n": int(m.sum()),
            "top1_right": int((m & right).sum()),
            "answered": int((m & answered).sum()),
            "answered_right": int((m & answered & right).sum()),
            "mistaken_for": [{"name": labels[k], "n": int(v)} for k, v in wrong.most_common(3)],
        }
    return stats, confident_right


def choose(recs: list[dict], confident_right: set[int], banned: set[int], pick: dict) -> list[dict]:
    reject = set(pick.get("reject", [])) | banned
    pins = [r for pid in pick.get("pin", []) for r in recs if r["photo_id"] == pid]
    pool = [r for r in recs if r["photo_id"] not in reject and r["license_code"] in LICENCE_RANK]
    pool.sort(key=lambda r: (r["photo_id"] not in confident_right, LICENCE_RANK[r["license_code"]],
                             SPLIT_RANK.get(r["split"], 9), -min(r["source_width"], r["source_height"]), r["photo_id"]))
    out, seen = [], set()
    for r in pins + pool:
        if r["observer_login"] in seen or any(o["photo_id"] == r["photo_id"] for o in out):
            continue
        out.append(r)
        seen.add(r["observer_login"])
        if len(out) == N_PHOTOS:
            break
    return out


def write_photo(r: dict, dest: Path, box: int) -> tuple[int, int]:
    with Image.open(r["local_path"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((box, box), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "WEBP", quality=WEBP_QUALITY, method=6)  # no exif/icc passed: metadata dropped
        return im.width, im.height


def contact_sheet(entries: list[dict], out: Path) -> None:
    cell, label_h, cols = 220, 30, 3 * 4
    rows = -(-len(entries) * 3 // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    k = 0
    for e in entries:
        for p in e["photos"]:
            x, y = (k % cols) * cell, (k // cols) * (cell + label_h)
            with Image.open(ROOT / "site" / p["src"]) as im:
                im.thumbnail((cell - 6, cell - 6))
                sheet.paste(im, (x + 3, y + 3))
            draw.text((x + 4, y + cell), f"{e['name'].split()[1]} {p['photo_id']}", fill="black")
            draw.text((x + 4, y + cell + 13), f"{p['licence_code']} {p['split']}", fill="gray")
            k += 1
    sheet.save(out, quality=85)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact-sheet", type=Path, help="also write a JPEG of every chosen photo, for review")
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text())
    taxa = json.loads(SECTIONS.read_text())["species"]
    sections = load_sections(SECTIONS, labels)
    records = load_records(labels)
    stats, confident_right = validation_stats(labels)
    banned = {p["photo_id"] for p in json.loads(FIXTURE.read_text())["photos"]} if FIXTURE.exists() else set()
    if not banned:
        print("warning: browser-parity fixture list not found; its photos are not excluded", file=sys.stderr)
    picks = json.loads(PICKS.read_text()) if PICKS.exists() else {}

    entries = []
    for name in labels:
        recs = records[name]
        train = sum(r["split"] == "train" for r in recs)
        common = Counter(r["common_name"] for r in recs if r.get("common_name")).most_common(1)
        s = slug(name)
        photos = []
        for r, w in zip(choose(recs, confident_right, banned, picks.get(name, {})), BOXES):
            rel = f"img/species/{s}/{r['photo_id']}-{w}.webp"
            pw, ph = write_photo(r, ROOT / "site" / rel, w)
            photos.append({
                "photo_id": r["photo_id"], "src": rel, "width": pw, "height": ph,
                "creator": r["creator"], "attribution": r["attribution"],
                "licence_code": r["license_code"], "licence": LICENCE_NAME[r["license_code"]],
                "licence_url": r["license_url"], "source_page": r["source_page"],
                "observed_on": r.get("observed_on"), "split": r["split"],
            })
        entries.append({
            "name": name, "slug": s, "common_name": common[0][0] if common else None,
            "taxon_id": taxa[name]["taxon_id"], "subgenus": taxa[name]["subgenus"], "section": sections[name],
            "train_photos": train, "rare": train < RARE_TRAIN,
            "validation": stats[name], "photos": photos,
        })
        print(f"{name}: {len(photos)} photos, train {train}, val {stats[name]['top1_right']}/{stats[name]['n']}")

    # Remove photos of an earlier build that are no longer chosen.
    keep = {ROOT / "site" / p["src"] for e in entries for p in e["photos"]}
    for f in IMG_DIR.rglob("*.webp"):
        if f not in keep:
            f.unlink()

    doc = {
        "note": "Generated by scripts/build_site_data.py; do not edit. Validation numbers are for the shipped "
                "int8 model (sundew-species v1.0.0) under the site's rule (p >= 0.65 and energy >= 5.955).",
        "model": "sundew-species v1.0.0",
        "species": entries,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")

    lines = ["# Photo credits", "",
             "Photos on this site are from iNaturalist observations, used under the licence shown, resized and "
             "re-encoded as WebP (metadata removed). Generated by `scripts/build_site_data.py`.", ""]
    for e in entries:
        lines.append(f"## {e['name']}")
        lines.append("")
        for p in e["photos"]:
            lines.append(f"- `{p['src']}`: {p['attribution']}, [{p['licence']}]({p['licence_url']}), "
                         f"<{p['source_page']}>")
        lines.append("")
    ATTRIBUTION.write_text("\n".join(lines))

    size = sum(f.stat().st_size for f in IMG_DIR.rglob("*.webp"))
    lic = Counter(p["licence_code"] for e in entries for p in e["photos"])
    print(f"wrote {OUT_JSON}, {ATTRIBUTION}; {sum(len(e['photos']) for e in entries)} photos, "
          f"{size / 1e6:.1f} MB, licences {dict(lic)}")
    if args.contact_sheet:
        contact_sheet(entries, args.contact_sheet)
        print(f"wrote {args.contact_sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
