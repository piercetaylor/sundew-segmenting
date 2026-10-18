#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pierce Taylor
"""The identifier's "try an example" photos (cluster only).

Picks CC0 / CC BY validation photos, writes each as a 768 px JPEG (plus a 200 px WebP
thumbnail) under site/img/examples/, and records in site/img/examples/examples.json what the
site should show for it: the Python reference (the same int8 ONNX model and the training
preprocessing, run on the exact committed JPEG), with the site's decision rule.

The set is meant to be honest about the tool's limits: mostly confident answers for
familiar species from different sections, plus one "probably section X" and one "not sure". Each example must sit
clearly inside its state (margins below), so small JPEG-decoder differences between Python
and browsers cannot flip it; privacy-check.mjs --examples checks the browser agrees.

Never picks the browser-parity fixture photos or photos already on the species pages.

Usage:
  LD_LIBRARY_PATH=<env>/lib PYTHONPATH=src <sundew-seg python> scripts/build_site_examples.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sundew_segmentation.species_decision import (  # noqa: E402
    E_ANSWER, P_ANSWER, P_SECTION, decide, load_sections, softmax_t,
)

CORPUS = ROOT.parent / "sundew-species-corpus"
RECORDS = CORPUS / "split-110-test/species-records.jsonl"
SECTIONS = ROOT / "data/species-110-sections.json"
LABELS = ROOT / "release/species-v1.0.0/labels.json"
MODEL = ROOT / "release/species-v1.0.0/model-int8.onnx"
LOGITS = ROOT / "models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/val-logits.npz"
FIXTURE = ROOT / "handoff/browser-parity/fixture/expected.json"
SPECIES_JSON = ROOT / "data/site/species.json"
OUT = ROOT / "site/img/examples"

LONG_SIDE, THUMB = 768, 200
JPEG_QUALITY = 85
N_ANSWERS = 5
LICENCES = {"cc0": "CC0 1.0", "cc-by": "CC BY 4.0"}
# Margins on the reference values, so the browser lands in the same state.
ANSWER = dict(p_min=0.85, e_min=E_ANSWER + 0.5)
SECTION = dict(sp_min=P_SECTION + 0.1, p_max=P_ANSWER - 0.2)
NOT_SURE = dict(p_max=P_ANSWER - 0.25, sp_max=P_SECTION - 0.15)


def model():
    import onnxruntime as ort
    from finetune_species_backbone import eval_transform
    tf = eval_transform("full", 224, [0, 0, 0], [1, 1, 1])
    tf.transforms = tf.transforms[:-1]  # normalisation is inside the graph
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 4, 1
    sess = ort.InferenceSession(str(MODEL), so, providers=["CPUExecutionProvider"])
    return lambda im: sess.run(None, {"pixels": tf(im).unsqueeze(0).numpy()})[0][0].astype(np.float64)


def write_example(r: dict, out: Path) -> tuple[Path, Path]:
    jpg, thumb = out / f"{r['photo_id']}.jpg", out / f"{r['photo_id']}-thumb.webp"
    with Image.open(r["local_path"]) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
        im.save(jpg, "JPEG", quality=JPEG_QUALITY, optimize=True)  # no EXIF written
        t = im.copy()
        t.thumbnail((THUMB, THUMB), Image.Resampling.LANCZOS)
        t.save(thumb, "WEBP", quality=75, method=6)
    return jpg, thumb


def main() -> int:
    labels = json.loads(LABELS.read_text())
    sections = load_sections(SECTIONS, labels)
    banned = {p["photo_id"] for p in json.loads(FIXTURE.read_text())["photos"]}
    banned |= {p["photo_id"] for s in json.loads(SPECIES_JSON.read_text())["species"] for p in s["photos"]}
    recs, photos_per_species = {}, {}
    with RECORDS.open() as fh:
        for line in fh:
            r = json.loads(line)
            photos_per_species[r["label"]] = photos_per_species.get(r["label"], 0) + 1
            if r["split"] == "validation" and r["license_code"] in LICENCES and r["photo_id"] not in banned:
                recs[r["photo_id"]] = r

    # Shortlist from the validation scores (576 px frames); the committed JPEG is re-scored below.
    d = np.load(LOGITS, allow_pickle=False)
    pre = []
    for pid, y, lg in zip(d["val_photo_id"].tolist(), d["val_label"].tolist(), d["scores_onnx_int8"]):
        if pid in recs:
            dec = decide(lg, labels, sections)
            pre.append((pid, labels[y], dec))
    want = {
        "answer": [x for x in pre if x[2]["state"] == "answer" and x[2]["label"] == x[1]],
        "section": [x for x in pre if x[2]["state"] == "section" and x[2]["section"] == sections[x[1]]
                    and x[1] in [t["label"] for t in x[2]["top"]]],
        "not-sure": [x for x in pre if x[2]["state"] == "not-sure" and x[1] in [t["label"] for t in x[2]["top"][:3]]],
    }
    # Answers: familiar species first (most photographed on iNaturalist); the others: clearest of their kind first.
    want["answer"].sort(key=lambda x: (-photos_per_species[x[1]], -x[2]["p"], x[0]))
    want["section"].sort(key=lambda x: (-x[2]["section_p"], x[0]))
    want["not-sure"].sort(key=lambda x: (x[2]["p"], x[0]))

    run = model()
    # Build in a scratch folder and swap it in only on success, so a failed run keeps the committed examples.
    tmp = OUT.with_name(OUT.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    chosen, used_sections, used_observers, used_species = [], set(), set(), set()

    def ok(state, dec):
        if state == "answer":
            return dec["state"] == "answer" and dec["p"] >= ANSWER["p_min"] and dec["energy"] >= ANSWER["e_min"]
        if state == "section":
            return dec["state"] == "section" and dec["section_p"] >= SECTION["sp_min"] and dec["p"] <= SECTION["p_max"]
        return (dec["state"] == "not-sure" and dec["p"] <= NOT_SURE["p_max"]
                and (dec["section_p"] or 0) <= NOT_SURE["sp_max"])

    for state, n in (("answer", N_ANSWERS), ("section", 1), ("not-sure", 1)):
        got = 0
        for pid, true, _ in want[state]:
            r = recs[pid]
            if (r["observer_login"] in used_observers or true in used_species
                    or (state == "answer" and sections[true] in used_sections)):
                continue
            jpg, thumb = write_example(r, tmp)
            with Image.open(jpg) as im:
                dec = decide(run(im.convert("RGB")), labels, sections)
            # Answers must name the species; a section must be the species' own; and for "section" and
            # "not sure", the species must still be among the five matches shown.
            in_top5 = true in [t["label"] for t in dec["top"]]
            right = (dec["label"] == true if state == "answer"
                     else in_top5 and dec["section"] == sections[true] if state == "section" else in_top5)
            if not (ok(state, dec) and right):
                jpg.unlink()
                thumb.unlink()
                continue
            used_observers.add(r["observer_login"])
            used_species.add(true)
            used_sections.add(sections[true])
            chosen.append({
                "file": jpg.name, "thumb": thumb.name, "photo_id": pid, "species": true, "section": sections[true],
                "attribution": r["attribution"], "licence": LICENCES[r["license_code"]], "licence_url": r["license_url"],
                "source_page": r["source_page"],
                "expected": {"state": dec["state"], "label": dec["label"], "section": dec["section"],
                             "top1": dec["top"][0]["label"], "p": round(dec["p"], 4), "energy": round(dec["energy"], 4),
                             "section_p": None if dec["section_p"] is None else round(dec["section_p"], 4)},
            })
            print(f"{state:8s} {true:28s} {pid}: p {dec['p']:.3f} energy {dec['energy']:.2f} "
                  f"section_p {dec['section_p'] or 0:.3f} -> {dec['label'] or dec['section'] or 'not sure'}")
            got += 1
            if got == n:
                break
        if got < n:
            print(f"only {got} of {n} {state} examples found; {OUT} left unchanged", file=sys.stderr)
            shutil.rmtree(tmp)
            return 1

    doc = {"note": "Generated by scripts/build_site_examples.py; do not edit. 'expected' is the Python reference "
                   "(int8 ONNX, training preprocessing) on the committed JPEG, with the site's decision rule.",
           "examples": chosen}
    (tmp / "examples.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    credits = ["# Example photo credits", "",
               "The identifier's example photos: iNaturalist photos used under the licence shown, resized and "
               "re-encoded (metadata removed). Generated by `scripts/build_site_examples.py`.", ""]
    credits += [f"- `{c['file']}` (*{c['species']}*): {c['attribution']}, [{c['licence']}]({c['licence_url']}), "
                f"<{c['source_page']}>" for c in chosen]
    (tmp / "ATTRIBUTION.md").write_text("\n".join(credits) + "\n")
    shutil.rmtree(OUT, ignore_errors=True)
    tmp.rename(OUT)
    size = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {len(chosen)} examples to {OUT} ({size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
