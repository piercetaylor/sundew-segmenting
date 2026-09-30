"""Score the open-set test once: how often does the species model name a sundew for a photo it doesn't know?

Pre-registered in docs/reports/species-open-set-prereg.md. Reads the records
written by scripts/acquire_open_set.py --dedup, runs the shipped int8 model
with the site's preprocessing and decision rule, and writes
docs/reports/species-open-set.{md,json}.

Primary: false-accept rate (share with p >= 0.65) on e = e1 + e2; pass if
below 20%. Intervals: 2,000 bootstrap resamples of observers.

Run from the repo root:
  PYTHONPATH=src python scripts/score_open_set.py \
      --records <sundew-open-set>/open-set-records.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sundew_segmentation.species_decision import P_ANSWER, decide, load_sections, softmax_t

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/species-v1.0.0"
PASS_BAR = 0.20
BOOT_REPS, BOOT_SEED = 2000, 20260930
IN_LIST_COVERAGE = 0.778  # validation, docs/reports/species-abstain.md


def _preprocess():
    spec = importlib.util.spec_from_file_location("oos", ROOT / "scripts/score_out_of_list_photos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.preprocess


def rate_ci(flags: np.ndarray, observers: np.ndarray) -> tuple[float, list[float]]:
    """Share of True, with an observer-clustered bootstrap 95% interval."""
    uniq, inv = np.unique(observers, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(BOOT_SEED)
    reps = []
    for _ in range(BOOT_REPS):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        reps.append(flags[idx].mean())
    return float(flags.mean()), [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=RELEASE / "model-int8.onnx")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/reports/species-open-set")
    args = ap.parse_args()

    import onnxruntime as ort

    preprocess = _preprocess()
    labels = json.loads((RELEASE / "labels.json").read_text())
    temp = json.loads((RELEASE / "release.json").read_text())["temperature"]
    sections = load_sections(ROOT / "data/species-110-sections.json", labels)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(args.model), so, providers=["CPUExecutionProvider"])

    records = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    rows = []
    for i, r in enumerate(records):
        logits = sess.run(None, {"pixels": preprocess(Path(r["local_path"]))})[0][0]
        d = decide(softmax_t(logits, temp), labels, sections)
        rows.append({"source": r["source"], "group": r["group"], "taxon_name": r["taxon_name"],
                     "observation_id": r["observation_id"], "photo_id": r["photo_id"],
                     "observer_login": r["observer_login"], "license_code": r["license_code"],
                     "state": d["state"], "p": round(d["p"], 4), "top1": d["top"][0]["label"],
                     "section": d["section"]})
        if (i + 1) % 100 == 0:
            print(f"scored {i + 1}/{len(records)}", flush=True)

    def summary(sel):
        f = np.array([r["state"] == "answer" for r in sel])
        obs = np.array([r["observer_login"] for r in sel])
        far, ci = rate_ci(f, obs)
        sec = float(np.mean([r["state"] == "section" for r in sel]))
        return {"photos": len(sel), "observers": int(len(set(obs))), "false_accept": far, "ci95": ci,
                "section_fallback": sec, "not_sure": 1 - far - sec}

    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    res = {s: summary(by[s]) for s in ("d", "e1", "e2") if by[s]}
    res["e"] = summary(by["e1"] + by["e2"])
    passed = res["e"]["false_accept"] < PASS_BAR
    groups = {s: {g: {"photos": n, "false_accept": float(np.mean([r["state"] == "answer" for r in by[s] if r["group"] == g]))}
                  for g, n in Counter(r["group"] for r in by[s]).items()} for s in ("e1",)}
    out = {"pre_registration": "docs/reports/species-open-set-prereg.md", "pass_bar": PASS_BAR, "passed": passed,
           "p_answer": P_ANSWER, "temperature": temp,
           "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
           "results": res, "e1_by_genus": groups["e1"], "photos": rows}
    args.out.with_suffix(".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    L = ["# Open-set test of the species model", "",
         "Pre-registered in [species-open-set-prereg.md](species-open-set-prereg.md) before any photo was pulled; "
         "scored once with the shipped int8 model and the site's rule (species named at p >= 0.65). Generated by "
         "`scripts/score_open_set.py`.", "",
         f"**Result: {'PASS' if passed else 'FAIL'}.** On other plants (e = e1 + e2, {res['e']['photos']} photos), "
         f"{pct(res['e']['false_accept'])} got a confident sundew name (95% {pct(res['e']['ci95'][0])}-"
         f"{pct(res['e']['ci95'][1])}, by observer), against a pre-registered bar of below {pct(PASS_BAR)}. "
         f"For comparison, {pct(IN_LIST_COVERAGE)} of in-list validation photos get a species name at the same threshold.", "",
         "| Source | Photos | Observers | Confident species name (95% CI) | Section only | Not sure |",
         "| --- | ---: | ---: | --- | ---: | ---: |"]
    names = {"d": "d: *Drosera* not in the 110 (no bar)", "e1": "e1: carnivorous look-alikes",
             "e2": "e2: other plants", "e": "**e = e1 + e2 (pass bar < 20%)**"}
    for s in ("d", "e1", "e2", "e"):
        if s in res:
            x = res[s]
            L.append(f"| {names[s]} | {x['photos']} | {x['observers']} | {pct(x['false_accept'])} "
                     f"({pct(x['ci95'][0])}-{pct(x['ci95'][1])}) | {pct(x['section_fallback'])} | {pct(x['not_sure'])} |")
    L += ["", "## Carnivorous look-alikes by genus", "", "| Genus | Photos | Confident species name |", "| --- | ---: | ---: |"]
    for g, x in sorted(groups["e1"].items()):
        L.append(f"| *{g}* | {x['photos']} | {pct(x['false_accept'])} |")
    L += ["", "## Most confident mistakes", "", "| Source | Photo is | Named as | p |", "| --- | --- | --- | ---: |"]
    for r in sorted((r for r in rows if r["state"] == "answer"), key=lambda r: -r["p"])[:15]:
        L.append(f"| {r['source']} | [*{r['taxon_name']}*](https://www.inaturalist.org/observations/{r['observation_id']}) "
                 f"| *{r['top1']}* | {r['p']:.2f} |")
    L += ["", "## Caveats", "",
          "- Research-grade iNaturalist photos only: mostly wild plants, photographed by people who post to iNaturalist. "
          "Site users may photograph cultivated plants and other things (people, objects) not tested here.",
          "- Python ONNX Runtime on x86; the browser's WASM int8 kernels agree on about 99% of top-1 answers, so "
          "photos near 0.65 can land differently on the site.",
          "- Photo ids and licences are in `species-open-set.json`; the photos are not in the repository."]
    args.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
