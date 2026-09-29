"""Does BioCLIP-2 score higher on validation photos it may have trained on?

BioCLIP-2 was pretrained on TreeOfLife-200M, which draws on a GBIF occurrence
download of May 2024 (doi:10.15468/dl.bfv433; iNaturalist research-grade
photos reach GBIF through it) and Encyclopedia of Life media accessed August
2024 (Gu et al. 2025, arXiv:2505.23883). A validation photo observed after
that cannot have been in its training set. DINOv2-L (LVD-142M, curated web
images, no iNaturalist dump) is the uncontaminated comparison.

Validation is split by `observed_on` into:

- old: observed on or before --old-end, long enough before the May 2024
  download to have been uploaded and reached research grade. These are
  *possibly* seen, not surely: TreeOfLife-200M filters and deduplicates;
- unseen: observed on or after --unseen-start, after the last source date
  with a margin for mis-entered dates;
- middle: in between. Reported, not used in the difference-in-differences.

For each stratum: frozen linear-probe balanced accuracy of both models (crop
and full arms), fine-tuned seed-mean balanced accuracy (full and crop arms,
best and last epoch), and the BioCLIP-2 advantage (BioCLIP-2 - DINOv2-L).
The key quantity is the difference-in-differences, advantage on old minus
advantage on unseen. Memorised validation photos would make it positive.

Intervals: observer-grouped bootstrap, observers resampled from the whole of
validation, both strata recomputed on each resample; for fine-tunes all five
seeds are rescored on the same resample and averaged. Balanced accuracy is
over the classes present in the (resampled) stratum. The strata differ in
species and observer mix, which moves both models; the difference-in-
differences cancels what the mix does to both alike, not what it does to one
model more than the other. As a check, every quantity is also computed on
the species present in both old and unseen strata.

The held-out test split is not read here: only validation predictions are
loaded, and records are filtered to split == "validation".
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

MODELS = ("bioclip-2", "dinov2-l-reg")
SEEDS = (17, 101, 202, 303, 404)
CORPUS = pathlib.Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-species-corpus/split-110-test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-dir", type=pathlib.Path, default=pathlib.Path("models/species-110/finetune"))
    p.add_argument("--screen-dir", type=pathlib.Path, default=pathlib.Path("models/species-110/screen"))
    p.add_argument("--records", type=pathlib.Path, default=CORPUS / "species-records.jsonl")
    p.add_argument("--old-end", default="2022-12-31", help="last observed_on date in the old stratum")
    p.add_argument("--unseen-start", default="2025-01-01", help="first observed_on date in the unseen stratum")
    p.add_argument("--sensitivity-start", default="2025-07-01",
                   help="a later unseen boundary, reported as a check (the earlier audit's)")
    p.add_argument("--out-md", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260926)
    return p.parse_args()


def ci(b):
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main() -> int:
    args = parse_args()
    rows = {}
    for line in open(args.records, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["split"] == "validation":
                rows[int(r["photo_id"])] = r

    frozen = {m: np.load(args.screen_dir / m / "predictions.npz") for m in MODELS}
    tuned = {(m, a, s, w): np.load(args.finetune_dir / m / a / f"seed-{s}" / f"predictions-{w}.npz")["scores"]
             for m in MODELS for a in ("full", "crop") for s in SEEDS for w in ("best", "last")}
    ref = np.load(args.finetune_dir / MODELS[0] / "full" / f"seed-{SEEDS[0]}" / "predictions-last.npz")
    y, observers, pid = ref["val_label"], ref["val_observer"], ref["val_photo_id"]
    for z in frozen.values():
        assert np.array_equal(z["val_photo_id"], pid) and np.array_equal(z["val_label"], y), "order differs"
    for m in MODELS:
        for a in ("full", "crop"):
            for s in SEEDS:
                z = np.load(args.finetune_dir / m / a / f"seed-{s}" / "predictions-best.npz")
                assert np.array_equal(z["val_photo_id"], pid), f"{m}/{a}/{s}: order differs"
    date = np.array([rows[int(p)]["observed_on"] or "" for p in pid])
    assert (date != "").all(), "validation photo without observed_on"

    n = 110
    strata = {"old": date <= args.old_end,
              "middle": (date > args.old_end) & (date < args.unseen_start),
              "unseen": date >= args.unseen_start,
              "unseen-late": date >= args.sensitivity_start}
    shared = np.zeros(n, bool)
    shared[np.intersect1d(y[strata["old"]], y[strata["unseen"]])] = True
    shared_late = np.zeros(n, bool)
    shared_late[np.intersect1d(y[strata["old"]], y[strata["unseen-late"]])] = True

    uniq, obs_idx = np.unique(observers, return_inverse=True)
    rng = np.random.default_rng(args.seed)
    W = np.vstack([np.ones(len(y))] + [np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[obs_idx]
                                       for _ in range(args.reps)]).astype(float)  # row 0 = point estimate
    onehot = np.eye(n)[y]

    def bal(pred, mask, classes=None):
        """Balanced accuracy on each weight row, over classes present in the stratum."""
        Wm = W * mask
        tot = Wm @ onehot
        ok = Wm @ (onehot * (pred == y)[:, None])
        have = tot > 0
        if classes is not None:
            have &= classes
        return np.where(have, ok / np.where(tot > 0, tot, 1), 0).sum(1) / have.sum(1)

    systems = {("frozen", "crop"): {m: [frozen[m]["crop/linear"].argmax(1)] for m in MODELS},
               ("frozen", "full"): {m: [frozen[m]["full/linear"].argmax(1)] for m in MODELS}}
    for a in ("full", "crop"):
        for w in ("last", "best"):
            systems[f"fine-tuned {w}", a] = {m: [tuned[m, a, s, w].argmax(1) for s in SEEDS] for m in MODELS}

    counts = {k: {"images": int(v.sum()), "observers": int(len(np.unique(observers[v]))),
                  "classes": int(len(np.unique(y[v])))} for k, v in strata.items()}
    counts["shared classes (old and unseen)"] = int(shared.sum())
    counts["shared classes (old and unseen-late)"] = int(shared_late.sum())
    summary = {"validation_images": int(len(y)), "validation_observers": int(len(uniq)), "bootstrap_reps": args.reps,
               "boundaries": {"old_end": args.old_end, "unseen_start": args.unseen_start,
                              "sensitivity_start": args.sensitivity_start},
               "training_data_sources": {"gbif_download": "May 2024, doi:10.15468/dl.bfv433",
                                         "eol_accessed": "August 2024", "reference": "arXiv:2505.23883"},
               "strata": counts, "results": {}}

    for (kind, arm), preds in systems.items():
        for unseen_key, cls_all, cls_shared in (("unseen", None, shared), ("unseen-late", None, shared_late)):
            for cls_name, cls in (("stratum classes", cls_all), ("shared classes", cls_shared)):
                acc = {}
                for m in MODELS:
                    for s in ("old", "middle", unseen_key):
                        if s == "middle" and cls is not None:
                            continue
                        acc[m, s] = np.mean([bal(p, strata[s], cls) for p in preds[m]], axis=0)
                adv = {s: acc["bioclip-2", s] - acc["dinov2-l-reg", s] for s in {k[1] for k in acc}}
                did = adv["old"] - adv[unseen_key]
                key = f"{kind}/{arm} | old vs {unseen_key} | {cls_name}"
                summary["results"][key] = {
                    "accuracy": {f"{m}/{s}": {"mean": float(v[0]), "ci95": ci(v[1:])} for (m, s), v in acc.items()},
                    "bioclip_advantage": {s: {"mean": float(v[0]), "ci95": ci(v[1:])} for s, v in adv.items()},
                    "did": {"mean": float(did[0]), "ci95": ci(did[1:]), "p_le_0": float(np.mean(did[1:] <= 0))},
                }

    def fmt(e, sign=False):
        f = "+.3f" if sign else ".3f"
        return f"{e['mean']:{f}} [{e['ci95'][0]:{f}}, {e['ci95'][1]:{f}}]"

    c = counts
    lines = [
        "# BioCLIP-2 contamination check: results table",
        "",
        "Generated by `scripts/check_bioclip_contamination.py`; do not edit by hand.",
        "Interpretation lives in `docs/reports/species-finetune.md`.",
        "",
        f"Validation: {len(y)} images, {len(uniq)} observers. BioCLIP-2's training data: GBIF occurrence download "
        f"May 2024 (doi:10.15468/dl.bfv433), EOL accessed August 2024 (arXiv:2505.23883). "
        f"Strata by `observed_on`: old <= {args.old_end}; unseen >= {args.unseen_start}; "
        f"unseen-late >= {args.sensitivity_start} (sensitivity check); middle in between, not in the DiD. "
        f"Intervals: observer-grouped bootstrap over all of validation, {args.reps} resamples, both strata "
        f"recomputed per resample; fine-tunes rescore all five seeds per resample.",
        "",
        "## Strata",
        "",
        "| Stratum | Images | Observers | Species |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in ("old", "middle", "unseen", "unseen-late"):
        lines.append(f"| {k} | {c[k]['images']} | {c[k]['observers']} | {c[k]['classes']} |")
    lines += ["", f"Species present in both old and unseen: {c['shared classes (old and unseen)']}; "
              f"in both old and unseen-late: {c['shared classes (old and unseen-late)']}.", ""]

    lines += ["## Difference-in-differences", "",
              "BioCLIP-2 advantage (BioCLIP-2 - DINOv2-L) on old minus on unseen. Positive is what memorised "
              "validation photos would produce. Fine-tuned rows are seed means.", "",
              "| System | Advantage, old | Advantage, middle | Advantage, unseen | DiD [95% CI] | P(DiD <= 0) |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for (kind, arm) in systems:
        r = summary["results"][f"{kind}/{arm} | old vs unseen | stratum classes"]
        a = r["bioclip_advantage"]
        lines.append(f"| {kind}, `{arm}` | {fmt(a['old'], True)} | {fmt(a['middle'], True)} | {fmt(a['unseen'], True)}"
                     f" | **{fmt(r['did'], True)}** | {r['did']['p_le_0']:.3f} |")
    lines += ["", "### Checks: shared species, and the later boundary", "",
              "| System | DiD, shared species | DiD, unseen-late | DiD, unseen-late, shared species |",
              "| --- | --- | --- | --- |"]
    for (kind, arm) in systems:
        g = lambda u, s: fmt(summary["results"][f"{kind}/{arm} | old vs {u} | {s}"]["did"], True)
        lines.append(f"| {kind}, `{arm}` | {g('unseen', 'shared classes')} | {g('unseen-late', 'stratum classes')}"
                     f" | {g('unseen-late', 'shared classes')} |")
    lines += ["", "## Balanced accuracy per stratum", "",
              "| System | Model | Old | Middle | Unseen |", "| --- | --- | --- | --- | --- |"]
    for (kind, arm) in systems:
        r = summary["results"][f"{kind}/{arm} | old vs unseen | stratum classes"]["accuracy"]
        for m in MODELS:
            lines.append(f"| {kind}, `{arm}` | `{m}` | {fmt(r[f'{m}/old'])} | {fmt(r[f'{m}/middle'])}"
                         f" | {fmt(r[f'{m}/unseen'])} |")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
