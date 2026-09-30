"""Abstain ("not sure") rule for the shipped species model, on VALIDATION only.

Reads the shipped model's int8 ONNX validation logits, applies the release
temperature (probabilities = softmax(logits / 0.74)), and reports:

  1. accuracy vs coverage for thresholds on the top-1 probability and on the
     top-1 minus top-2 margin;
  2. top-1/3/5 accuracy per confidence band;
  3. the same split by training count (rare = under 40 training photos);
  4. a section fallback: section probability = sum of its species'
     probabilities; when the species answer abstains, answer the section if its
     probability clears a section threshold;
  5. an observer-clustered bootstrap interval for the recommended rule.

Validation holds only the 110 species, so none of this says anything about
non-sundew or out-of-list photos. The threshold is chosen on the same photos it
is scored on, so the numbers are slightly optimistic.

Never reads models/species-110/test-scoring/.

It also writes the decision test vectors for the site (--vectors): a sample of
validation logits with the decision of sundew_segmentation.species_decision,
which site/test/decision.test.mjs checks the browser code against.

Run from the repo root:
  PYTHONPATH=src python scripts/analyze_species_abstain.py \
      --records <sundew-species-corpus>/split-110-test/species-records.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sundew_segmentation.species_decision import (
    P_ANSWER, P_SECTION, decide, load_sections, softmax_t,
)

ROOT = Path(__file__).resolve().parents[1]
LOGITS = ROOT / "models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/val-logits.npz"
LABELS = ROOT / "release/species-v1.0.0/labels.json"
RELEASE = ROOT / "release/species-v1.0.0/release.json"
SECTIONS = ROOT / "data/species-110-sections.json"
OUT = ROOT / "docs/reports"
VECTORS = ROOT / "site/test/fixtures/decision-vectors.json"

P_GRID = [round(x, 2) for x in np.arange(0.30, 0.901, 0.05)]
M_GRID = [round(x, 2) for x in np.arange(0.00, 0.801, 0.10)]
SECTION_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
BANDS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
TRAIN_BINS = [(0, 40, "under 40 (rare)"), (40, 80, "40-79"), (80, 130, "80-129"), (130, 10**9, "130 and over")]
# Pre-stated rule for the recommendation: the smallest top-1 probability
# threshold on P_GRID whose answered photos are at least this accurate.
TARGET_ANSWERED_ACC = 0.90
BOOT_REPS, BOOT_SEED = 2000, 20260929


def train_counts(records: Path, labels: list[str]) -> np.ndarray:
    train = Counter()
    with records.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "train":
                train[r["label"]] += 1
    return np.array([train[n] for n in labels])


def load(logits: Path, records: Path):
    d = np.load(logits, allow_pickle=False)
    labels = json.loads(LABELS.read_text())
    temp = float(json.loads(RELEASE.read_text())["temperature"])
    sec_of = load_sections(SECTIONS, labels)
    sections = sorted(set(sec_of.values()))
    sp_to_sec = np.array([sections.index(sec_of[n]) for n in labels])
    return {
        "logits": d["scores_onnx_int8"],
        "probs": softmax_t(d["scores_onnx_int8"], temp),
        "y": d["val_label"].astype(int),
        "observer": d["val_observer"],
        "labels": labels, "temp": temp, "sections": sections,
        "sp_to_sec": sp_to_sec, "sec_of": sec_of, "train_counts": train_counts(records, labels),
    }


def balanced(correct: np.ndarray, y: np.ndarray) -> float:
    """Mean over species present of per-species accuracy (NaN if empty)."""
    if correct.size == 0:
        return float("nan")
    per = [correct[y == c].mean() for c in np.unique(y)]
    return float(np.mean(per))


def rule_stats(answer: np.ndarray, correct: np.ndarray, in3: np.ndarray, in5: np.ndarray, y: np.ndarray) -> dict:
    n = answer.size
    a, ab = answer, ~answer
    per_species_cov = [answer[y == c].mean() for c in np.unique(y)]
    return {
        "coverage": float(a.mean()),
        "coverage_species_balanced": float(np.mean(per_species_cov)),
        "answered": int(a.sum()),
        "acc_answered": float(correct[a].mean()) if a.any() else float("nan"),
        "balanced_acc_answered": balanced(correct[a], y[a]),
        "wrong_answers_per_100_photos": float(100 * (a & ~correct).sum() / n),
        "abstained": int(ab.sum()),
        "abstained_top1_would_be_right": float(correct[ab].mean()) if ab.any() else float("nan"),
        "abstained_true_in_top3": float(in3[ab].mean()) if ab.any() else float("nan"),
        "abstained_true_in_top5": float(in5[ab].mean()) if ab.any() else float("nan"),
    }


def analyse(D: dict, mask: np.ndarray | None = None) -> dict:
    P, y = D["probs"], D["y"]
    if mask is not None:
        P, y = P[mask], y[mask]
    order = np.argsort(-P, axis=1)
    p1 = P[np.arange(len(P)), order[:, 0]]
    p2 = P[np.arange(len(P)), order[:, 1]]
    margin = p1 - p2
    correct = order[:, 0] == y
    in3 = (order[:, :3] == y[:, None]).any(1)
    in5 = (order[:, :5] == y[:, None]).any(1)

    # section probabilities
    S = np.zeros((len(P), len(D["sections"])))
    for c, s in enumerate(D["sp_to_sec"]):
        S[:, s] += P[:, c]
    sec_pred = S.argmax(1)
    sec_p = S.max(1)
    y_sec = D["sp_to_sec"][y]
    sec_correct = sec_pred == y_sec
    sec_of_species_top1_correct = D["sp_to_sec"][order[:, 0]] == y_sec

    out = {
        "n": int(len(y)), "species_present": int(len(np.unique(y))),
        "top1": float(correct.mean()), "top3": float(in3.mean()), "top5": float(in5.mean()),
        "balanced_top1": balanced(correct, y),
        "section_acc_summed": float(sec_correct.mean()),
        "section_balanced_summed": balanced(sec_correct, y),
        "section_balanced_via_species_top1": balanced(sec_of_species_top1_correct, y),
        "prob_threshold": {str(t): rule_stats(p1 >= t, correct, in3, in5, y) for t in P_GRID},
        "margin_threshold": {str(t): rule_stats(margin >= t, correct, in3, in5, y) for t in M_GRID},
        "bands": {},
        "section_fallback": {},
    }
    for lo, hi in BANDS:
        b = (p1 >= lo) & (p1 < hi)
        out["bands"][f"{lo:.1f}-{min(hi, 1.0):.1f}"] = {
            "n": int(b.sum()), "share": float(b.mean()),
            "mean_p1": float(p1[b].mean()) if b.any() else float("nan"),
            "top1": float(correct[b].mean()) if b.any() else float("nan"),
            "top3": float(in3[b].mean()) if b.any() else float("nan"),
            "top5": float(in5[b].mean()) if b.any() else float("nan"),
            "section_summed_correct": float(sec_correct[b].mean()) if b.any() else float("nan"),
        }
    for t in P_GRID:
        ab = p1 < t
        row = {"species_abstains": int(ab.sum()),
               "section_acc_on_abstained_no_threshold": float(sec_correct[ab].mean()) if ab.any() else float("nan")}
        for ts in SECTION_GRID:
            sec_ans = ab & (sec_p >= ts)
            row[f"section_threshold_{ts}"] = {
                "section_answered": int(sec_ans.sum()),
                "share_of_abstained": float(sec_ans.sum() / ab.sum()) if ab.any() else float("nan"),
                "section_acc": float(sec_correct[sec_ans].mean()) if sec_ans.any() else float("nan"),
                "still_not_sure_share_of_all": float((ab & ~sec_ans).mean()),
            }
        out["section_fallback"][str(t)] = row
    out["_arrays"] = {"p1": p1, "correct": correct, "in5": in5, "sec_p": sec_p, "sec_correct": sec_correct}
    return out


def bootstrap(D: dict, t: float, ts: float) -> dict:
    """Observer-clustered bootstrap of the recommended rule (validation)."""
    A = analyse(D)["_arrays"]
    obs = D["observer"]
    uniq, inv = np.unique(obs, return_inverse=True)
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(BOOT_SEED)
    cov, acc, sec_acc = [], [], []
    for _ in range(BOOT_REPS):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        p1, c = A["p1"][idx], A["correct"][idx]
        a = p1 >= t
        cov.append(a.mean())
        acc.append(c[a].mean())
        s = (~a) & (A["sec_p"][idx] >= ts)
        sec_acc.append(A["sec_correct"][idx][s].mean() if s.any() else np.nan)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"observers": int(len(uniq)), "reps": BOOT_REPS, "coverage_ci95": ci(cov),
            "acc_answered_ci95": ci(acc), "section_acc_ci95": ci(sec_acc)}


def f(x, pct=True):
    return "-" if x != x else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")


def write_vectors(D: dict, path: Path, n_random: int = 30, seed: int = BOOT_SEED) -> int:
    """Validation logits with the reference decision, for the browser test.

    Picks photos near the species and section thresholds plus a random sample,
    and skips any within 1e-6 of a threshold (float64 exp differs by an ulp
    between numpy and JS, which must not flip the expected answer).
    """
    labels, P = D["labels"], D["probs"]
    p1 = P.max(1)
    S = np.zeros((len(P), len(D["sections"])))
    for c, s in enumerate(D["sp_to_sec"]):
        S[:, s] += P[:, c]
    sec_p = S.max(1)
    safe = (np.abs(p1 - P_ANSWER) > 1e-6) & (np.abs(sec_p - P_SECTION) > 1e-6)
    rng = np.random.default_rng(seed)

    def pick(mask, k):
        idx = np.flatnonzero(mask & safe)
        return rng.choice(idx, size=min(k, idx.size), replace=False) if idx.size else np.array([], int)

    chosen = np.unique(np.concatenate([
        pick((p1 >= P_ANSWER - 0.05) & (p1 < P_ANSWER + 0.05), 25),
        pick((p1 < P_ANSWER) & (np.abs(sec_p - P_SECTION) < 0.08), 25),
        pick(p1 < 0.3, 10),
        pick(np.ones_like(safe), n_random),
    ]))
    cases = []
    for i in chosen:
        # float32 printed shortest: JS and Python parse the same decimal to the same double.
        logits = [float(str(np.float32(v))) for v in D["logits"][i]]
        probs = softmax_t(np.array(logits), D["temp"])
        dec = decide(probs, labels, D["sec_of"])
        cases.append({"val_index": int(i), "logits": logits, "expected": {
            "state": dec["state"], "label": dec["label"], "p": dec["p"],
            "section": dec["section"], "section_p": dec["section_p"],
            "top": [t["label"] for t in dec["top"]],
        }})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "note": ("Generated by scripts/analyze_species_abstain.py from the species model v1.0.0 "
                 "int8 validation logits; expected answers from sundew_segmentation.species_decision."),
        "temperature": D["temp"], "p_answer": P_ANSWER, "p_section": P_SECTION,
        "labels": labels, "cases": cases,
    }, separators=(",", ":")) + "\n")
    return len(cases)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logits", type=Path, default=LOGITS, help="val-logits.npz of the shipped model")
    ap.add_argument("--records", type=Path, required=True,
                    help="split-110-test/species-records.jsonl (training counts per species)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--vectors", type=Path, default=VECTORS)
    args = ap.parse_args()

    D = load(args.logits, args.records)
    counts = D["train_counts"]
    rare_species = counts < 40
    y = D["y"]
    allr = analyse(D)
    subsets = {"rare (<40 train photos)": rare_species[y], "not rare (>=40)": ~rare_species[y]}
    for lo, hi, name in TRAIN_BINS:
        subsets[f"bin {name}"] = (counts[y] >= lo) & (counts[y] < hi)
    subs = {k: analyse(D, m) for k, m in subsets.items()}

    # recommendation
    rec_t = next((t for t in P_GRID if allr["prob_threshold"][str(t)]["acc_answered"] >= TARGET_ANSWERED_ACC), None)
    rec_ts = P_SECTION
    if rec_t != P_ANSWER:
        raise SystemExit(f"pre-stated rule now picks p >= {rec_t}, but the site ships P_ANSWER = {P_ANSWER}: "
                         "update sundew_segmentation.species_decision and site/js/decision.js together")
    boot = bootstrap(D, rec_t, rec_ts)

    result = {
        "note": ("VALIDATION only (110 in-list species). Says nothing about non-sundew or "
                 "out-of-list photos. Threshold chosen on the scored photos: slightly optimistic."),
        "source_logits": str(args.logits)[str(args.logits).find("models/"):] if "models/" in str(args.logits) else str(args.logits), "scores": "scores_onnx_int8",
        "temperature": D["temp"], "n_photos": int(len(y)), "n_observers": int(len(np.unique(D["observer"]))),
        "rare_species": int(rare_species.sum()),
        "recommendation_rule": f"smallest top-1 probability threshold on {P_GRID} with answered accuracy >= {TARGET_ANSWERED_ACC}",
        "recommended_threshold": rec_t, "recommended_section_threshold": rec_ts,
        "bootstrap": boot,
        "all": {k: v for k, v in allr.items() if k != "_arrays"},
        "subsets": {name: {k: v for k, v in s.items() if k != "_arrays"} for name, s in subs.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "species-abstain.json").write_text(json.dumps(result, indent=2) + "\n")
    n_vec = write_vectors(D, args.vectors)

    # ---------- markdown ----------
    L = []
    w = L.append
    w("# Abstain rule for the species model (validation)\n")
    w(f"Generated by `scripts/analyze_species_abstain.py`. Shipped model, int8 ONNX validation logits "
      f"(`{result['source_logits']}`, `scores_onnx_int8`), probabilities = softmax(logits / {D['temp']}). "
      f"{len(y):,} photos, {result['n_observers']} observers, 110 species; {int(rare_species.sum())} species "
      f"have under 40 training photos (\"rare\").\n")
    w("**Read this first.** Validation contains only the 110 listed species. Nothing here says how the model "
      "behaves on non-sundew photos or on *Drosera* outside the list: those may get high probabilities too. "
      "The threshold is picked on the same photos it is scored on, so the numbers are slightly optimistic.\n")
    w(f"Sanity check against `release.json`: top-1 {f(allr['top1'])} (0.800), balanced {allr['balanced_top1']:.4f} "
      f"(0.7635), top-5 {f(allr['top5'])} (95.7%), section balanced {allr['section_balanced_via_species_top1']:.4f} "
      f"via the species top-1 / {allr['section_balanced_summed']:.4f} via summed section probabilities (0.9097).\n")

    R = allr["prob_threshold"][str(rec_t)]
    SF = allr["section_fallback"][str(rec_t)][f"section_threshold_{rec_ts}"]
    w("## Recommendation\n")
    w(f"Rule stated in the script before it was first run: the smallest top-1 probability threshold whose answered photos are at least "
      f"{int(100 * TARGET_ANSWERED_ACC)}% right. That is **p >= {rec_t}**.\n")
    w(f"- **Answer the species when p >= {rec_t}:** covers {f(R['coverage'])} of photos "
      f"(bootstrap 95% {f(boot['coverage_ci95'][0])}-{f(boot['coverage_ci95'][1])}, by observer), "
      f"{f(R['acc_answered'])} of those answers right ({f(boot['acc_answered_ci95'][0])}-{f(boot['acc_answered_ci95'][1])}). "
      f"Species-balanced coverage {f(R['coverage_species_balanced'])}.")
    w(f"- **Otherwise say \"not sure\" and show the top 5.** On the {R['abstained']} abstained photos, the true "
      f"species is in the top 5 {f(R['abstained_true_in_top5'])} of the time (top 3: {f(R['abstained_true_in_top3'])}); "
      f"the top-1 would have been right {f(R['abstained_top1_would_be_right'])}.")
    w(f"- **Section fallback** (section threshold {rec_ts} is a judgement call from the table below, not a pre-stated rule): when the species abstains and the summed section probability is >= {rec_ts}, "
      f"name the section. That answers {f(SF['share_of_abstained'])} of abstained photos at "
      f"{f(SF['section_acc'])} section accuracy (95% {f(boot['section_acc_ci95'][0])}-{f(boot['section_acc_ci95'][1])}); "
      f"{f(SF['still_not_sure_share_of_all'])} of all photos get no species and no section.")
    Rr = subs["rare (<40 train photos)"]["prob_threshold"][str(rec_t)]
    Rc = subs["not rare (>=40)"]["prob_threshold"][str(rec_t)]
    w(f"- **Rare species:** coverage {f(Rr['coverage'])} at {f(Rr['acc_answered'])} right, against "
      f"{f(Rc['coverage'])} at {f(Rc['acc_answered'])} for the rest. Rare species get both fewer answers and "
      f"less accurate ones; the site should not imply the 90% figure holds for every species.\n")

    def ptable(res: dict, key: str, label: str):
        w(f"| {label} | Coverage | Right when answered | Balanced (answered) | Wrong answers / 100 photos | Abstained: true in top 5 |")
        w("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for t, r in res[key].items():
            mark = " **(rec.)**" if key == "prob_threshold" and float(t) == rec_t else ""
            w(f"| {t}{mark} | {f(r['coverage'])} | {f(r['acc_answered'])} | {f(r['balanced_acc_answered'])} | "
              f"{r['wrong_answers_per_100_photos']:.1f} | {f(r['abstained_true_in_top5'])} |")
        w("")

    w("## Accuracy vs coverage: top-1 probability\n")
    w(f"No threshold: every photo answered, {f(allr['top1'])} right, {100 * (1 - allr['top1']):.1f} wrong answers per 100 photos.\n")
    ptable(allr, "prob_threshold", "p >=")
    w("## Accuracy vs coverage: margin (top-1 minus top-2 probability)\n")
    ptable(allr, "margin_threshold", "margin >=")
    w("At matched coverage the margin gives almost the same trade-off (e.g. margin >= 0.5: 76.1% covered, 90.5% right, "
      "vs p >= 0.65: 77.8%, 90.1%). The top-1 probability is simpler to explain, so use it.\n")

    w("## By confidence band\n")
    w("| Top-1 probability | Photos | Share | Mean p | Top-1 right | True in top 3 | True in top 5 | Section right (summed) |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for b, r in allr["bands"].items():
        w(f"| {b} | {r['n']} | {f(r['share'])} | {f(r['mean_p1'], False)} | {f(r['top1'])} | {f(r['top3'])} | {f(r['top5'])} | {f(r['section_summed_correct'])} |")
    w("\nMean p is within about 5 points of top-1 accuracy in every band (a few points too high in the 0.5-0.9 "
      "bands), so on in-list photos the displayed probabilities are roughly honest. That is not true for "
      "out-of-list photos, which were not measured.\n")

    w("## Rare vs common species\n")
    w("| Group | Species | Photos | Top-1 | Top-5 | Coverage at rec. | Right when answered | Abstained: true in top 5 |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, s in subs.items():
        r = s["prob_threshold"][str(rec_t)]
        w(f"| {name} | {s['species_present']} | {s['n']} | {f(s['top1'])} | {f(s['top5'])} | {f(r['coverage'])} | {f(r['acc_answered'])} | {f(r['abstained_true_in_top5'])} |")
    w("")
    w("Confidence bands, rare species only:\n")
    w("| Top-1 probability | Photos | Top-1 right | True in top 3 | True in top 5 |")
    w("| --- | ---: | ---: | ---: | ---: |")
    for b, r in subs["rare (<40 train photos)"]["bands"].items():
        w(f"| {b} | {r['n']} | {f(r['top1'])} | {f(r['top3'])} | {f(r['top5'])} |")
    w("")

    w("## Section fallback\n")
    w("Section probability = sum of the probabilities of its species (16 sections, iNaturalist's infrageneric "
      "ranks; `data/species-110-sections.json`). Rows: species threshold. Cells: section accuracy on the photos "
      "where the species abstains and the section probability clears the section threshold (share of abstained "
      "photos answered in brackets).\n")
    w("| Species p < | Abstained | Section, no threshold | " + " | ".join(f"section p >= {ts}" for ts in SECTION_GRID) + " |")
    w("| ---: | ---: | ---: | " + " | ".join("---:" for _ in SECTION_GRID) + " |")
    for t, row in allr["section_fallback"].items():
        cells = [f"{f(row[f'section_threshold_{ts}']['section_acc'])} ({f(row[f'section_threshold_{ts}']['share_of_abstained'])})" for ts in SECTION_GRID]
        w(f"| {t} | {row['species_abstains']} | {f(row['section_acc_on_abstained_no_threshold'])} | " + " | ".join(cells) + " |")
    w("")
    w("## Caveats\n")
    w("- In-list photos only; open-set behaviour is unmeasured; see `species-out-of-list-check.md` for a small spot check.")
    w("- Threshold and section threshold picked on validation; expect slightly lower numbers in use.")
    w("- Validation is photo-weighted toward common species, like real use, but real users photograph "
      "cultivated plants more than iNaturalist wild photos, and browser preprocessing differs (EXIF, resize).")
    w("- int8 results vary slightly by CPU (99.1% top-1 agreement between two server CPUs), so a photo near the "
      "threshold can flip between answer and \"not sure\".")
    (args.out_dir / "species-abstain.md").write_text("\n".join(L) + "\n")
    print((args.out_dir / "species-abstain.md").read_text())
    print(f"wrote {n_vec} decision vectors to {args.vectors}")


if __name__ == "__main__":
    main()
