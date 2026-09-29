"""Per-species errors and calibration of the chosen teacher, on validation.

The teacher is the 5-seed, last-epoch ensemble (mean softmax) of one fine-tuned
model and arm, by default `dinov2-l-reg` / `full`. Writes a Markdown report and
a JSON summary with:

- calibration: expected calibration error (15 equal-width bins) before and
  after one temperature, fitted by grid search on validation negative
  log-likelihood. Fitted and scored on the same images, so the post-scaling
  ECE is optimistic; the held-out test split checks it;
- per-species accuracy against training count;
- per-section balanced accuracy, and the share of errors that stay in section;
- the most confused species pairs.

The held-out test split is not read here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import numpy as np

SEEDS = (17, 101, 202, 303, 404)
CORPUS = pathlib.Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-species-corpus/split-110-test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-dir", type=pathlib.Path, default=pathlib.Path("models/species-110/finetune"))
    p.add_argument("--model", default="dinov2-l-reg")
    p.add_argument("--arm", default="full")
    p.add_argument("--records", type=pathlib.Path, default=CORPUS / "species-records.jsonl")
    p.add_argument("--labels", type=pathlib.Path, default=CORPUS / "labels.json")
    p.add_argument("--sections", type=pathlib.Path, default=pathlib.Path("data/species-110-sections.json"))
    p.add_argument("--out-md", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-pairs", type=int, default=15)
    return p.parse_args()


def softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ece(prob, y, bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return float(total)


def tempered(logp, t):
    return softmax(logp / t)


def main() -> int:
    args = parse_args()
    labels = json.loads(args.labels.read_text())
    n = len(labels)
    sec_map = json.loads(args.sections.read_text())["species"]
    section = [sec_map[lab]["section"] for lab in labels]
    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    train_count = Counter(r["label"] for r in rows if r["split"] == "train")

    runs = [np.load(args.finetune_dir / args.model / args.arm / f"seed-{s}" / "predictions-last.npz") for s in SEEDS]
    y = runs[0]["val_label"]
    for z in runs:
        assert np.array_equal(z["val_label"], y), "validation order differs between seeds"
    prob = np.mean([softmax(z["scores"]) for z in runs], axis=0)
    pred = prob.argmax(1)
    logp = np.log(np.clip(prob, 1e-12, None))

    # Temperature on the ensemble's log-probabilities.
    grid = np.round(np.arange(0.30, 2.001, 0.01), 2)
    nll = [float(-np.mean(np.log(np.clip(tempered(logp, t)[np.arange(len(y)), y], 1e-12, None)))) for t in grid]
    t_best = float(grid[int(np.argmin(nll))])
    cal = tempered(logp, t_best)

    per_class = []
    for c in range(n):
        m = y == c
        per_class.append({"species": labels[c], "section": section[c], "train": int(train_count[labels[c]]),
                          "validation": int(m.sum()), "accuracy": float(np.mean(pred[m] == c)) if m.any() else None})
    acc = np.array([r["accuracy"] for r in per_class], dtype=float)
    tr = np.array([r["train"] for r in per_class])
    rank = lambda v: np.argsort(np.argsort(v))
    spearman = float(np.corrcoef(rank(tr), rank(acc))[0, 1])
    bands = [(0, 40), (40, 80), (80, 130), (130, 10**9)]
    band_rows = []
    for lo, hi in bands:
        m = (tr >= lo) & (tr < hi)
        band_rows.append({"train_range": [lo, hi], "species": int(m.sum()), "mean_accuracy": float(acc[m].mean())})

    wrong = pred != y
    in_section = float(np.mean([section[p] == section[t] for p, t in zip(pred[wrong], y[wrong])]))
    pairs = Counter()
    for t, p in zip(y[wrong], pred[wrong]):
        pairs[tuple(sorted((int(t), int(p))))] += 1
    directed = Counter(zip(y[wrong].tolist(), pred[wrong].tolist()))
    top_pairs = [{"a": labels[a], "b": labels[b], "errors": k, "a_as_b": directed[a, b], "b_as_a": directed[b, a],
                  "same_section": section[a] == section[b]} for (a, b), k in pairs.most_common(args.top_pairs)]

    sections = sorted(set(section))
    sec_rows = []
    for s in sections:
        cls = [c for c in range(n) if section[c] == s]
        sec_rows.append({"section": s, "species": len(cls), "validation": int(np.isin(y, cls).sum()),
                         "balanced_accuracy": float(np.nanmean([acc[c] for c in cls]))})
    sec_rows.sort(key=lambda r: r["balanced_accuracy"])

    summary = {
        "teacher": f"{args.model}/{args.arm}, 5-seed last-epoch ensemble",
        "validation_images": int(len(y)),
        "balanced_accuracy": float(np.nanmean(acc)), "accuracy": float(np.mean(pred == y)),
        "calibration": {"temperature": t_best, "ece_before": ece(prob, y), "ece_after": ece(cal, y),
                        "mean_confidence_before": float(prob.max(1).mean()),
                        "mean_confidence_after": float(cal.max(1).mean()),
                        "nll_before": nll[int(np.where(grid == 1.0)[0][0])], "nll_after": min(nll)},
        "train_count_spearman": spearman, "train_count_bands": band_rows,
        "errors": int(wrong.sum()), "errors_in_section": in_section,
        "top_pairs": top_pairs, "sections": sec_rows, "per_species": per_class,
    }

    c = summary["calibration"]
    lines = [
        "# Teacher errors and calibration",
        "",
        "Generated by `scripts/analyze_species_teacher.py`; do not edit by hand.",
        "Interpretation lives in `docs/reports/species-finetune.md`.",
        "",
        f"Teacher: `{args.model}` / `{args.arm}`, 5-seed last-epoch ensemble. Validation: {len(y)} images. "
        f"Balanced accuracy {summary['balanced_accuracy']:.4f}, plain accuracy {summary['accuracy']:.4f}.",
        "",
        "## Calibration",
        "",
        "One temperature on the ensemble's log-probabilities, fitted and scored on validation, "
        "so the 'after' column is optimistic.",
        "",
        "| | Before (T = 1) | After |",
        "| --- | ---: | ---: |",
        f"| Temperature | 1.00 | {c['temperature']:.2f} |",
        f"| ECE, 15 bins | {c['ece_before']:.4f} | {c['ece_after']:.4f} |",
        f"| Mean confidence | {c['mean_confidence_before']:.4f} | {c['mean_confidence_after']:.4f} |",
        f"| NLL | {c['nll_before']:.4f} | {c['nll_after']:.4f} |",
        "",
        "## Accuracy against training count",
        "",
        f"Spearman correlation across species: {spearman:.2f}.",
        "",
        "| Training images | Species | Mean accuracy |",
        "| --- | ---: | ---: |",
    ]
    for b in band_rows:
        lo, hi = b["train_range"]
        lines.append(f"| {lo}-{hi - 1 if hi < 10**9 else ''} | {b['species']} | {b['mean_accuracy']:.3f} |")
    lines += ["", f"## Most confused pairs ({summary['errors']} errors; {in_section:.0%} stay within section)", "",
              "| Species A | Species B | Errors | A as B | B as A | Same section |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for p in top_pairs:
        lines.append(f"| *{p['a']}* | *{p['b']}* | {p['errors']} | {p['a_as_b']} | {p['b_as_a']} | {'yes' if p['same_section'] else 'no'} |")
    lines += ["", "## Sections, worst first", "", "| Section | Species | Validation images | Balanced accuracy |",
              "| --- | ---: | ---: | ---: |"]
    for s in sec_rows:
        lines.append(f"| {s['section']} | {s['species']} | {s['validation']} | {s['balanced_accuracy']:.3f} |")
    lines += ["", "## Per species, worst first", "", "| Species | Section | Train | Validation | Accuracy |",
              "| --- | --- | ---: | ---: | ---: |"]
    for r in sorted(per_class, key=lambda r: (r["accuracy"] if r["accuracy"] is not None else 2)):
        a = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "n/a"
        lines.append(f"| *{r['species']}* | {r['section']} | {r['train']} | {r['validation']} | {a} |")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
