"""Tabulate the stage-2 fine-tunes written by finetune_species_backbone.py.

Reads <finetune-dir>/<model>/<arm>/seed-<seed>/classifier-metrics.json and
predictions-{best,last}.npz, and writes a Markdown table plus a JSON summary:

- per-seed validation balanced accuracy, best and last epoch;
- paired differences of the seed mean (arm vs arm, model vs model) with an
  observer-grouped bootstrap interval: every seed is rescored on the same
  resample of observers, then averaged. This is the primary interval, because
  it covers both seed noise and which photos happen to be in validation;
- the seed-paired t interval on 4 df, as in docs/reports/species-110-baseline.md,
  as a secondary check. It treats validation as fixed, so it is narrower;
- the 5-seed ensemble (mean softmax) with an observer-grouped bootstrap
  interval, and paired ensemble differences on shared resamples.

The best epoch is chosen on validation, so best-epoch numbers carry a small
selection bias; last-epoch numbers do not and are reported beside them.
The held-out test split is not read here.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

MODELS = ("dinov2-l-reg", "bioclip-2")
ARMS = ("full", "crop", "full-square")
SEEDS = (17, 101, 202, 303, 404)
T975_4DF = 2.776445
# ResNet-18 rerun on split-110-test, 5 seeds, best epoch.
# docs/reports/species-110-baseline.md
ANCHOR = {"full": 0.4985, "crop": 0.5156}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--finetune-dir", type=pathlib.Path, default=pathlib.Path("models/species-110/finetune"))
    p.add_argument("--out-md", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS),
                   help="model tags; model differences are first minus each later one")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--title", default="Fine-tuned teachers")
    p.add_argument("--report", default="docs/reports/species-finetune.md", help="where the interpretation lives")
    p.add_argument("--reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260926)
    return p.parse_args()


def balanced(y, pred, weight, n):
    """Balanced accuracy where each image carries a bootstrap multiplicity."""
    tot = np.bincount(y, weights=weight, minlength=n)
    ok = np.bincount(y, weights=weight * (pred == y), minlength=n)
    have = tot > 0
    return float(np.mean(ok[have] / tot[have]))


def softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def paired_t(d):
    d = np.asarray(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"mean": float(d.mean()), "t": float(d.mean() / se),
            "ci95": [float(d.mean() - T975_4DF * se), float(d.mean() + T975_4DF * se)],
            "wins": int((d > 0).sum())}


def main() -> int:
    global MODELS, ARMS
    args = parse_args()
    MODELS, ARMS = tuple(args.models), tuple(args.arms)
    runs = {}
    for m in MODELS:
        for a in ARMS:
            for s in SEEDS:
                d = args.finetune_dir / m / a / f"seed-{s}"
                runs[m, a, s] = (json.loads((d / "classifier-metrics.json").read_text()),
                                 np.load(d / "predictions-best.npz"), np.load(d / "predictions-last.npz"))

    first = runs[MODELS[0], ARMS[0], SEEDS[0]][1]
    y, observers = first["val_label"], first["val_observer"]
    for k, (_, zb, zl) in runs.items():
        assert np.array_equal(zb["val_label"], y) and np.array_equal(zl["val_label"], y), f"{k}: order differs"
    n = int(y.max()) + 1
    ones = np.ones(len(y))
    uniq, obs_idx = np.unique(observers, return_inverse=True)
    rng = np.random.default_rng(args.seed)
    weights = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[obs_idx]
                        for _ in range(args.reps)]).astype(float)

    score = {}  # (model, arm, seed, which) -> balanced accuracy
    for (m, a, s), (_, zb, zl) in runs.items():
        score[m, a, s, "best"] = balanced(y, zb["scores"].argmax(1), ones, n)
        score[m, a, s, "last"] = balanced(y, zl["scores"].argmax(1), ones, n)

    # Seed-mean balanced accuracy on each bootstrap resample, per arm.
    seed_boot = {}
    for m in MODELS:
        for a in ARMS:
            for w in ("best", "last"):
                z = 1 if w == "best" else 2
                preds = [runs[m, a, s][z]["scores"].argmax(1) for s in SEEDS]
                seed_boot[m, a, w] = np.mean([[balanced(y, p, wt, n) for wt in weights] for p in preds], axis=0)

    summary = {"validation_images": int(len(y)), "validation_observers": int(len(uniq)),
               "bootstrap_reps": args.reps, "anchor_resnet18": ANCHOR, "per_seed": {}, "paired": {},
               "seed_mean": {}, "paired_bootstrap": {}, "ensemble": {}, "ensemble_paired": {}}
    for m in MODELS:
        for a in ARMS:
            summary["seed_mean"][f"{m}/{a}"] = {
                w: {"mean": float(np.mean([score[m, a, s, w] for s in SEEDS])),
                    "ci95": [float(np.percentile(seed_boot[m, a, w], q)) for q in (2.5, 97.5)]}
                for w in ("best", "last")}
            summary["per_seed"][f"{m}/{a}"] = {
                "best": [score[m, a, s, "best"] for s in SEEDS],
                "last": [score[m, a, s, "last"] for s in SEEDS],
                "best_epoch": [runs[m, a, s][0]["best_epoch"] for s in SEEDS],
                "epochs_run": [runs[m, a, s][0]["epochs_run"] for s in SEEDS],
            }

    pairs = [(m, x, m, z) for m in MODELS for x, z in (("crop", "full"), ("crop", "full-square"), ("full", "full-square"))
             if x in ARMS and z in ARMS]
    pairs += [(m1, a, m0, a) for i, m1 in enumerate(MODELS) for m0 in MODELS[i + 1:] for a in ARMS]
    for m1, a1, m0, a0 in pairs:
        key = f"{m1}/{a1} - {m0}/{a0}"
        summary["paired"][key] = {w: paired_t([score[m1, a1, s, w] - score[m0, a0, s, w] for s in SEEDS])
                                  for w in ("best", "last")}
        summary["paired_bootstrap"][key] = {}
        for w in ("best", "last"):
            d = seed_boot[m1, a1, w] - seed_boot[m0, a0, w]
            summary["paired_bootstrap"][key][w] = {
                "mean": summary["paired"][key][w]["mean"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_le_0": float(np.mean(d <= 0))}

    boot = {}
    for m in MODELS:
        for a in ARMS:
            prob = np.mean([softmax(runs[m, a, s][2]["scores"]) for s in SEEDS], axis=0)  # last epoch
            pred = prob.argmax(1)
            top5 = np.argsort(-prob, 1)[:, :5]
            b = np.array([balanced(y, pred, w, n) for w in weights])
            boot[m, a] = b
            summary["ensemble"][f"{m}/{a}"] = {
                "balanced_accuracy": balanced(y, pred, ones, n),
                "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                "top5_accuracy": float(np.mean((top5 == y[:, None]).any(1))),
            }
    for m1, a1, m0, a0 in pairs:
        d = boot[m1, a1] - boot[m0, a0]
        point = summary["ensemble"][f"{m1}/{a1}"]["balanced_accuracy"] - summary["ensemble"][f"{m0}/{a0}"]["balanced_accuracy"]
        summary["ensemble_paired"][f"{m1}/{a1} - {m0}/{a0}"] = {
            "mean": point, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_le_0": float(np.mean(d <= 0))}

    lines = [
        f"# {args.title}: results table",
        "",
        "Generated by `scripts/summarize_species_finetune.py`; do not edit by hand.",
        f"Interpretation lives in `{args.report}`.",
        "",
        f"Validation: {len(y)} images, {len(uniq)} observers, 110 species. Training split `split-110-test`. "
        f"Ensemble intervals: observer-grouped bootstrap, {args.reps} resamples. "
        f"ResNet-18 anchor on the same split: full {ANCHOR['full']:.3f}, crop {ANCHOR['crop']:.3f}.",
        "",
        "## Per seed, balanced accuracy",
        "",
        "| Model | Arm | " + " | ".join(str(s) for s in SEEDS) + " | Mean best | SD | Mean last | Best epochs | Epochs run |",
        "| --- | --- | " + " | ".join("---:" for _ in SEEDS) + " | ---: | ---: | ---: | --- | --- |",
    ]
    for m in MODELS:
        for a in ARMS:
            p = summary["per_seed"][f"{m}/{a}"]
            lines.append(f"| `{m}` | `{a}` | " + " | ".join(f"{v:.4f}" for v in p["best"])
                         + f" | **{np.mean(p['best']):.4f}** | {np.std(p['best'], ddof=1):.4f} | {np.mean(p['last']):.4f}"
                         + f" | {', '.join(map(str, p['best_epoch']))} | {', '.join(map(str, p['epochs_run']))} |")
    lines += ["", "## Seed mean with observer-bootstrap interval", "",
              "| Model | Arm | Best epoch [95% CI] | Last epoch [95% CI] |", "| --- | --- | --- | --- |"]
    for key, v in summary["seed_mean"].items():
        m, a = key.split("/")
        b, l = v["best"], v["last"]
        lines.append(f"| `{m}` | `{a}` | {b['mean']:.4f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"
                     f" | {l['mean']:.4f} [{l['ci95'][0]:.3f}, {l['ci95'][1]:.3f}] |")
    lines += ["", "## Paired differences of the seed mean", "",
              "Primary: observer-grouped bootstrap (seed noise and validation sampling). "
              "Secondary: seed-paired t on 4 df (seed noise only, validation held fixed).", "",
              "| Comparison | Best epoch | Bootstrap 95% CI | P(delta <= 0) | t 95% CI | Wins | Last epoch | Bootstrap 95% CI |",
              "| --- | ---: | --- | ---: | --- | ---: | ---: | --- |"]
    for key, v in summary["paired"].items():
        b, l = v["best"], v["last"]
        bb, bl = summary["paired_bootstrap"][key]["best"], summary["paired_bootstrap"][key]["last"]
        lines.append(f"| {key} | {b['mean']:+.4f} | [{bb['ci95'][0]:+.3f}, {bb['ci95'][1]:+.3f}] | {bb['p_le_0']:.3f}"
                     f" | [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | {b['wins']}/5"
                     f" | {l['mean']:+.4f} | [{bl['ci95'][0]:+.3f}, {bl['ci95'][1]:+.3f}] |")
    lines += ["", "## 5-seed ensemble (mean softmax, last epoch)", "",
              "| Model | Arm | Balanced acc [95% CI] | Top-5 |", "| --- | --- | --- | ---: |"]
    for key, v in summary["ensemble"].items():
        m, a = key.split("/")
        lines.append(f"| `{m}` | `{a}` | {v['balanced_accuracy']:.4f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v['top5_accuracy']:.4f} |")
    lines += ["", "| Ensemble comparison | Delta | 95% CI | P(delta <= 0) |", "| --- | ---: | --- | ---: |"]
    for key, v in summary["ensemble_paired"].items():
        lines.append(f"| {key} | {v['mean']:+.4f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_le_0']:.3f} |")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
