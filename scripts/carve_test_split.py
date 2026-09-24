"""Carve a held-out test split out of the training side of an existing species split.

Stage 1 of "After the screen" in docs/species-classifier-plan.md. Every
configuration so far was selected on validation, and the fine-tune stage would
select again. The test split is scored once, after everything is fixed.

Why from TRAIN, not from validation: validation has already chosen the screened
backbones, so photos taken from it are not untouched. Taking from train leaves
validation byte-identical, so every validation number already reported still
reads the same; the cost is a smaller training set, and any model trained on
the old split has seen the test photos and cannot be scored on them.

Observer order is scarce-first, then random, not largest-first as in
build_species_split.py. Largest-first concentrates the split in a few big
observers (validation: 308 observers, the top 50 hold 57% of it). Pure random
order spreads it well but starves the Western Australian endemics: they are
photographed almost only by prolific observers covering 20-40 species each,
and by the time one comes up the common classes' quotas are full, so it no
longer fits (D. nitidula got 2 of 6, D. androsacea 3 of 9). Visiting first the
observers of the classes with the fewest training observers, in random order
within that, meets every quota and still gives ~1,100 test observers with the
top 50 holding about a third.

Each class has a hard quota, --test-fraction of its total images across all
splits. An observer goes to test only if none of its classes would exceed its
quota, so multi-species observers cannot overshoot a thin class.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter, defaultdict
from random import Random


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=pathlib.Path, required=True, help="Existing split directory (train/validation).")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260923)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.resolve() == args.split.resolve():
        raise SystemExit("refusing to overwrite the source split; models trained on it reference it")
    rows = [json.loads(l) for l in open(args.split / "species-records.jsonl", encoding="utf-8") if l.strip()]
    labels = json.loads((args.split / "labels.json").read_text())
    if {r["split"] for r in rows} != {"train", "validation"}:
        raise SystemExit("source split must contain exactly train and validation")

    total = Counter(r["label"] for r in rows)
    quota = {lab: math.floor(args.test_fraction * total[lab]) for lab in labels}
    by_observer: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_observer[r["observer_login"]].append(r)

    # Scarcity of an observer = fewest training observers among its classes.
    n_obs = Counter(lab for group in by_observer.values() for lab in {r["label"] for r in group})
    rng = Random(args.seed)
    tiebreak = {o: rng.random() for o in sorted(by_observer)}
    order = sorted(by_observer, key=lambda o: (min(n_obs[r["label"]] for r in by_observer[o]), tiebreak[o]))
    have: Counter[str] = Counter()
    test_obs = set()
    for obs in order:
        contribution = Counter(r["label"] for r in by_observer[obs])
        if all(have[lab] + k <= quota[lab] for lab, k in contribution.items()):
            have += contribution
            test_obs.add(obs)

    for r in rows:
        if r["observer_login"] in test_obs:
            r["split"] = "test"

    splits = defaultdict(set)
    for r in rows:
        splits[r["observer_login"]].add(r["split"])
    if any(len(s) > 1 for s in splits.values()):
        raise SystemExit("observer leakage across splits")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "species-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    count = Counter((r["label"], r["split"]) for r in rows)
    obs_count = Counter()
    for o, s in splits.items():
        obs_count[next(iter(s))] += 1
    print(f"\n{'class':<30}{'train':>7}{'val':>7}{'test':>7}{'quota':>7}")
    for lab in labels:
        print(f"  {lab:<28}{count[lab, 'train']:>7}{count[lab, 'validation']:>7}"
              f"{count[lab, 'test']:>7}{quota[lab]:>7}")
    n = Counter(r["split"] for r in rows)
    print(f"  {'TOTAL':<28}{n['train']:>7}{n['validation']:>7}{n['test']:>7}{sum(quota.values()):>7}")
    print(f"\nobservers: {obs_count['train']} train, {obs_count['validation']} validation, "
          f"{obs_count['test']} test, zero shared")
    short = [lab for lab in labels if count[lab, "test"] < quota[lab]]
    if short:
        print(f"{len(short)} classes under quota: " + ", ".join(
            f"{lab} {count[lab, 'test']}/{quota[lab]}" for lab in short))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
