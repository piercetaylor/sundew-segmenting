"""Assemble the crop-vs-full-frame classifier dataset with an observer-grouped split.

Two exclusions matter and are applied here rather than left to the training run:

1. Images the SEGMENTATION model trained on. On those the predicted crop is
   unrealistically good, which would hand the crop arm an advantage that a real
   scrape would never give it. This is the whole comparison, so the bias would
   not be a detail.
2. Subspecies labels are folded into their parent species, so the task stays a
   clean 10-way problem.

Observers are assigned whole to one side, as in the segmentation splits, because
a photographer's equipment, habitat and processing leak identity otherwise.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from random import Random


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--species-root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--exclude", type=pathlib.Path, nargs="*", default=[],
                   help="Manifests whose photo_ids must not appear (segmentation training data).")
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260921)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    banned: set[str] = set()
    for path in args.exclude:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                banned.add(str(json.loads(line)["photo_id"]))
    print(f"excluding {len(banned)} photo_ids seen by the segmentation model")

    rows = []
    for md in sorted(args.species_root.glob("*/metadata.jsonl")):
        for line in open(md, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r["photo_id"]) in banned:
                continue
            # Fold subspecies into the parent binomial.
            parts = str(r["taxon_name"]).split()
            r["label"] = " ".join(parts[:2])
            r["image"] = str(md.parent / "images" / f"inat_{r['photo_id']}.jpg")
            rows.append(r)

    labels = sorted({r["label"] for r in rows})
    print(f"{len(rows)} images, {len(labels)} classes, {len({r['observer_login'] for r in rows})} observers")

    # Observer assignment, whole observers only. Largest first, and each goes to
    # whichever side leaves the per-class validation counts closest to target.
    # A pure fill-greedily rule overshoots: one large observer can push a small
    # class far past its quota and starve its training set.
    by_observer: dict[str, list] = defaultdict(list)
    for r in rows:
        by_observer[r["observer_login"]].append(r)
    per_class = Counter(r["label"] for r in rows)
    target_val = {lab: args.val_fraction * per_class[lab] for lab in labels}
    val_have: Counter[str] = Counter()

    order = sorted(by_observer, key=lambda o: (-len(by_observer[o]), o))
    rng = Random(args.seed)

    def cost(counts: Counter) -> float:
        return sum(abs(counts[lab] - target_val[lab]) for lab in labels)

    assign: dict[str, str] = {}
    for obs in order:
        group = by_observer[obs]
        contribution = Counter(r["label"] for r in group)
        to_val = val_have + contribution
        # Tie-break randomly so the split is not an artefact of observer name order.
        cost_val, cost_train = cost(to_val), cost(val_have)
        if cost_val < cost_train or (cost_val == cost_train and rng.random() < 0.5):
            assign[obs] = "validation"
            val_have = to_val
        else:
            assign[obs] = "train"

    for r in rows:
        r["split"] = assign[r["observer_login"]]

    leak = {o for o in by_observer if len({r["split"] for r in by_observer[o]}) > 1}
    if leak:
        raise SystemExit(f"observer leakage: {sorted(leak)[:5]}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "species-records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (args.output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'class':<30}{'train':>7}{'val':>7}{'val %':>8}")
    for lab in labels:
        t = sum(1 for r in rows if r["label"] == lab and r["split"] == "train")
        v = sum(1 for r in rows if r["label"] == lab and r["split"] == "validation")
        print(f"  {lab:<28}{t:>7}{v:>7}{v/(t+v)*100:>7.1f}%")
    nt = sum(1 for r in rows if r["split"] == "train")
    nv = len(rows) - nt
    print(f"  {'TOTAL':<28}{nt:>7}{nv:>7}{nv/len(rows)*100:>7.1f}%")
    print(f"\nobservers: {sum(1 for o in assign.values() if o=='train')} train, "
          f"{sum(1 for o in assign.values() if o=='validation')} validation, zero shared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
