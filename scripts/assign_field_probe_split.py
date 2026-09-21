"""Split the field-probe masks into a training half and a held-out evaluation half.

If all 47 masks go into training, the retrained model's field score is measured
on its own training data and means nothing — telling whether the fix worked
would need a second annotation round. Holding part of the batch back keeps one
honest field number to measure against.

Three rules decide the assignment:

* Images by the same observer stay on the same side. Shared gear, location and
  capture habit would otherwise leak across the boundary and inflate the
  held-out score. `build_curated_dataset.py` groups the curated splits the same
  way.
* `ambiguous` reviews go to training. An evaluation label the annotator was
  unsure of muddies the number it exists to produce; in training a soft label
  costs much less.
* Images the scrape probe named as failures are pinned to evaluation. They are
  the cases the batch was collected to measure, so fixing them on data the model
  trained on would prove nothing.

Everything else is shuffled with a fixed seed, so the assignment reproduces.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Named in reports/scrape-probe-50.md as confident, silent failures of the
# current checkpoint. inat_329881443 was the other one; it came back rejected
# because the plant was dead.
PINNED_TO_EVAL = {686282548}

TRAIN_SPLIT = "field-train"
EVAL_SPLIT = "field-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/field-probe-records.jsonl"))
    parser.add_argument("--masks", type=Path, default=Path("data/curated/masks"))
    parser.add_argument("--source-split", default="field-probe")
    parser.add_argument("--eval-size", type=int, default=17, help="target number of held-out images")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--report", type=Path, default=Path("data/reports/field-probe-split.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no records in {args.manifest}")

    by_observer: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_observer[str(row.get("observer_login") or f"unknown-{row['photo_id']}")].append(row)

    forced_train: list[str] = []
    forced_eval: list[str] = []
    free: list[str] = []
    for observer, group in by_observer.items():
        if any(int(r["photo_id"]) in PINNED_TO_EVAL for r in group):
            forced_eval.append(observer)
        elif any(r.get("quality") == "ambiguous" for r in group):
            # One ambiguous review keeps the whole observer group out of the
            # evaluation half, rather than splitting the group.
            forced_train.append(observer)
        else:
            free.append(observer)

    random.Random(args.seed).shuffle(free)
    eval_observers = list(forced_eval)
    held = sum(len(by_observer[o]) for o in eval_observers)
    for observer in free:
        if held >= args.eval_size:
            break
        eval_observers.append(observer)
        held += len(by_observer[observer])

    eval_set = set(eval_observers)
    assignment: dict[int, str] = {}
    for observer, group in by_observer.items():
        split = EVAL_SPLIT if observer in eval_set else TRAIN_SPLIT
        for row in group:
            assignment[int(row["photo_id"])] = split

    # Move each mask into its split directory, then drop the now-empty source.
    moved = Counter()
    for row in rows:
        split = assignment[int(row["photo_id"])]
        stem = Path(row["curated_path"]).stem
        source = args.masks / args.source_split / f"{stem}.png"
        target_dir = args.masks / split
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{stem}.png"
        if source.exists():
            shutil.move(str(source), str(target))
        elif not target.exists():
            raise SystemExit(f"mask missing for {stem}: neither {source} nor {target}")
        row["split"] = split
        row["mask_path"] = target.as_posix()
        moved[split] += 1

    stale = args.masks / args.source_split
    if stale.exists() and not any(stale.iterdir()):
        stale.rmdir()

    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["split"], r["photo_id"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def summarise(split: str) -> dict:
        group = [r for r in rows if r["split"] == split]
        return {
            "images": len(group),
            "observers": len({r["observer_login"] for r in group}),
            "species": len({r["taxon_name"] for r in group}),
            "by_quality": dict(Counter(str(r["quality"]) for r in group)),
            "by_growth_form": dict(Counter(str(r["growth_form"]) for r in group)),
            "mean_foreground_fraction": round(sum(r["foreground_fraction"] for r in group) / len(group), 6),
            "photo_ids": sorted(int(r["photo_id"]) for r in group),
        }

    report = {
        "seed": args.seed,
        "eval_size_target": args.eval_size,
        "pinned_to_eval": sorted(PINNED_TO_EVAL),
        "grouped_by": "observer_login",
        "ambiguous_policy": "training only",
        TRAIN_SPLIT: summarise(TRAIN_SPLIT),
        EVAL_SPLIT: summarise(EVAL_SPLIT),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")

    overlap = {r["observer_login"] for r in rows if r["split"] == TRAIN_SPLIT} & {
        r["observer_login"] for r in rows if r["split"] == EVAL_SPLIT
    }
    if overlap:
        raise SystemExit(f"observer leak across splits: {sorted(overlap)}")

    for split in (TRAIN_SPLIT, EVAL_SPLIT):
        s = report[split]
        print(f"{split}: {s['images']} images, {s['observers']} observers, "
              f"{s['species']} species, quality {s['by_quality']}")
    print("no observer appears on both sides")


if __name__ == "__main__":
    main()
