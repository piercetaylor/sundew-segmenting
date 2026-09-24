"""Acquire a per-species image set for the crop-vs-full-frame classifier test.

scripts/acquire_inaturalist.py draws one random pool across all Drosera and caps
species, which is right for a diverse segmentation set and wrong here: this
needs a few hundred images each of the best-represented species. It queries per
taxon instead, reusing the same licence filters, download validation, provenance
records and duplicate checks.

Observer caps are enforced per species, and the resulting manifest carries
observer_login so the classifier split can be grouped by observer exactly as the
segmentation splits are.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sundew_segmentation.acquisition import (  # noqa: E402
    INAT_API,
    Candidate,
    download_candidates,
    extract_candidates,
    fetch_json,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a per-species Drosera set.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--species", nargs="+", required=True, help="Taxon names, e.g. 'Drosera rotundifolia'")
    p.add_argument("--per-species", type=int, default=200)
    p.add_argument("--max-per-observer", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--delay", type=float, default=1.05, help="Seconds between API pages.")
    return p.parse_args()


def taxon_id_for(name: str) -> int | None:
    url = "https://api.inaturalist.org/v1/taxa?" + urlencode({"q": name, "rank": "species", "per_page": "5"})
    for row in fetch_json(url).get("results") or []:
        if str(row.get("name", "")).lower() == name.lower():
            return int(row["id"])
    return None


def pool_for_taxon(taxon_id: int, pages: int, delay: float) -> list[Candidate]:
    out: list[Candidate] = []
    for page in range(1, pages + 1):
        params = {
            "taxon_id": str(taxon_id), "photos": "true", "quality_grade": "research",
            "captive": "false", "photo_license": "cc0,cc-by", "order_by": "random",
            "per_page": "200", "page": str(page),
        }
        payload = fetch_json(f"{INAT_API}?{urlencode(params)}")
        got = extract_candidates(payload.get("results") or [])
        out.extend(got)
        if not got:
            break
        if page < pages:
            time.sleep(delay)
    return out


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = Random(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for name in args.species:
        tid = taxon_id_for(name)
        if tid is None:
            print(f"!! no exact taxon match for {name!r}; skipping", flush=True)
            continue
        time.sleep(args.delay)
        # Over-fetch: observer caps and download failures both shrink the yield.
        pages = max(2, (args.per_species * 4) // 200 + 1)
        pool = pool_for_taxon(tid, pages, args.delay)
        rng.shuffle(pool)

        per_observer: Counter[str] = Counter()
        chosen: list[Candidate] = []
        for c in pool:
            if len(chosen) >= args.per_species:
                break
            if per_observer[c.observer_login] >= args.max_per_observer:
                continue
            chosen.append(c)
            per_observer[c.observer_login] += 1

        out_dir = args.output / name.replace(" ", "_")
        stats = download_candidates(chosen, out_dir, target_count=args.per_species)
        summary[name] = {
            "taxon_id": tid, "pool": len(pool), "selected": len(chosen),
            "observers": len(per_observer), **stats,
        }
        print(f"{name:<32} pool {len(pool):>5}  selected {len(chosen):>4}  "
              f"accepted {stats.get('accepted', 0):>4}  observers {len(per_observer):>4}", flush=True)
        time.sleep(args.delay)

    (args.output / "acquisition.json").write_text(
        json.dumps({"completed_at": utc_now(), "seed": args.seed,
                    "per_species": args.per_species,
                    "max_per_observer": args.max_per_observer,
                    "filters": {"quality_grade": "research", "captive": False,
                                "photo_license": ["cc0", "cc-by"]},
                    "coordinate_fields_acquired": False,
                    "species": summary}, indent=2) + "\n", encoding="utf-8")
    total = sum(v.get("accepted", 0) for v in summary.values())
    print(f"\ntotal accepted: {total} across {len(summary)} species")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
