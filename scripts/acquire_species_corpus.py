"""Acquire the full species corpus for the crop-then-classify experiment.

`acquire_species_set.py` built the ten-species, CC0/CC-BY set that answered
whether the crop is worth anything at all (`reports/species-crop-comparison.md`).
It is left alone so that result stays reproducible. This is the scaled version
and differs in four ways:

  * **The class list is derived, not typed.** Species are selected by a floor on
    available observations, queried at run time and recorded, so the corpus
    definition is a rule rather than a list someone pasted.
  * **NonCommercial photographs are included.** See `docs/licence-policy.md`.
    ShareAlike and NoDerivatives are not. The wider set is passed explicitly
    into `extract_candidates`; nothing widens by default.
  * **The selection is persisted before anything is downloaded.** iNaturalist's
    `order_by=random` is not stable across calls, so a job that resumes after
    hitting its walltime would otherwise draw a different sample and the seed
    would not reproduce the corpus. `selection.jsonl` is written first and
    replayed on resume. Resume is manifest-driven: a row in `metadata.jsonl` is
    taken to mean the file landed, which holds because the row is appended only
    after the download is renamed into place. Deleting images while leaving the
    manifest intact will not restore them; `scripts/verify_corpus.py` detects
    that case.
  * **Images the segmentation model has seen are excluded.** A crop taken from
    an image the segmenter trained on is unrealistically good, and that
    advantage is precisely what the downstream experiment measures.

Coordinates are neither requested nor stored, as everywhere else in this
project.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from random import Random
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sundew_segmentation.acquisition import (  # noqa: E402
    INAT_API,
    LICENSE_URLS,
    Candidate,
    download_candidates,
    extract_candidates,
    fetch_json,
    read_manifest,
    utc_now,
)

SPECIES_COUNTS_API = "https://api.inaturalist.org/v1/observations/species_counts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the scaled per-species Drosera corpus.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-observations", type=int, default=50,
                   help="Species floor: skip species with fewer available observations.")
    p.add_argument("--per-species", type=int, default=300, help="Per-species cap.")
    p.add_argument("--max-per-observer", type=int, default=5)
    p.add_argument("--licenses", default="cc0,cc-by,cc-by-nc",
                   help="Comma-separated photo licences; must all be known to LICENSE_URLS.")
    p.add_argument("--exclude-manifest", type=Path, nargs="*", default=(),
                   help="JSONL files whose photo_id values must not be acquired.")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--delay", type=float, default=1.05, help="Seconds between API pages.")
    p.add_argument("--download-delay", type=float, default=0.25,
                   help="Seconds between image downloads; keeps the request rate polite.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve the species list and print the plan without downloading.")
    return p.parse_args()


def species_floor(licenses: str, floor: int, delay: float) -> list[tuple[str, int, int]]:
    """Return (name, taxon_id, available) for species at or above the floor."""
    rows, page = [], 1
    while True:
        params = {
            "taxon_id": "51935", "photos": "true", "quality_grade": "research",
            "captive": "false", "photo_license": licenses,
            "per_page": "500", "page": str(page),
        }
        payload = fetch_json(f"{SPECIES_COUNTS_API}?{urlencode(params)}")
        rows.extend(payload.get("results") or [])
        if len(rows) >= int(payload.get("total_results", 0)):
            break
        page += 1
        time.sleep(delay)

    out = []
    for row in rows:
        taxon = row.get("taxon") or {}
        count = int(row.get("count", 0))
        # rank guards against genus- or section-level rows joining the class list
        if count >= floor and taxon.get("rank") == "species" and isinstance(taxon.get("id"), int):
            out.append((str(taxon["name"]), int(taxon["id"]), count))
    out.sort(key=lambda item: (-item[2], item[0]))
    return out


def pool_for_taxon(taxon_id: int, licenses: str, allowed, pages: int, delay: float) -> list[Candidate]:
    out: list[Candidate] = []
    for page in range(1, pages + 1):
        params = {
            "taxon_id": str(taxon_id), "photos": "true", "quality_grade": "research",
            "captive": "false", "photo_license": licenses, "order_by": "random",
            "per_page": "200", "page": str(page),
        }
        got = extract_candidates(
            fetch_json(f"{INAT_API}?{urlencode(params)}").get("results") or [],
            allowed_licenses=allowed,
        )
        out.extend(got)
        if not got:
            break
        if page < pages:
            time.sleep(delay)
    return out


def excluded_photo_ids(paths) -> set[int]:
    seen: set[int] = set()
    for path in paths:
        for row in read_manifest(Path(path)):
            if isinstance(row.get("photo_id"), int):
                seen.add(int(row["photo_id"]))
    return seen


def main() -> int:
    args = parse_args()
    codes = [c.strip() for c in args.licenses.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LICENSE_URLS]
    if unknown:
        raise SystemExit(f"licences not permitted by policy: {unknown}")
    allowed = {c: LICENSE_URLS[c] for c in codes}

    args.output.mkdir(parents=True, exist_ok=True)
    excluded = excluded_photo_ids(args.exclude_manifest)
    print(f"excluding {len(excluded)} photo ids the segmentation model has seen", flush=True)

    species = species_floor(args.licenses, args.min_observations, args.delay)
    planned = sum(min(count, args.per_species) for _, _, count in species)
    print(f"{len(species)} species at or above {args.min_observations} observations; "
          f"at most {planned} images before observer capping\n", flush=True)
    if args.dry_run:
        for name, tid, count in species:
            print(f"  {name:<32}{tid:>8}{count:>8}  -> {min(count, args.per_species)}")
        return 0

    summary: dict[str, dict] = {}
    for index, (name, tid, available) in enumerate(species, start=1):
        out_dir = args.output / name.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        selection_path = out_dir / "selection.jsonl"

        if selection_path.exists():
            # Resume: replay the recorded sample rather than redrawing it, since
            # order_by=random is not stable between calls.
            chosen = [Candidate(**row) for row in read_manifest(selection_path)]
            pool_size = -1
        else:
            # Over-fetch: observer caps and download failures both shrink the yield.
            pages = max(2, (args.per_species * 4) // 200 + 1)
            pool = pool_for_taxon(tid, args.licenses, allowed, pages, args.delay)
            pool = [c for c in pool if c.photo_id not in excluded]
            # Seeded per taxon rather than from one stream, so that resuming --
            # which skips species that already hold a selection -- does not
            # shift the draw for the species that follow.
            Random(args.seed + tid).shuffle(pool)
            per_observer: Counter[str] = Counter()
            chosen = []
            for candidate in pool:
                if len(chosen) >= args.per_species:
                    break
                if per_observer[candidate.observer_login] >= args.max_per_observer:
                    continue
                chosen.append(candidate)
                per_observer[candidate.observer_login] += 1
            pool_size = len(pool)
            with selection_path.open("w", encoding="utf-8") as handle:
                for candidate in chosen:
                    handle.write(json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True) + "\n")
            time.sleep(args.delay)

        stats = download_candidates(
            chosen, out_dir, target_count=len(chosen), delay_seconds=args.download_delay
        )
        observers = len({c.observer_login for c in chosen})
        summary[name] = {
            "taxon_id": tid, "available": available, "pool": pool_size,
            "selected": len(chosen), "observers": observers, **stats,
        }
        print(f"[{index:>3}/{len(species)}] {name:<30} avail {available:>5}  "
              f"selected {len(chosen):>4}  accepted {stats.get('accepted', 0):>4}  "
              f"observers {observers:>4}", flush=True)

    (args.output / "acquisition.json").write_text(
        json.dumps({
            "completed_at": utc_now(),
            "seed": args.seed,
            "min_observations": args.min_observations,
            "per_species": args.per_species,
            "max_per_observer": args.max_per_observer,
            "filters": {
                "quality_grade": "research", "captive": False, "photo_license": codes,
            },
            "licence_policy": "docs/licence-policy.md",
            "coordinate_fields_acquired": False,
            "excluded_photo_ids": len(excluded),
            "exclude_manifests": [str(p) for p in args.exclude_manifest],
            "species": summary,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = sum(v.get("accepted", 0) for v in summary.values())
    print(f"\ntotal accepted: {total} across {len(summary)} species")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
