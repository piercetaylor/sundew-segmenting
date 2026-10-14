"""Acquire the open-set test photos: plants the species model was never taught.

Pre-registered in docs/reports/species-open-set-prereg.md (written 2026-09-30,
before any pull). Sources, research grade, licences CC0 / CC BY / CC BY-NC:

  d   Drosera identified to species rank, not one of the 110       300 photos
  e1  Pinguicula, Byblis, Roridula, Drosophyllum, Dionaea,
      Utricularia (as evenly as their pools allow)                  200 photos
  e2  any other plant (Plantae without the carnivorous families)    200 photos

Sampling: observation ids below a ceiling fetched once and recorded. A query
with at most --full-below matching observations is paged through completely;
a larger one is sampled with random windows (id_above drawn uniformly below
the ceiling, seeded), 200 observations per window. The pool is shuffled with
the seed and filled under the caps (3 photos per observer per source; 30 per
species in d).

Exclusions, applied in order and counted: every observer in any split or the
transfer set; every observation and photo id in them (and the segmentation
manifests); an observation chosen for another source. After download,
--dedup drops sha256 matches and dHash distance <= 6 against every split, the
transfer set and earlier photos in this set.

Modes:
  (default)   plan: metadata only. Writes pool/<source>*.jsonl (cached),
              selection.jsonl and open-set-plan.{json,md}.
  --download  download the selection into <source>/images.
  --dedup     writes open-set-records.jsonl and open-set-dedup.json.
Coordinates are neither requested nor stored.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from random import Random
from urllib.parse import urlencode

import numpy as np

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

TAXA_API = "https://api.inaturalist.org/v1/taxa"
PLANTAE = 47126  # names are matched within Plantae: "Byblis" is also an animal genus
LOOKALIKE_GENERA = ("Pinguicula", "Byblis", "Roridula", "Drosophyllum", "Dionaea", "Utricularia")
CARNIVOROUS_FAMILIES = ("Droseraceae", "Lentibulariaceae", "Byblidaceae", "Roridulaceae", "Drosophyllaceae",
                        "Sarraceniaceae", "Nepenthaceae", "Cephalotaceae")
TARGETS = {"d": 300, "e1": 200, "e2": 200}
SOURCES = ("d", "e1", "e2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", type=Path, required=True, help="sundew-species-corpus (acquisition.json)")
    p.add_argument("--records", type=Path, required=True, help="split records: every split")
    p.add_argument("--transfer", type=Path, required=True, help="transfer-records.jsonl of the transfer set")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--licenses", default="cc0,cc-by,cc-by-nc")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--max-per-observer", type=int, default=3)
    p.add_argument("--max-per-species", type=int, default=30, help="Source d only.")
    p.add_argument("--full-below", type=int, default=1000,
                   help="Page through a query completely when it matches at most this many observations.")
    p.add_argument("--windows", type=int, default=60, help="Random windows per sampled query.")
    p.add_argument("--delay", type=float, default=1.05, help="Seconds between API requests.")
    p.add_argument("--download-delay", type=float, default=0.25)
    p.add_argument("--dhash-max-distance", type=int, default=6)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--dedup", action="store_true")
    return p.parse_args()


def taxon_id(name: str, rank: str, delay: float) -> int:
    res = fetch_json(f"{TAXA_API}?{urlencode({'q': name, 'rank': rank, 'per_page': '30'})}").get("results") or []
    time.sleep(delay)
    hits = [r for r in res if r.get("name") == name and r.get("rank") == rank
            and (r.get("id") == PLANTAE or PLANTAE in (r.get("ancestor_ids") or []))]
    if len(hits) != 1:
        raise SystemExit(f"{rank} {name!r}: expected one exact iNaturalist match, got {len(hits)}")
    return int(hits[0]["id"])


def queries(args, species_ids: list[int], meta: dict) -> dict[str, dict]:
    """Per pool: the API query (without paging params). Taxon ids are resolved once and recorded."""
    if "taxa" not in meta:
        d = args.delay
        meta["taxa"] = {
            "Drosera": taxon_id("Drosera", "genus", d),
            "Plantae": taxon_id("Plantae", "kingdom", d),
            **{g: taxon_id(g, "genus", d) for g in LOOKALIKE_GENERA},
            **{f: taxon_id(f, "family", d) for f in CARNIVOROUS_FAMILIES},
        }
    t = meta["taxa"]
    base = {"photos": "true", "photo_license": args.licenses, "quality_grade": "research"}
    q = {"d": {**base, "taxon_id": str(t["Drosera"]), "rank": "species",
               "without_taxon_id": ",".join(map(str, sorted(species_ids)))}}
    for g in LOOKALIKE_GENERA:
        q[f"e1-{g}"] = {**base, "taxon_id": str(t[g])}
    q["e2"] = {**base, "taxon_id": str(t["Plantae"]),
               "without_taxon_id": ",".join(str(t[f]) for f in CARNIVOROUS_FAMILIES)}
    return q


def fetch_page(query: dict, above: int, below: int, delay: float, per_page: int = 200) -> dict:
    params = {**query, "order_by": "id", "order": "asc", "id_above": str(above), "id_below": str(below),
              "per_page": str(per_page)}
    payload = fetch_json(f"{INAT_API}?{urlencode(params)}")
    time.sleep(delay)
    return payload


def pull_pool(name: str, query: dict, args, allowed, ceiling: int) -> tuple[dict, list[Candidate]]:
    """Every observation (small query) or random windows (large query) below the ceiling."""
    total = int(fetch_page(query, 0, ceiling, args.delay, per_page=0).get("total_results") or 0)
    obs: dict[int, dict] = {}
    if total <= args.full_below:
        above, method = 0, "all"
        while True:
            results = fetch_page(query, above, ceiling, args.delay).get("results") or []
            obs.update((int(r["id"]), r) for r in results)
            if len(results) < 200:
                break
            above = max(int(r["id"]) for r in results)
    else:
        method = f"{args.windows} random windows"
        rng = Random(f"{args.seed}-{name}-windows")
        for _ in range(args.windows):
            results = fetch_page(query, rng.randrange(ceiling), ceiling, args.delay).get("results") or []
            obs.update((int(r["id"]), r) for r in results)
    cands = extract_candidates([obs[k] for k in sorted(obs)], allowed_licenses=allowed,
                               quality_grades=("research",), allow_captive=False)
    return {"matching": total, "method": method, "observations": len(obs), "candidates": len(cands)}, cands


def select(pool: list[Candidate], n: int, args, key: str, excluded, chosen_obs: set[int], stats: Counter,
           species_cap: int | None = None) -> list[Candidate]:
    observers, photos, observations = excluded
    kept = []
    for c in pool:
        if c.observer_login in observers:
            stats["excluded_observer"] += 1
        elif c.observation_id in observations:
            stats["split_or_transfer_observation"] += 1
        elif c.photo_id in photos:
            stats["excluded_photo"] += 1
        elif c.observation_id in chosen_obs:
            stats["other_source"] += 1
        else:
            kept.append(c)
    stats["eligible"] += len(kept)
    Random(f"{args.seed}-{key}").shuffle(kept)
    per_observer, per_species, out = Counter(), Counter(), []
    for c in kept:
        if len(out) >= n:
            stats["cap"] += 1
        elif per_observer[c.observer_login] >= args.max_per_observer:
            stats["observer_cap"] += 1
        elif species_cap is not None and per_species[c.taxon_name] >= species_cap:
            stats["species_cap"] += 1
        else:
            per_observer[c.observer_login] += 1
            per_species[c.taxon_name] += 1
            chosen_obs.add(c.observation_id)
            out.append(c)
    stats["selected"] += len(out)
    return out


def load_exclusions(args):
    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    rows += read_manifest(args.transfer)
    observers = {r["observer_login"] for r in rows}
    observations = {int(r["observation_id"]) for r in rows}
    photos = {int(r["photo_id"]) for r in rows}
    acquisition = json.loads((args.corpus / "acquisition.json").read_text())
    for manifest in acquisition.get("exclude_manifests", []):
        photos |= {int(r["photo_id"]) for r in read_manifest(Path(manifest)) if isinstance(r.get("photo_id"), int)}
    return (observers, photos, observations), rows


def plan(args, allowed) -> int:
    acq = json.loads((args.corpus / "acquisition.json").read_text())["species"]
    species_ids = sorted(int(v["taxon_id"]) for v in acq.values())
    if len(species_ids) != 110:
        raise SystemExit(f"expected the corpus's 110 species, found {len(species_ids)}")
    excluded, rows = load_exclusions(args)
    meta_path = args.output / "pull.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "id_below" not in meta:
        ceiling = fetch_json(f"{INAT_API}?{urlencode({'order_by': 'id', 'order': 'desc', 'per_page': '1'})}")
        meta.update(id_below=int(ceiling["results"][0]["id"]) + 1, queried_at=utc_now())
    q = queries(args, species_ids, meta)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ceiling {meta['id_below']} (queried {meta['queried_at']}); excluding {len(excluded[0])} observers, "
          f"{len(excluded[2])} observations, {len(excluded[1])} photo ids", flush=True)

    pools, pool_stats = {}, {}
    (args.output / "pool").mkdir(parents=True, exist_ok=True)
    for name, query in q.items():
        path = args.output / "pool" / f"{name}.jsonl"
        if path.exists():
            lines = read_manifest(path)
            pool_stats[name], pools[name] = lines[0], [Candidate(**r) for r in lines[1:]]
        else:
            pool_stats[name], pools[name] = pull_pool(name, query, args, allowed, meta["id_below"])
            with path.open("w", encoding="utf-8") as h:
                h.write(json.dumps(pool_stats[name]) + "\n")
                for c in pools[name]:
                    h.write(json.dumps(asdict(c), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"  pool {name:<18} {json.dumps(pool_stats[name])}", flush=True)

    chosen_obs: set[int] = set()
    selection, stats = [], {s: Counter() for s in SOURCES}
    for c in select(pools["d"], TARGETS["d"], args, "d", excluded, chosen_obs, stats["d"], args.max_per_species):
        selection.append({**asdict(c), "source": "d", "group": c.taxon_name})
    # e1: split the target as evenly as the genus pools allow, smallest pools first.
    remaining, genera = TARGETS["e1"], sorted(LOOKALIKE_GENERA, key=lambda g: len(pools[f"e1-{g}"]))
    for i, g in enumerate(genera):
        share = math.ceil(remaining / (len(genera) - i))
        got = select(pools[f"e1-{g}"], share, args, f"e1-{g}", excluded, chosen_obs, stats["e1"])
        remaining -= len(got)
        selection.extend({**asdict(c), "source": "e1", "group": g} for c in got)
    for c in select(pools["e2"], TARGETS["e2"], args, "e2", excluded, chosen_obs, stats["e2"]):
        selection.append({**asdict(c), "source": "e2", "group": c.taxon_name})

    with (args.output / "selection.jsonl").open("w", encoding="utf-8") as h:
        for r in selection:
            h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    by_source = Counter(r["source"] for r in selection)
    report = {"pull": meta, "seed": args.seed, "targets": TARGETS, "max_per_observer": args.max_per_observer,
              "max_per_species_d": args.max_per_species, "licenses": args.licenses.split(","),
              "queries": q, "pools": pool_stats, "selection": {s: dict(stats[s]) for s in SOURCES},
              "selected": dict(by_source),
              "groups": {s: dict(Counter(r["group"] for r in selection if r["source"] == s)) for s in SOURCES}}
    (args.output / "open-set-plan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("selected " + ", ".join(f"{s}: {by_source[s]}/{TARGETS[s]}" for s in SOURCES))
    return 0


def download(args) -> int:
    rows = read_manifest(args.output / "selection.jsonl")
    if not rows:
        raise SystemExit("no selection.jsonl: run the plan first")
    for s in SOURCES:
        sel = [r for r in rows if r["source"] == s]
        cands = [Candidate(**{k: v for k, v in r.items() if k not in ("source", "group")}) for r in sel]
        st = download_candidates(cands, args.output / s, target_count=len(cands), delay_seconds=args.download_delay)
        print(f"{s}: selected {len(cands)} accepted {st['accepted']}", flush=True)
    return 0


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def dedup(args) -> int:
    _, ref_rows = load_exclusions(args)
    info = {int(r["photo_id"]): r for r in read_manifest(args.output / "selection.jsonl")}
    rows = []
    for s in SOURCES:
        if (args.output / s / "metadata.jsonl").exists():
            rows.extend(read_manifest(args.output / s / "metadata.jsonl"))
    rows.sort(key=lambda r: int(r["photo_id"]))
    ref_sha = {r["sha256"] for r in ref_rows if r.get("sha256")}
    ref_hash = np.array([int(r["dhash64"], 16) for r in ref_rows if r.get("dhash64")], dtype=np.uint64)
    kept, kept_hash, reasons = [], [], Counter()
    for r in rows:
        h = np.uint64(int(r["dhash64"], 16))
        if r["sha256"] in ref_sha:
            reasons["sha256_split_or_transfer"] += 1
        elif popcount64(ref_hash ^ h).min() <= args.dhash_max_distance:
            reasons["dhash_split_or_transfer"] += 1
        elif kept_hash and popcount64(np.array(kept_hash, dtype=np.uint64) ^ h).min() <= args.dhash_max_distance:
            reasons["dhash_within"] += 1
        else:
            sel = info[int(r["photo_id"])]
            kept.append({**r, "source": sel["source"], "group": sel["group"]})
            kept_hash.append(int(r["dhash64"], 16))
    with (args.output / "open-set-records.jsonl").open("w", encoding="utf-8") as h:
        for r in kept:
            h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"downloaded": len(rows), "kept": len(kept), "removed": dict(reasons),
               "by_source": dict(Counter(r["source"] for r in kept)), "completed_at": utc_now()}
    (args.output / "open-set-dedup.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    codes = [c.strip() for c in args.licenses.split(",") if c.strip()]
    unknown = [c for c in codes if c not in LICENSE_URLS]
    if unknown:
        raise SystemExit(f"licences not permitted by policy: {unknown}")
    allowed = {c: LICENSE_URLS[c] for c in codes}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.download:
        return download(args)
    if args.dedup:
        return dedup(args)
    return plan(args, allowed)


if __name__ == "__main__":
    raise SystemExit(main())
