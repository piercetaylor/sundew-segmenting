"""Acquire the distillation transfer set: unlabeled-for-training photos of the 110 species.

Pre-registered in docs/species-classifier-plan.md, "Transfer set (written
2026-09-27, before any acquisition or run)". Three sources, the 110 species
only, licences CC0 / CC BY / CC BY-NC as for the corpus:

  a  wild, research grade, beyond the corpus caps   quality_grade=research, captive=false
  b  captive / cultivated, any grade                captive=true (iNaturalist makes these casual)
  c  wild, needs-ID or casual, species-level ID     quality_grade=needs_id,casual, captive=false

Other Drosera and other genera are never pulled here: they are the open-set
evaluation set, which is built later and must be observer-disjoint from this one.

Exclusions, applied in this order and counted:
  1. every validation and test observer (any taxon);
  2. every observation already in any split;
  3. every photo id in the corpus or the segmentation manifests;
  4. an observation already chosen from another source.
Then, per species: sources are filled in the order b, c, a, each pool shuffled
with a seed fixed per taxon and source; b and c are capped at 100, the total at
300, and each observer at 5 photos per species across sources. b and c go first
because they are scarce and are the only new plants for the thin species; a is
plentiful and fills the rest.

Paging is by observation id (id_above) below a ceiling fetched once and
recorded, because the API refuses page * per_page beyond 10,000; with a fixed
ceiling and seed the selection reproduces. One request per --delay seconds.

Modes:
  (default)   dry run: metadata only, no images. Writes pool/<species>/<source>.jsonl
              (cached; delete to refetch), selection.jsonl and transfer-plan.{json,md}.
  --download  download the recorded selection into <species>/images, as the corpus does.
  --dedup     after download: drop sha256 duplicates and dHash distance <= 6 against
              every split and within the set; writes transfer-records.jsonl.
Coordinates are neither requested nor stored.
"""
from __future__ import annotations

import argparse
import json
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

SOURCES = {
    "a": {"query": {"quality_grade": "research", "captive": "false"}, "grades": ("research",), "captive": False},
    "b": {"query": {"quality_grade": "research,needs_id,casual", "captive": "true"},
          "grades": ("research", "needs_id", "casual"), "captive": True},
    "c": {"query": {"quality_grade": "needs_id,casual", "captive": "false"},
          "grades": ("needs_id", "casual"), "captive": False},
}
FILL_ORDER = ("b", "c", "a")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", type=Path, required=True, help="sundew-species-corpus (acquisition.json)")
    p.add_argument("--records", type=Path, required=True, help="split records: every split, for exclusions")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-species", type=int, default=300)
    p.add_argument("--source-cap", type=int, default=100, help="Cap per species for sources b and c.")
    p.add_argument("--max-per-observer", type=int, default=5)
    p.add_argument("--licenses", default="cc0,cc-by,cc-by-nc")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--delay", type=float, default=1.05, help="Seconds between API requests.")
    p.add_argument("--download-delay", type=float, default=0.25)
    p.add_argument("--limit-species", type=int, default=None, help="Test run: the first N species only.")
    p.add_argument("--dhash-max-distance", type=int, default=6)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--dedup", action="store_true")
    return p.parse_args()


def id_ceiling() -> int:
    """The newest observation id right now; everything is pulled below it."""
    payload = fetch_json(f"{INAT_API}?{urlencode({'order_by': 'id', 'order': 'desc', 'per_page': '1'})}")
    return int(payload["results"][0]["id"]) + 1


def pull(taxon_id: int, source: str, licenses: str, allowed, ceiling: int, delay: float) -> tuple[int, list[Candidate]]:
    """Every observation of one taxon and source below the ceiling: (observations seen, candidates)."""
    spec = SOURCES[source]
    seen, out, above = 0, [], 0
    while True:
        params = {"taxon_id": str(taxon_id), "photos": "true", "photo_license": licenses,
                  "order_by": "id", "order": "asc", "id_above": str(above), "id_below": str(ceiling),
                  "per_page": "200", **spec["query"]}
        results = fetch_json(f"{INAT_API}?{urlencode(params)}").get("results") or []
        time.sleep(delay)
        if not results:
            break
        seen += len(results)
        above = max(int(r["id"]) for r in results)
        out.extend(extract_candidates(results, allowed_licenses=allowed,
                                      quality_grades=spec["grades"], allow_captive=spec["captive"]))
        if len(results) < 200:
            break
    return seen, out


def dry_run(args, species, allowed) -> int:
    split_rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    held_out_observers = {r["observer_login"] for r in split_rows if r["split"] in ("validation", "test")}
    split_observations = {int(r["observation_id"]) for r in split_rows}
    excluded_photos = {int(r["photo_id"]) for r in split_rows}
    acquisition = json.loads((args.corpus / "acquisition.json").read_text())
    for manifest in acquisition.get("exclude_manifests", []):
        excluded_photos |= {int(r["photo_id"]) for r in read_manifest(Path(manifest)) if isinstance(r.get("photo_id"), int)}

    meta_path = args.output / "pull.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())  # a resumed dry run keeps the first ceiling
    else:
        meta = {"id_below": id_ceiling(), "queried_at": utc_now()}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"observation id ceiling {meta['id_below']} (queried {meta['queried_at']}); "
          f"excluding {len(held_out_observers)} validation/test observers, {len(split_observations)} split "
          f"observations, {len(excluded_photos)} photo ids", flush=True)

    plan, selection = {}, []
    for i, (name, tid) in enumerate(species, 1):
        pool_dir = args.output / "pool" / name.replace(" ", "_")
        pool_dir.mkdir(parents=True, exist_ok=True)
        pools, stats = {}, {}
        for src in FILL_ORDER:
            path = pool_dir / f"{src}.jsonl"
            if path.exists():
                rows = read_manifest(path)
                seen = rows[0]["_seen"] if rows else 0
                cands = [Candidate(**{k: v for k, v in r.items() if k != "_seen"}) for r in rows[1:]]
            else:
                seen, cands = pull(tid, src, args.licenses, allowed, meta["id_below"], args.delay)
                with path.open("w", encoding="utf-8") as h:
                    h.write(json.dumps({"_seen": seen}) + "\n")
                    for c in cands:
                        h.write(json.dumps(asdict(c), ensure_ascii=False, sort_keys=True) + "\n")
            pools[src] = cands
            stats[src] = Counter(observations=seen, no_eligible_photo=seen - len(cands))

        chosen_obs: set[int] = set()
        per_observer: Counter[str] = Counter()
        total = 0
        for src in FILL_ORDER:
            st, cap = stats[src], (args.source_cap if src in ("b", "c") else None)
            kept = []
            for c in pools[src]:
                if c.observer_login in held_out_observers:
                    st["val_test_observer"] += 1
                elif c.observation_id in split_observations:
                    st["split_observation"] += 1
                elif c.photo_id in excluded_photos:
                    st["excluded_photo"] += 1
                elif c.observation_id in chosen_obs:
                    st["other_source"] += 1
                else:
                    kept.append(c)
            st["eligible"] = len(kept)
            Random(f"{args.seed}-{tid}-{src}").shuffle(kept)
            n_src = 0
            for c in kept:
                if total >= args.per_species or (cap is not None and n_src >= cap):
                    st["cap"] += 1
                    continue
                if per_observer[c.observer_login] >= args.max_per_observer:
                    st["observer_cap"] += 1
                    continue
                per_observer[c.observer_login] += 1
                chosen_obs.add(c.observation_id)
                selection.append({**asdict(c), "source": src, "species": name})
                n_src += 1
                total += 1
            st["planned"] = n_src
        plan[name] = {"taxon_id": tid, **{s: dict(stats[s]) for s in FILL_ORDER}, "planned": total}
        print(f"[{i:>3}/{len(species)}] {name:<30} " + "  ".join(
            f"{s}: {stats[s]['observations']:>5} seen {stats[s]['eligible']:>5} eligible {stats[s]['planned']:>3} planned"
            for s in FILL_ORDER) + f"  -> {total}", flush=True)

    with (args.output / "selection.jsonl").open("w", encoding="utf-8") as h:
        for row in selection:
            h.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"pull": meta, "seed": args.seed, "per_species": args.per_species, "source_cap": args.source_cap,
              "max_per_observer": args.max_per_observer, "licenses": args.licenses.split(","),
              "fill_order": list(FILL_ORDER), "sources": {k: v["query"] for k, v in SOURCES.items()},
              "excluded": {"validation_test_observers": len(held_out_observers),
                           "split_observations": len(split_observations), "photo_ids": len(excluded_photos)},
              "planned": len(selection), "species": plan}
    (args.output / "transfer-plan.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_markdown(args, report, split_rows)
    print(f"\nplanned {len(selection)} images across {len(plan)} species -> {args.output}/transfer-plan.md")
    return 0


def write_markdown(args, report, split_rows) -> None:
    train = Counter(r["label"] for r in split_rows if r["split"] == "train")
    tot = defaultdict(Counter)
    for sp in report["species"].values():
        for s in FILL_ORDER:
            tot[s].update(sp[s])
    reasons = ("observations", "no_eligible_photo", "val_test_observer", "split_observation", "excluded_photo",
               "other_source", "eligible", "observer_cap", "cap", "planned")
    lines = ["# Transfer set: dry-run plan", "",
             "Generated by `scripts/acquire_transfer_set.py` (metadata only, no images); do not edit by hand.",
             "Design: `docs/species-classifier-plan.md`, \"Transfer set\".", "",
             f"Observation id ceiling {report['pull']['id_below']}, queried {report['pull']['queried_at']}. "
             f"Seed {report['seed']}. Caps: {report['per_species']} per species, {report['source_cap']} for b and c, "
             f"{report['max_per_observer']} per observer per species. Fill order b, c, a.", "",
             f"**Planned: {report['planned']} images.**", "",
             "## By source", "", "| Source | " + " | ".join(reasons) + " |",
             "| --- | " + " | ".join("---:" for _ in reasons) + " |"]
    for s in FILL_ORDER:
        lines.append(f"| {s} | " + " | ".join(str(tot[s][r]) for r in reasons) + " |")
    bins = [(0, 40, "under 40"), (40, 80, "40-79"), (80, 130, "80-129"), (130, 10**6, "130 and over")]
    lines += ["", "## By training count", "", "| Training images | Species | a | b | c | Planned | Planned per species (min-max) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for lo, hi, label in bins:
        names = [n for n in report["species"] if lo <= train[n] < hi]
        per = [report["species"][n]["planned"] for n in names]
        lines.append(f"| {label} | {len(names)} | " + " | ".join(
            str(sum(report["species"][n][s]["planned"] for n in names)) for s in "abc")
            + f" | {sum(per)} | {min(per) if per else 0}-{max(per) if per else 0} |")
    lines += ["", "## Per species", "", "| Species | Train | a | b | c | Planned |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n, sp in sorted(report["species"].items(), key=lambda kv: train[kv[0]]):
        lines.append(f"| *{n}* | {train[n]} | {sp['a']['planned']} | {sp['b']['planned']} | {sp['c']['planned']} | {sp['planned']} |")
    (args.output / "transfer-plan.md").write_text("\n".join(lines) + "\n")


def download(args) -> int:
    rows = read_manifest(args.output / "selection.jsonl")
    if not rows:
        raise SystemExit("no selection.jsonl: run the dry run first")
    by_species = defaultdict(list)
    for r in rows:
        by_species[r["species"]].append(r)
    for i, (name, sel) in enumerate(sorted(by_species.items()), 1):
        cands = [Candidate(**{k: v for k, v in r.items() if k not in ("source", "species")}) for r in sel]
        stats = download_candidates(cands, args.output / name.replace(" ", "_"), target_count=len(cands),
                                    delay_seconds=args.download_delay)
        print(f"[{i:>3}/{len(by_species)}] {name:<30} selected {len(cands):>4} accepted {stats['accepted']:>4}", flush=True)
    return 0


def popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def dedup(args) -> int:
    split_rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    source_of = {int(r["photo_id"]): r["source"] for r in read_manifest(args.output / "selection.jsonl")}
    rows = []
    for d in sorted(p for p in args.output.iterdir() if (p / "metadata.jsonl").exists()):
        rows.extend(read_manifest(d / "metadata.jsonl"))
    rows.sort(key=lambda r: int(r["photo_id"]))
    split_sha = {r["sha256"] for r in split_rows}
    split_hash = np.array([int(r["dhash64"], 16) for r in split_rows], dtype=np.uint64)
    kept, kept_hash, reasons = [], [], Counter()
    for r in rows:
        h = np.uint64(int(r["dhash64"], 16))
        if r["sha256"] in split_sha:
            reasons["sha256_split"] += 1
        elif popcount64(split_hash ^ h).min() <= args.dhash_max_distance:
            reasons["dhash_split"] += 1
        elif kept_hash and popcount64(np.array(kept_hash, dtype=np.uint64) ^ h).min() <= args.dhash_max_distance:
            reasons["dhash_within"] += 1
        else:
            kept.append({**r, "image": r["local_path"], "split": "transfer", "label": None,
                         "inat_taxon_name": r["taxon_name"], "source": source_of[int(r["photo_id"])]})
            kept_hash.append(int(r["dhash64"], 16))
    with (args.output / "transfer-records.jsonl").open("w", encoding="utf-8") as h:
        for r in kept:
            h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"downloaded": len(rows), "kept": len(kept), "removed": dict(reasons),
               "dhash_max_distance": args.dhash_max_distance, "by_source": dict(Counter(r["source"] for r in kept)),
               "completed_at": utc_now()}
    (args.output / "transfer-dedup.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    acq = json.loads((args.corpus / "acquisition.json").read_text())["species"]
    species = sorted((name, int(v["taxon_id"])) for name, v in acq.items())
    if len(species) != 110:
        raise SystemExit(f"expected the corpus's 110 species, found {len(species)}")
    species = species[:args.limit_species] if args.limit_species else species
    return dry_run(args, species, allowed)


if __name__ == "__main__":
    raise SystemExit(main())
