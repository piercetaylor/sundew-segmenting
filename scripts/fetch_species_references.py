#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pierce Taylor
"""Fetch the original published description (protologue) of each of the 110 species from IPNI.

The site's species pages cite where each species was first described. This script asks
the International Plant Names Index (https://www.ipni.org, the API behind its search page)
for every label in release/species-v1.0.0/labels.json. It writes data/site/references.json,
which is committed: the site build reads that file and never calls IPNI, and the site
itself makes no third-party requests.

For a name that IPNI records as a new combination (comb. nov., stat. nov., nom. nov.), it
also follows the basionym, so the page can cite the paper that first described the plant
under its earlier name.

Usage (needs network; standard library only):
    python3 scripts/fetch_species_references.py            # writes data/site/references.json
    python3 scripts/fetch_species_references.py --only "Drosera regia"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "release/species-v1.0.0/labels.json"
OUT = ROOT / "data/site/references.json"
API = "https://www.ipni.org/api/1"
UA = "sundew-segmenting site build (https://github.com/piercetaylor/sundew-segmenting)"
PAUSE_S = 0.5  # be gentle with IPNI

# Combinations whose IPNI record has no basionym link, found by searching IPNI for the epithet
# under the parenthetical author (2026-10-17).
BASIONYM_PINS = {
    "Drosera esterhuyseniae": "77274546-1",  # Drosera curviscapa var. esterhuyseniae T.M.Salter
    "Drosera glabripes": "77274726-1",       # Drosera ramentacea var. glabripes Harv. ex Planch.
}
# Works IPNI cites without a year on some records; the year is on IPNI's other records of the same work.
WORK_YEARS = {
    "Sketch Veg. Swan R.": 1839,       # Lindley; IPNI gives 1 Dec 1839 for Drosera gigantea
    "Hist. Pl. Remarq. Bresil": 1824,  # A.St.-Hil.; IPNI gives 1824 for Drosera communis
}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - retry anything transient
            if attempt == 3 or getattr(e, "code", None) == 404:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise AssertionError


def https(url: str | None) -> str | None:
    return url.replace("http://", "https://", 1) if url else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def bhl_scan(openurl: str | None) -> str | None:
    """The scanned page an IPNI BHL OpenURL resolves to, or None if BHL has no single page for it
    (not digitised, or several candidate pages)."""
    if not openurl:
        return None
    req = urllib.request.Request(openurl, headers={"User-Agent": UA})
    try:
        urllib.request.build_opener(_NoRedirect).open(req, timeout=30)
        return None
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location", "") if e.code in (301, 302, 303, 307, 308) else ""
    except Exception as e:  # noqa: BLE001
        print(f"  BHL lookup failed for {openurl}: {e}", file=sys.stderr)
        return None
    m = re.fullmatch(r"https?://www\.biodiversitylibrary\.org/page/(\d+)", loc)
    return f"https://www.biodiversitylibrary.org/page/{m.group(1)}" if m else None


def citation(rec: dict) -> dict:
    """The fields the site shows for one IPNI name record."""
    pub = rec.get("linkedPublication") or {}
    ref = (rec.get("reference") or "").strip()
    year = rec.get("publicationYear")
    if not year:
        m = re.search(r"\b(1[6-9]\d\d|20\d\d)\b", ref)
        year = int(m.group(1)) if m else WORK_YEARS.get(rec.get("publication"))
    return {
        "ipni_id": rec["id"],
        "name": rec["name"],
        "authors": rec.get("authors"),
        "citation_type": rec.get("citationType"),
        "reference": (rec.get("reference") or "").strip() or None,
        "publication": rec.get("publication"),
        "publication_title": pub.get("title"),
        "collation": rec.get("referenceCollation"),
        "year": year,
        "distribution": rec.get("distribution"),
        "bhl_page": https(rec.get("bhlLink")),
        "bhl_scan": bhl_scan(https(rec.get("bhlLink"))),
        "bhl_title": https(pub.get("bhlTitleLink")),
        "wfo_id": rec.get("wfoId"),
        "ipni_url": f"https://www.ipni.org/n/{rec['id']}",
        "powo_url": f"https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:{rec['id']}" if rec.get("inPowo") else None,
    }


def species_record(name: str) -> dict | None:
    q = urllib.parse.urlencode({"q": name, "perPage": 50})
    hits = [r for r in get(f"{API}/search?{q}")["results"]
            if r.get("name") == name and r.get("rank") == "spec." and not r.get("suppressed")]
    if not hits:
        return None
    # Prefer IPNI's "top copy" of a name; among those the earliest publication.
    hits.sort(key=lambda r: (not r.get("topCopy"), r.get("publicationYear") or 9999))
    return hits[0]


def wikipedia(name: str) -> str | None:
    """The English Wikipedia article titled with exactly this name, if any; only the link is kept."""
    title = urllib.parse.quote(name.replace(" ", "_"))
    try:
        d = get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
    except Exception:  # noqa: BLE001 - 404: no article
        return None
    # A redirect to another species (Wikipedia sinks some of these names) would mislabel the page.
    if d.get("type") != "standard" or d.get("title", "").replace("_", " ") != name:
        return None
    return d.get("content_urls", {}).get("desktop", {}).get("page")


def fetch(name: str) -> dict:
    rec = species_record(name)
    if rec is None:
        return {"name": name, "found": False}
    time.sleep(PAUSE_S)
    detail = get(f"{API}/n/{rec['id']}")
    out = {"name": name, "found": True, **citation(detail), "wikipedia": wikipedia(name)}
    # A combination's basionym is where the plant was first described.
    bas = detail.get("basionym")
    if isinstance(bas, list):
        bas = bas[0] if bas else None
    bas_id = BASIONYM_PINS.get(name) or (bas or {}).get("id")
    if bas_id:
        time.sleep(PAUSE_S)
        out["basionym"] = citation(get(f"{API}/n/{bas_id}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", help="fetch just this name (repeatable); merges into the existing file")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    labels = json.loads(LABELS.read_text())
    names = args.only or labels
    existing = json.loads(args.out.read_text())["species"] if args.only and args.out.exists() else {}
    for i, n in enumerate(names, 1):
        existing[n] = fetch(n)
        r = existing[n]
        print(f"[{i}/{len(names)}] {n}: " + (f"{r['authors']}, {r['reference']}" + (" (basionym " + r['basionym']['name'] + ")" if 'basionym' in r else "") if r["found"] else "NOT FOUND"))
        time.sleep(PAUSE_S)

    doc = {
        "source": "International Plant Names Index (IPNI), https://www.ipni.org, via its public API",
        "fetched": dt.date.today().isoformat(),
        "note": "Original publication (protologue) of each species name; for combinations, also the basionym. "
                "Bibliographic facts only. Review entries flagged found=false by hand. "
                "wikipedia: the English Wikipedia article if one exists (link only; no text is copied). "
                "bhl_scan: the Biodiversity Heritage Library page that IPNI's OpenURL (bhl_page) resolves to, "
                "when it resolves to exactly one scanned page.",
        "species": {n: existing[n] for n in labels if n in existing},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    missing = [n for n, r in doc["species"].items() if not r["found"]]
    print(f"wrote {args.out} ({len(doc['species'])} species, {len(missing)} not found: {missing})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
