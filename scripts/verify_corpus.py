"""Check an acquired corpus against its own manifests before anything trains on it.

Cheap, and it catches the failure modes that are silent otherwise: a manifest
row whose image never landed, a photograph acquired under a licence the policy
in docs/licence-policy.md does not permit, an attribution field left empty on a
licence that requires one, a duplicate that slipped past the per-species dedup
because it appeared under two species, or a coordinate field that should never
have been stored.

    python scripts/verify_corpus.py --root ../sundew-species-corpus
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sundew_segmentation.acquisition import LICENSE_URLS  # noqa: E402

COORDINATE_HINTS = ("lat", "lon", "coord", "geoprivacy", "place")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--licenses", default="cc0,cc-by,cc-by-nc")
    p.add_argument("--min-per-species", type=int, default=0,
                   help="Warn about classes thinner than this.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    allowed = {c.strip() for c in args.licenses.split(",") if c.strip()}
    unknown = allowed - set(LICENSE_URLS)
    if unknown:
        raise SystemExit(f"licences not permitted by policy: {sorted(unknown)}")

    problems: list[str] = []
    per_species: dict[str, int] = {}
    licences: collections.Counter[str] = collections.Counter()
    photo_ids: dict[int, str] = {}
    hashes: dict[str, str] = {}
    total = 0

    for manifest in sorted(args.root.glob("*/metadata.jsonl")):
        species = manifest.parent.name
        rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        per_species[species] = len(rows)
        for row in rows:
            total += 1
            where = f"{species}/{row.get('photo_id')}"

            path = Path(row["local_path"])
            if not path.exists():
                problems.append(f"missing image: {where} -> {path}")

            code = str(row.get("license_code", ""))
            licences[code] += 1
            if code not in allowed:
                problems.append(f"licence outside policy: {where} is {code!r}")
            if code != "cc0" and not str(row.get("attribution", "")).strip():
                problems.append(f"empty attribution on {code}: {where}")
            if row.get("license_url") != LICENSE_URLS.get(code):
                problems.append(f"licence url does not match code: {where}")

            leaked = [k for k in row if any(h in k.lower() for h in COORDINATE_HINTS)]
            if leaked:
                problems.append(f"location field stored: {where} has {leaked}")

            pid = row.get("photo_id")
            if pid in photo_ids and photo_ids[pid] != species:
                problems.append(f"photo {pid} appears in both {photo_ids[pid]} and {species}")
            photo_ids[pid] = species
            digest = row.get("sha256")
            if digest in hashes and hashes[digest] != where:
                problems.append(f"identical bytes in {hashes[digest]} and {where}")
            hashes[digest] = where

    orphans = sum(1 for p in args.root.glob("*/images/*.jpg")
                  if int(p.stem.removeprefix("inat_")) not in photo_ids)

    print(f"root            {args.root}")
    print(f"species         {len(per_species)}")
    print(f"images          {total}")
    print(f"unique photos   {len(photo_ids)}")
    print(f"unique bytes    {len(hashes)}")
    print(f"orphan files    {orphans}  (on disk, absent from every manifest)")
    print(f"licences        {dict(licences)}")
    if per_species:
        order = sorted(per_species.items(), key=lambda kv: kv[1])
        print(f"thinnest class  {order[0][0]} ({order[0][1]})")
        print(f"largest class   {order[-1][0]} ({order[-1][1]})")
        if args.min_per_species:
            thin = [f"{k} ({v})" for k, v in order if v < args.min_per_species]
            if thin:
                print(f"below {args.min_per_species}: {', '.join(thin)}")

    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for line in problems[:40]:
            print(f"  {line}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("\nno problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
