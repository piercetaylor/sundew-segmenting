"""Audit raw and curated dataset manifests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from sundew_segmentation.acquisition import read_manifest


def summarize(rows: list[dict]) -> dict:
    hashes = Counter(str(row.get("sha256")) for row in rows)
    dhashes = Counter(str(row.get("dhash64")) for row in rows)
    return {
        "images": len(rows),
        "licenses": dict(Counter(str(row.get("license_code")) for row in rows)),
        "species": len({str(row.get("taxon_name")) for row in rows}),
        "observers": len({str(row.get("observer_login")) for row in rows}),
        "source_total_bytes": sum(int(row.get("bytes") or 0) for row in rows),
        "derivative_total_bytes": sum(
            Path(str(row["curated_path"])).stat().st_size
            for row in rows
            if row.get("curated_path") and Path(str(row["curated_path"])).exists()
        ),
        "minimum_width": min((int(row.get("decoded_width") or 0) for row in rows), default=0),
        "minimum_height": min((int(row.get("decoded_height") or 0) for row in rows), default=0),
        "duplicate_sha256_groups": sum(1 for count in hashes.values() if count > 1),
        "duplicate_dhash_groups": sum(1 for count in dhashes.values() if count > 1),
        "missing_source_pages": sum(1 for row in rows if not row.get("source_page")),
        "missing_creators": sum(1 for row in rows if not row.get("creator")),
    }


def main() -> None:
    raw = read_manifest(Path("data/raw/inaturalist/metadata.jsonl"))
    curated = read_manifest(Path("data/curated/metadata.jsonl"))
    report = {"raw": summarize(raw), "curated": summarize(curated)}
    output = Path("data/reports/dataset_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
