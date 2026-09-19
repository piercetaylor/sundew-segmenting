"""Add morphology priors to an existing curated manifest without rebuilding images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sundew_segmentation.growth_forms import growth_form_for_taxon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/curated/metadata.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/curated/summary.json"))
    args = parser.parse_args()

    rows = []
    counts: Counter[str] = Counter()
    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["growth_form"] = growth_form_for_taxon(str(row["taxon_name"]))
        counts[row["growth_form"]] += 1
        rows.append(row)
    unknown = [row["taxon_name"] for row in rows if row["growth_form"] == "unknown"]
    if unknown:
        raise SystemExit(f"Missing growth-form mappings: {sorted(set(unknown))}")
    args.metadata.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    if args.summary.is_file():
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        summary["growth_form_counts"] = dict(sorted(counts.items()))
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"updated": len(rows), "growth_form_counts": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
