from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sundew_segmentation.acquisition import (  # noqa: E402
    download_candidates,
    fetch_candidate_pool,
    select_diverse,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download licensed sundew images from iNaturalist.")
    parser.add_argument("--count", type=int, default=500, help="Total successfully downloaded images desired.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/inaturalist"))
    parser.add_argument("--pages", type=int, default=None, help="API pages to inspect; defaults from target count.")
    parser.add_argument("--max-per-species", type=int, default=40)
    parser.add_argument("--max-per-observer", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")

    pages = args.pages or max(5, math.ceil(args.count * 5 / 200))
    pool = fetch_candidate_pool(pages=pages)
    reserve_count = max(args.count, math.ceil(args.count * 1.5))
    selected = select_diverse(
        pool,
        limit=reserve_count,
        max_per_species=args.max_per_species,
        max_per_observer=args.max_per_observer,
        seed=args.seed,
    )
    summary = download_candidates(selected, args.output, target_count=args.count)
    run = {
        "source": "iNaturalist",
        "taxon_id": 51935,
        "filters": {
            "photos": True,
            "quality_grade": "research",
            "captive": False,
            "photo_license": ["cc0", "cc-by"],
        },
        "coordinate_fields_acquired": False,
        "requested_count": args.count,
        "api_pages_inspected": pages,
        "candidate_pool_size": len(pool),
        "selected_with_reserve": len(selected),
        "seed": args.seed,
        "max_per_species": args.max_per_species,
        "max_per_observer": args.max_per_observer,
        "completed_at": utc_now(),
        **summary,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "acquisition.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0 if summary["accepted"] >= args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
