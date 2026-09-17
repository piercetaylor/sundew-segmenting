"""Create Label Studio tasks from the curated provenance manifest."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from sundew_segmentation.acquisition import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/curated/metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/label-studio-tasks.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = []
    for row in read_manifest(args.metadata):
        relative_path = Path(row["curated_path"]).relative_to("data/curated").as_posix()
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={relative_path}",
                "photo_id": row["photo_id"],
                "species": row["taxon_name"],
                "split": row["split"],
                "source_page": row["source_page"],
                "license": row["license_code"],
            },
            "meta": {
                "review_index": row["review_index"],
                "creator": row["creator"],
                "annotation_version": "v1.0",
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(tasks)} tasks to {args.output}")


if __name__ == "__main__":
    main()
