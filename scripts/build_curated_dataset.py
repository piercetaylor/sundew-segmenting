"""Apply the visual screen and build a 250-image annotation-ready dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import json

from sundew_segmentation.acquisition import read_manifest
from sundew_segmentation.curation import normalize_image, select_core_set, split_for_observer, write_jsonl
from sundew_segmentation.growth_forms import growth_form_for_taxon


# Initial visual screen from the indexed contact sheets. These frames are dominated
# by flowers/seed stalks, show too little sundew tissue, are too distant, or are too
# blurred/occluded for efficient first-pass mask annotation. Raw files are retained.
REJECTED_INDICES = {
    0, 1, 3, 4, 5, 7, 11, 14, 19, 22, 23, 24, 26,
    33, 35, 36, 40, 42, 43, 50, 51, 52, 53, 55, 56, 57,
    61, 64, 66, 68, 70, 71, 73, 74, 78, 81, 82, 83, 84, 87, 88,
    90, 95, 97, 98, 101, 102, 103, 106, 107, 108, 109, 110, 111, 112,
    123, 134, 137, 140, 142, 144, 145, 149,
    152, 157, 163, 164, 171, 173, 175, 176, 179,
    180, 185, 187, 191, 192, 195, 198, 200, 202, 204, 207, 209,
    218, 219, 222, 223, 229, 234, 235, 238,
    240, 242, 245, 252, 253, 255, 259, 262, 263, 269,
    271, 280, 289, 292, 295, 298,
    301, 302, 308, 310, 315, 318, 320, 321, 322, 326, 328,
    335, 338, 341, 345, 346, 347, 349, 351, 355, 356, 357, 358, 359,
    362, 369, 372, 373, 376, 378, 379, 383, 385, 386, 388, 389,
    391, 392, 398, 399, 400, 401, 402, 410, 413, 414, 416, 417, 419,
    422, 423, 429, 431, 433, 437, 439, 440, 441,
    451, 453, 461, 464, 466, 468, 470, 473, 477,
    480, 481, 485, 486, 489, 490, 491, 493, 495, 496, 497,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/inaturalist/metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/curated"))
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--max-side", type=int, default=1600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.metadata)
    if len(rows) != 500:
        raise SystemExit(f"Expected the reviewed 500-image manifest; found {len(rows)} rows")

    selected, reserve = select_core_set(rows, REJECTED_INDICES, target_count=args.count)
    if len(selected) != args.count:
        raise SystemExit(f"Only {len(selected)} suitable images were available")

    selected_set = set(selected)
    reserve_set = set(reserve)
    records = []
    curation_rows = []
    split_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    growth_form_counts: Counter[str] = Counter()
    total_bytes = 0
    expected_images: set[Path] = set()

    for index, source_row in enumerate(rows):
        row = dict(source_row)
        if index in REJECTED_INDICES:
            decision = "rejected"
            reason = "visual_screen"
        elif index in selected_set:
            decision = "core"
            reason = "passed_visual_screen_and_diversity_selection"
        elif index in reserve_set:
            decision = "reserve"
            reason = "passed_visual_screen_not_selected_for_core"
        else:
            raise RuntimeError(f"Unclassified image index {index}")

        curation_record = {
            "review_index": index,
            "photo_id": row["photo_id"],
            "decision": decision,
            "reason": reason,
            "source_path": row["local_path"],
            "split": None,
        }

        if decision == "core":
            split = split_for_observer(str(row.get("observer_login") or "unknown"))
            destination = args.output / "images" / split / f"inat_{row['photo_id']}.jpg"
            expected_images.add(destination)
            width, height = normalize_image(Path(row["local_path"]), destination, args.max_side)
            row.update({
                "curation_status": "core",
                "review_index": index,
                "split": split,
                "curated_path": destination.as_posix(),
                "curated_width": width,
                "curated_height": height,
                "change_notice": "EXIF orientation applied, converted to RGB JPEG, and resized to at most 1600 px; segmentation annotation pending.",
                "growth_form": growth_form_for_taxon(str(row["taxon_name"])),
            })
            curation_record["split"] = split
            records.append(row)
            split_counts[split] += 1
            license_counts[str(row["license_code"])] += 1
            species_counts[str(row["taxon_name"])] += 1
            growth_form_counts[str(row["growth_form"])] += 1
            total_bytes += destination.stat().st_size

        curation_rows.append(curation_record)

    write_jsonl(args.output / "curation.jsonl", curation_rows)
    write_jsonl(args.output / "metadata.jsonl", records)

    for stale_image in (args.output / "images").glob("*/*.jpg"):
        if stale_image not in expected_images:
            stale_image.unlink()

    summary = {
        "raw_images_reviewed": len(rows),
        "core_images": len(selected),
        "reserve_images": len(reserve),
        "rejected_images": len(REJECTED_INDICES),
        "split_counts": dict(sorted(split_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "species_count": len(species_counts),
        "growth_form_counts": dict(sorted(growth_form_counts.items())),
        "top_species": species_counts.most_common(15),
        "curated_bytes": total_bytes,
        "selection": {
            "seed": 20260916,
            "max_per_species": 25,
            "max_per_observer": 8,
            "visual_screen": "review of indexed contact sheets",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
