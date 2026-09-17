"""Fail fast on provenance, split, file, and checksum errors."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import argparse
import json

from PIL import Image

from sundew_segmentation.acquisition import ALLOWED_LICENSES, read_manifest


FORBIDDEN_LOCATION_KEYS = {"location", "latitude", "longitude", "lat", "lon", "geojson", "place_guess"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    raw = read_manifest(Path("data/raw/inaturalist/metadata.jsonl"))
    curated = read_manifest(Path("data/curated/metadata.jsonl"))
    tasks = json.loads(Path("data/annotations/label-studio-tasks.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    if len(raw) != 500:
        errors.append(f"raw manifest has {len(raw)} rows, expected 500")
    if len(curated) != 250:
        errors.append(f"curated manifest has {len(curated)} rows, expected 250")
    if len(tasks) != len(curated):
        errors.append(f"task count {len(tasks)} differs from curated count {len(curated)}")

    raw_photo_ids = [int(row["photo_id"]) for row in raw]
    raw_hashes = [str(row["sha256"]) for row in raw]
    if len(set(raw_photo_ids)) != len(raw_photo_ids):
        errors.append("raw manifest contains duplicate photo IDs")
    if len(set(raw_hashes)) != len(raw_hashes):
        errors.append("raw manifest contains duplicate SHA-256 values")

    observer_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in raw:
        leaked_keys = FORBIDDEN_LOCATION_KEYS & set(row)
        if leaked_keys:
            errors.append(f"photo {row['photo_id']} contains location keys: {sorted(leaked_keys)}")
        if row.get("license_code") not in ALLOWED_LICENSES:
            errors.append(f"photo {row['photo_id']} has disallowed license {row.get('license_code')}")
        source = Path(row["local_path"])
        if not source.is_file():
            errors.append(f"missing raw file: {source}")
        elif args.verify_hashes and file_sha256(source) != row["sha256"]:
            errors.append(f"checksum mismatch: {source}")

    for row in curated:
        image_path = Path(row["curated_path"])
        if not image_path.is_file():
            errors.append(f"missing curated file: {image_path}")
            continue
        observer_splits[str(row["observer_login"])].add(str(row["split"]))
        with Image.open(image_path) as image:
            if image.size != (int(row["curated_width"]), int(row["curated_height"])):
                errors.append(f"dimension mismatch: {image_path}")
            if image.mode != "RGB":
                errors.append(f"non-RGB derivative: {image_path}")

    leaking_observers = {observer: splits for observer, splits in observer_splits.items() if len(splits) > 1}
    if leaking_observers:
        errors.append(f"observers cross splits: {leaking_observers}")

    disk_images = set(Path("data/curated/images").glob("*/*.jpg"))
    manifest_images = {Path(row["curated_path"]) for row in curated}
    if disk_images != manifest_images:
        errors.append(
            f"curated image set mismatch: {len(disk_images)} files and {len(manifest_images)} manifest paths"
        )

    if errors:
        raise SystemExit("Dataset validation failed:\n- " + "\n- ".join(errors))
    mode = "including source checksums" if args.verify_hashes else "without source checksum scan"
    print(f"Validated 500 raw and 250 curated records {mode}; no errors found.")


if __name__ == "__main__":
    main()
