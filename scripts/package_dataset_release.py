"""Build a metadata-only, license-aware dataset release directory.

Images and masks stay outside Git. The release directory contains the manifests
needed to reproduce the image set and a SHA-256 inventory of every released file.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import argparse
import json
import shutil


RELEASE_FILES = (
    Path("reports/dataset-card.md"),
    Path("data/raw/inaturalist/metadata.jsonl"),
    Path("data/raw/inaturalist/acquisition.json"),
    Path("data/curated/metadata.jsonl"),
    Path("data/curated/curation.jsonl"),
    Path("data/curated/summary.json"),
    Path("data/annotations/label-studio-tasks.json"),
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_release(root: Path, output: Path) -> Path:
    missing = [path.as_posix() for path in RELEASE_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing release inputs: " + ", ".join(missing))
    if output.exists():
        raise FileExistsError(f"Release output already exists: {output}")
    output.mkdir(parents=True)
    inventory = []
    for source in RELEASE_FILES:
        target = output / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, target)
        inventory.append({"path": source.as_posix(), "sha256": digest(target), "bytes": target.stat().st_size})
    manifest = {
        "release_name": "sundew-segmenting-core",
        "release_version": "v0.1.0-metadata",
        "payload": "metadata_only",
        "images_included": False,
        "masks_included": False,
        "license_policy": "CC0 or CC BY per image; verify source terms before redistribution",
        "files": inventory,
    }
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dist/sundew-segmenting-metadata-v0.1.0"))
    args = parser.parse_args()
    result = build_release(Path.cwd(), args.output)
    print(f"Built metadata-only dataset release at {result}")


if __name__ == "__main__":
    main()
