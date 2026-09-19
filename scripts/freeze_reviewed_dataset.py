"""Freeze reviewed image/mask pairs into an immutable training snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path


SUPPORT_FILES = {
    "metadata/annotation-audit.json": Path("data/reports/annotation-audit.json"),
    "metadata/mask-audit.json": Path("data/reports/mask-audit/report.json"),
    "metadata/reviewed-mask-export.json": Path("data/reports/reviewed-mask-export.json"),
    "docs/annotation-policy.md": Path("data/annotation-policy.md"),
    "docs/dataset-card.md": Path("reports/dataset-card.md"),
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_snapshot(root: Path, output: Path, version: str) -> Path:
    """Create a checked, self-contained snapshot of accepted pairs."""
    if output.exists():
        raise FileExistsError(f"Snapshot output already exists: {output}")

    annotation_audit = load_json(root / SUPPORT_FILES["metadata/annotation-audit.json"])
    mask_audit = load_json(root / SUPPORT_FILES["metadata/mask-audit.json"])
    export = load_json(root / SUPPORT_FILES["metadata/reviewed-mask-export.json"])
    if annotation_audit["tasks"] != annotation_audit["annotated_tasks"]:
        raise ValueError("Cannot freeze while tasks remain unreviewed")
    if annotation_audit["complete_without_mask"] or export["skipped_complete"]:
        raise ValueError("Cannot freeze accepted annotations without masks")
    if mask_audit["errors"]:
        raise ValueError("Cannot freeze masks that failed the mask audit")
    if export["exported"] != mask_audit["masks"]:
        raise ValueError("Export and mask-audit counts disagree")

    metadata_rows = load_jsonl(root / "data/curated/metadata.jsonl")
    metadata_by_image = {Path(row["curated_path"]).name: row for row in metadata_rows}
    audit_by_mask = {Path(row["mask"]).name: row for row in mask_audit["rows"]}
    training_records = []

    output.mkdir(parents=True)
    for row in sorted(export["masks"], key=lambda item: (item["split"], item["mask"])):
        source_mask = root / row["mask"]
        image_name = source_mask.with_suffix(".jpg").name
        metadata = metadata_by_image.get(image_name)
        if metadata is None:
            raise ValueError(f"No curated metadata for {image_name}")
        source_image = root / metadata["curated_path"]
        if not source_image.is_file() or not source_mask.is_file():
            raise FileNotFoundError(f"Missing accepted pair for {image_name}")

        split = row["split"]
        relative_image = Path("images") / split / image_name
        relative_mask = Path("masks") / split / source_mask.name
        (output / relative_image).parent.mkdir(parents=True, exist_ok=True)
        (output / relative_mask).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, output / relative_image)
        shutil.copy2(source_mask, output / relative_mask)

        mask_details = audit_by_mask[source_mask.name]
        training_records.append({
            **metadata,
            "task_id": row["task_id"],
            "annotation_id": row["annotation_id"],
            "snapshot_image": relative_image.as_posix(),
            "snapshot_mask": relative_mask.as_posix(),
            "curated_image_sha256": digest(source_image),
            "mask_sha256": digest(source_mask),
            "foreground_fraction": mask_details["foreground_fraction"],
            "review_quality": "complete",
        })

    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(metadata_dir / "training-records.jsonl", training_records)
    for target, source in SUPPORT_FILES.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, destination)

    inventory = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        inventory.append({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size})

    split_counts = Counter(row["split"] for row in training_records)
    growth_counts = Counter(row["growth_form"] for row in training_records)
    license_counts = Counter(row["license_code"] for row in training_records)
    manifest = {
        "release_name": "sundew-segmenting-reviewed-dataset",
        "release_version": version,
        "payload": "accepted_image_mask_pairs",
        "pairs": len(training_records),
        "reviewed_tasks": annotation_audit["annotated_tasks"],
        "review_decisions": annotation_audit["quality_by_task"],
        "by_split": dict(sorted(split_counts.items())),
        "by_growth_form": dict(sorted(growth_counts.items())),
        "by_license": dict(sorted(license_counts.items())),
        "test_split_locked": True,
        "files": inventory,
    }
    (output / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0.3.0-reviewed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/sundew-segmenting-reviewed-v0.3.0"),
    )
    parser.add_argument("--zip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    result = build_snapshot(Path.cwd(), args.output, args.version)
    print(f"Built reviewed dataset snapshot at {result}")
    if args.zip:
        archive = Path(shutil.make_archive(str(result), "zip", result.parent, result.name))
        checksum_path = archive.with_suffix(archive.suffix + ".sha256")
        checksum_path.write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
        print(f"Built transfer archive at {archive}")


if __name__ == "__main__":
    main()
