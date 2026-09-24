"""Convert field-probe brush annotations into PNG masks and a training manifest.

`export_reviewed_masks.py` pulls from a running Label Studio instance and keys
off `.tools/label-studio-project.json`, which points at the curated project. It
also deletes masks it does not expect, so aiming it at a second project would
remove the curated ones. This script reads the Label Studio database directly,
read-only, and writes only inside its own output directory.

Unlike the curated export, masks are written for every reviewed annotation that
carries a brush, `ambiguous` included, and the quality choice is recorded per
row in the manifest. Which of them to train on is a downstream decision; losing
the `ambiguous` masks here would make that decision unrecoverable.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from label_studio_converter.brush import decode_rle
from PIL import Image

QUALITY_FIELD = "quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/label-studio/label_studio.sqlite3"))
    parser.add_argument("--project", type=int, required=True, help="Label Studio project id")
    parser.add_argument("--images", type=Path, default=Path("data/curated/images/field-probe"))
    parser.add_argument("--split", default="field-probe", help="split name used for the output directory")
    parser.add_argument("--output", type=Path, default=Path("data/curated/masks"))
    parser.add_argument("--provenance", type=Path, default=None,
                        help="scrape metadata.jsonl, joined on photo_id to carry licence and credit")
    parser.add_argument("--manifest", type=Path, default=Path("data/curated/field-probe-records.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/curated/field-probe-annotations.json"),
                        help="raw annotation export, kept for provenance")
    parser.add_argument("--report", type=Path, default=Path("data/reports/field-probe-mask-export.json"))
    return parser.parse_args()


def latest_annotations(db: Path, project: int) -> list[dict]:
    """One annotation per task: the most recently updated review that was not cancelled."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT t.id, t.data, a.id, a.result, a.updated_at, a.was_cancelled
            FROM task t
            JOIN task_completion a ON a.task_id = t.id
            WHERE t.project_id = ?
            ORDER BY t.id, a.updated_at
            """,
            (project,),
        ).fetchall()
    finally:
        connection.close()

    by_task: dict[int, dict] = {}
    for task_id, data, annotation_id, result, updated_at, was_cancelled in rows:
        if was_cancelled:
            continue
        by_task[task_id] = {
            "task_id": task_id,
            "data": json.loads(data),
            "annotation_id": annotation_id,
            "result": json.loads(result) if isinstance(result, str) else (result or []),
            "updated_at": updated_at,
        }
    return [by_task[key] for key in sorted(by_task)]


def quality_of(result: list[dict]) -> str | None:
    for item in result:
        if item.get("from_name") == QUALITY_FIELD or item.get("type") == "choices":
            choices = item.get("value", {}).get("choices") or []
            if choices:
                return str(choices[0])
    return None


def image_name(task_data: dict) -> str:
    return Path(task_data["image"].split("d=")[-1]).name


def brush_mask(result: list[dict], size: tuple[int, int], name: str) -> "np.ndarray | None":
    """Union of every brush layer, as a boolean array shaped (height, width)."""
    width, height = size
    combined = None
    for item in result:
        if item.get("type") != "brushlabels":
            continue
        frame = (item["original_width"], item["original_height"])
        if frame != size:
            raise ValueError(f"{name}: annotation frame {frame} does not match image {size}")
        # Same threshold as `export_reviewed_masks.py`, so field and curated
        # masks agree on anti-aliased brush edges.
        rgba = decode_rle(item["value"]["rle"]).reshape(height, width, 4)
        layer = rgba.max(axis=2) >= 128
        combined = layer if combined is None else (combined | layer)
    return combined


def load_provenance(path: "Path | None") -> dict:
    if path is None:
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[int(row["photo_id"])] = row
    return records


def main() -> None:
    args = parse_args()
    mask_dir = args.output / args.split
    mask_dir.mkdir(parents=True, exist_ok=True)
    provenance = load_provenance(args.provenance)

    annotations = latest_annotations(args.db, args.project)
    if not annotations:
        raise SystemExit(f"no annotations found for project {args.project} in {args.db}")

    exported = []
    rejected = []
    missing_brush = []
    raw_export = []

    for entry in annotations:
        data = entry["data"]
        name = image_name(data)
        image_path = args.images / name
        if not image_path.exists():
            raise SystemExit(f"{name}: annotated image not found at {image_path}")

        quality = quality_of(entry["result"])
        raw_export.append({
            "id": entry["task_id"],
            "data": data,
            "annotations": [{
                "id": entry["annotation_id"],
                "result": entry["result"],
                "updated_at": entry["updated_at"],
            }],
        })

        if quality == "reject":
            rejected.append(name)
            continue

        with Image.open(image_path) as image:
            size = image.size
        mask = brush_mask(entry["result"], size, name)
        if mask is None:
            # A reviewed image with no brush is only coherent as a reject; the
            # policy allows a near-empty mask but not a missing one.
            missing_brush.append(name)
            continue

        mask_path = mask_dir / f"{image_path.stem}.png"
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

        photo_id = int(data.get("photo_id"))
        source = provenance.get(photo_id, {})
        exported.append({
            "photo_id": photo_id,
            "curated_path": f"{args.images.as_posix()}/{name}",
            "mask_path": mask_path.as_posix(),
            "split": args.split,
            "quality": quality,
            "growth_form": data.get("growth_form") or "unknown",
            "taxon_name": data.get("species") or source.get("taxon_name"),
            "common_name": source.get("common_name"),
            "observation_id": source.get("observation_id"),
            "observer_login": source.get("observer_login"),
            "creator": source.get("creator"),
            "attribution": source.get("attribution"),
            "license_code": source.get("license_code") or data.get("license"),
            "license_url": source.get("license_url"),
            "source_page": data.get("source_page") or source.get("source_page"),
            "sha256": source.get("sha256"),
            "curated_width": size[0],
            "curated_height": size[1],
            "foreground_fraction": round(float(mask.mean()), 6),
            "annotation_id": entry["annotation_id"],
            "reviewed_at": entry["updated_at"],
            "curation_status": "field-probe",
            "change_notice": "Downloaded without intentional modification; binary mask traced in Label Studio.",
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(exported, key=lambda r: r["photo_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    args.annotations.write_text(json.dumps(raw_export, indent=1, sort_keys=True), encoding="utf-8")

    by_quality = {}
    for row in exported:
        by_quality[str(row["quality"])] = by_quality.get(str(row["quality"]), 0) + 1

    report = {
        "project": args.project,
        "split": args.split,
        "reviewed": len(annotations),
        "exported": len(exported),
        "by_quality": by_quality,
        "rejected": sorted(rejected),
        "missing_brush": sorted(missing_brush),
        "mask_dir": mask_dir.as_posix(),
        "manifest": args.manifest.as_posix(),
        "mean_foreground_fraction": round(
            sum(r["foreground_fraction"] for r in exported) / len(exported), 6
        ) if exported else None,
        "empty_masks": sorted(r["photo_id"] for r in exported if r["foreground_fraction"] == 0.0),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")

    print(f"reviewed {len(annotations)} -> exported {len(exported)} masks into {mask_dir}")
    print(f"  by quality: {by_quality}")
    if rejected:
        print(f"  rejected (no mask written): {len(rejected)}")
    if missing_brush:
        print(f"  reviewed but no brush: {missing_brush}")


if __name__ == "__main__":
    main()
