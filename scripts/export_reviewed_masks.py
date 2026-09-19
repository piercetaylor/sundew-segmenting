"""Export only human-reviewed complete Label Studio annotations as PNG masks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from label_studio_converter import brush


BASE_URL = "http://127.0.0.1:8080"


def request_json(path: str, token: str) -> Any:
    request = Request(f"{BASE_URL}{path}", headers={"Authorization": f"Token {token}"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def quality(annotation: dict[str, Any]) -> str | None:
    for result in annotation.get("result", []):
        if result.get("type") == "choices" and result.get("from_name") == "quality":
            return next(iter(result.get("value", {}).get("choices", [])), None)
    return None


def decode_mask(results: list[dict[str, Any]]) -> np.ndarray | None:
    masks = []
    shape = None
    for result in results:
        if result.get("type") != "brushlabels":
            continue
        height, width = int(result["original_height"]), int(result["original_width"])
        rgba = brush.decode_rle(result["value"]["rle"]).reshape(height, width, 4)
        mask = rgba.max(axis=2) >= 128
        if shape is not None and mask.shape != shape:
            raise ValueError("brush regions within an annotation have inconsistent dimensions")
        shape = mask.shape
        masks.append(mask)
    return np.logical_or.reduce(masks) if masks else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/curated/masks"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/reviewed-mask-export.json"))
    args = parser.parse_args()
    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    payload = request_json(f"/api/tasks?project={project['id']}&page_size=1000&fields=all", credentials["token"])
    tasks = payload.get("tasks") or payload.get("results") or payload

    exported = []
    skipped_complete = []
    expected_paths: set[Path] = set()
    for task in tasks:
        completed = [
            item for item in task.get("annotations", [])
            if not item.get("was_cancelled") and quality(item) == "complete"
        ]
        if not completed:
            continue
        annotation = max(completed, key=lambda item: item.get("updated_at", ""))
        mask = decode_mask(annotation.get("result", []))
        if mask is None:
            skipped_complete.append({"task_id": task["id"], "annotation_id": annotation["id"], "reason": "no_brush_mask"})
            continue
        split = task["data"]["split"]
        image_name = Path(task["data"]["image"].split("d=")[-1]).name
        image_path = Path("data/curated/images") / split / image_name
        with Image.open(image_path) as image:
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"image/mask size mismatch for task {task['id']}")
        destination = args.output / split / f"{Path(image_name).stem}.png"
        expected_paths.add(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(destination)
        exported.append({
            "task_id": task["id"], "annotation_id": annotation["id"], "split": split,
            "species": task["data"]["species"],
            "growth_form": task["data"].get("growth_form", "unknown"),
            "mask": destination.as_posix(),
        })

    removed_stale = []
    for stale in args.output.glob("*/*.png"):
        if stale not in expected_paths:
            stale.unlink()
            removed_stale.append(stale.as_posix())

    report = {
        "exported": len(exported),
        "by_split": dict(Counter(row["split"] for row in exported)),
        "by_growth_form": dict(Counter(row["growth_form"] for row in exported)),
        "skipped_complete": skipped_complete,
        "removed_stale": removed_stale,
        "masks": exported,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "masks"}, indent=2))


if __name__ == "__main__":
    main()
