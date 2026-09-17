"""Compare saved SAM proposals with completed Label Studio annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from label_studio_converter import brush


LABEL_STUDIO_URL = "http://127.0.0.1:8080"


def request_json(path: str, token: str) -> Any:
    request = Request(
        f"{LABEL_STUDIO_URL}{path}",
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def decode_regions(results: list[dict[str, Any]]) -> np.ndarray | None:
    regions = [item for item in results if item.get("type") == "brushlabels"]
    if not regions:
        return None
    masks = []
    expected_shape = None
    for region in regions:
        height = int(region["original_height"])
        width = int(region["original_width"])
        rgba = brush.decode_rle(region["value"]["rle"]).reshape(height, width, 4)
        mask = rgba.max(axis=2) >= 128
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError("annotation regions have inconsistent dimensions")
        expected_shape = mask.shape
        masks.append(mask)
    return np.logical_or.reduce(masks)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"mask dimensions differ: {prediction.shape} != {target.shape}")
    intersection = float(np.logical_and(prediction, target).sum())
    pred_area = float(prediction.sum())
    target_area = float(target.sum())
    union = pred_area + target_area - intersection
    return {
        "iou": intersection / union if union else 1.0,
        "dice": 2.0 * intersection / (pred_area + target_area)
        if pred_area + target_area
        else 1.0,
    }


def annotation_quality(annotation: dict[str, Any]) -> str | None:
    for result in annotation.get("result", []):
        if result.get("type") == "choices":
            choices = result.get("value", {}).get("choices", [])
            if choices:
                return choices[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", default="MobileSAM-CLIPSeg-multiplant-v2")
    parser.add_argument(
        "--proposal-root",
        type=Path,
        default=Path("data/annotations/sam-proposals"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/sam-preannotations/comparison.json"),
    )
    args = parser.parse_args()

    credentials = json.loads(
        Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8")
    )
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    token = credentials["token"]
    task_payload = request_json(
        f"/api/tasks?project={project['id']}&page_size=1000&fields=all",
        token,
    )
    tasks = task_payload.get("tasks") or task_payload.get("results") or task_payload
    predictions = request_json(
        f"/api/predictions?project={project['id']}&page_size=1000",
        token,
    )
    prediction_by_task = {
        item["task"]: item
        for item in predictions
        if item.get("model_version") == args.model_version
    }

    rows = []
    for task in tasks:
        completed = [
            annotation
            for annotation in task.get("annotations", [])
            if not annotation.get("was_cancelled")
            and annotation_quality(annotation) == "complete"
        ]
        if not completed:
            continue
        annotation = max(completed, key=lambda item: item.get("updated_at", ""))
        target = decode_regions(annotation["result"])
        prediction = prediction_by_task.get(task["id"])
        if target is None or prediction is None:
            continue
        old_mask = decode_regions(prediction["result"])
        image_name = Path(task["data"]["image"].split("d=")[-1]).name
        proposal_path = args.proposal_root / task["data"]["split"] / f"{Path(image_name).stem}.png"
        if old_mask is None or not proposal_path.is_file():
            continue
        from PIL import Image

        with Image.open(proposal_path) as source:
            new_mask = np.asarray(source.convert("L")) >= 128
        rows.append(
            {
                "task_id": task["id"],
                "species": task["data"]["species"],
                "split": task["data"]["split"],
                "old": metrics(old_mask, target),
                "new": metrics(new_mask, target),
            }
        )

    if not rows:
        raise SystemExit("No completed human reviews match the available proposals")

    summary = {
        "reviewed_complete": len(rows),
        "old_mean_iou": float(np.mean([row["old"]["iou"] for row in rows])),
        "new_mean_iou": float(np.mean([row["new"]["iou"] for row in rows])),
        "old_mean_dice": float(np.mean([row["old"]["dice"] for row in rows])),
        "new_mean_dice": float(np.mean([row["new"]["dice"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
