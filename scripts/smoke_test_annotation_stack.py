"""Verify a real Label Studio -> MobileSAM interactive mask request."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image


LABEL_STUDIO_URL = "http://127.0.0.1:8080"
TARGET_FILENAME = "inat_430092977.jpg"


def request_json(path: str, token: str, method: str = "GET", body: Any | None = None) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{LABEL_STUDIO_URL}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: {error.code} {details}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default=TARGET_FILENAME)
    args = parser.parse_args()

    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    token = credentials["token"]
    tasks_payload = request_json(f"/api/tasks?project={project['id']}&page_size=1000", token)
    tasks = tasks_payload.get("tasks") or tasks_payload.get("results") or tasks_payload
    task = next((item for item in tasks if item["data"]["image"].endswith(args.filename)), None)
    if task is None:
        raise RuntimeError(f"No Label Studio task found for {args.filename}")

    image_path = next(Path("data/curated/images").rglob(args.filename))
    with Image.open(image_path) as image:
        width, height = image.size

    context = {
        "result": [
            {
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {"x": 50.0, "y": 50.0, "keypointlabels": ["sundew"]},
                "is_positive": True,
                "id": "smoke-point",
                "from_name": "sam_points",
                "to_name": "image",
                "type": "keypointlabels",
                "origin": "manual",
            }
        ]
    }
    started = time.perf_counter()
    response = request_json(
        f"/api/ml/{project['ml_backend_id']}/interactive-annotating",
        token,
        method="POST",
        body={"task": task["id"], "context": context},
    )
    elapsed = time.perf_counter() - started
    if response.get("errors"):
        raise RuntimeError("; ".join(response["errors"]))
    prediction = response.get("data", response)
    regions = prediction.get("result", [])
    if len(regions) != 1 or regions[0].get("type") != "brushlabels":
        raise RuntimeError(f"Unexpected MobileSAM response: {response}")

    summary = {
        "task_id": task["id"],
        "filename": args.filename,
        "elapsed_seconds": round(elapsed, 3),
        "score": round(float(prediction["score"]), 4),
        "region_type": regions[0]["type"],
        "label": regions[0]["value"]["brushlabels"][0],
        "rle_values": len(regions[0]["value"]["rle"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
