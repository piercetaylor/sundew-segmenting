"""Generate reviewable MobileSAM preannotations for every Label Studio task.

The output is stored as model predictions and proposal PNGs. It is deliberately
kept separate from ``data/curated/masks`` so unreviewed predictions cannot enter
the training pipeline as ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2
import numpy as np
import torch
from label_studio_converter import brush
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image, ImageDraw


LABEL_STUDIO_URL = "http://127.0.0.1:8080"
MODEL_VERSION = "MobileSAM-CLIPSeg-autoprompt-v1"
PROMPT_FRACTIONS = (
    (0.50, 0.50),
    (0.32, 0.32),
    (0.68, 0.32),
    (0.32, 0.68),
    (0.68, 0.68),
)


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
        with urlopen(request, timeout=180) as response:
            payload = response.read()
            return json.loads(payload) if payload else None
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Label Studio API {method} {path} failed: {error.code} {details}") from error


def local_image_path(image_url: str) -> Path:
    relative = parse_qs(urlparse(image_url).query).get("d", [None])[0]
    if not relative:
        raise ValueError(f"Unsupported image URL: {image_url}")
    root = Path("data/curated").resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Image is outside the curated root or missing: {path}")
    return path


def candidate_score(
    mask: np.ndarray,
    predicted_iou: float,
    point: tuple[int, int],
    saturation: np.ndarray,
    semantic_heat: np.ndarray | None,
) -> float:
    height, width = mask.shape
    area = float(mask.mean())
    if area < 0.004 or area > 0.52:
        return -math.inf
    area_prior = math.exp(-1.35 * abs(math.log(area / 0.14)))
    point_distance = math.dist(point, (width / 2, height / 2)) / math.hypot(width / 2, height / 2)
    center_prior = 1.0 - min(point_distance, 1.0)
    border_fraction = float(
        np.concatenate((mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])).mean()
    )
    inside_saturation = float(saturation[mask].mean()) if mask.any() else 0.0
    saturation_contrast = inside_saturation - float(saturation.mean())
    semantic_contrast = 0.0
    semantic_peak = 0.0
    if semantic_heat is not None:
        semantic_contrast = float(semantic_heat[mask].mean()) - float(semantic_heat.mean())
        semantic_peak = float(semantic_heat[point[1], point[0]])
    large_region_penalty = max(0.0, (area - 0.32) / 0.20)
    return (
        0.48 * predicted_iou
        + 0.26 * area_prior
        + 0.10 * center_prior
        + 0.16 * saturation_contrast
        + 0.52 * semantic_contrast
        + 0.14 * semantic_peak
        - 0.35 * border_fraction
        - 0.25 * large_region_penalty
    )


def semantic_prompt_points(semantic_heat: np.ndarray, count: int = 5) -> list[tuple[int, int]]:
    height, width = semantic_heat.shape
    working = cv2.GaussianBlur(semantic_heat, (0, 0), sigmaX=max(min(width, height) / 100, 2))
    working = working.copy()
    border_y = max(int(height * 0.04), 1)
    border_x = max(int(width * 0.04), 1)
    working[:border_y, :] = 0
    working[-border_y:, :] = 0
    working[:, :border_x] = 0
    working[:, -border_x:] = 0
    global_peak = float(working.max())
    if global_peak < 0.25:
        return [(int(x * width), int(y * height)) for x, y in PROMPT_FRACTIONS]
    points = []
    radius = max(int(min(width, height) * 0.14), 1)
    yy, xx = np.ogrid[:height, :width]
    for _ in range(count):
        flat_index = int(np.argmax(working))
        y, x = np.unravel_index(flat_index, working.shape)
        if float(working[y, x]) < global_peak * 0.62:
            break
        points.append((int(x), int(y)))
        working[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = 0
    return points or [(width // 2, height // 2)]


class SemanticGuide:
    def __init__(self, cache_dir: Path):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        model_id = "CIDAS/clipseg-rd64-refined"
        try:
            self.processor = CLIPSegProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            self.model = CLIPSegForImageSegmentation.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=True,
            ).eval()
        except OSError:
            self.processor = CLIPSegProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = CLIPSegForImageSegmentation.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            ).eval()
        self.prompts = ["a sundew carnivorous plant", "a Drosera plant with sticky leaves"]

    def heatmap(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        inputs = self.processor(
            text=self.prompts,
            images=[pil_image] * len(self.prompts),
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        heat = torch.sigmoid(logits).amax(dim=0).cpu().numpy()
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        return np.clip(heat, 0.0, 1.0)


def propose_mask(
    predictor: SamPredictor,
    image: np.ndarray,
    semantic_heat: np.ndarray | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    saturation = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    predictor.set_image(image)
    best: tuple[float, np.ndarray, float, tuple[int, int]] | None = None
    points = (
        semantic_prompt_points(semantic_heat)
        if semantic_heat is not None
        else [(int(x * width), int(y * height)) for x, y in PROMPT_FRACTIONS]
    )
    for point in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
        for mask, predicted_iou in zip(masks, scores):
            rank = candidate_score(mask, float(predicted_iou), point, saturation, semantic_heat)
            if best is None or rank > best[0]:
                best = (rank, mask, float(predicted_iou), point)
    if best is None or not math.isfinite(best[0]):
        raise RuntimeError("MobileSAM did not return a usable proposal")
    return best[1], float(np.clip(best[2], 0.0, 1.0)), best[3]


def prediction_result(mask: np.ndarray, width: int, height: int, score: float) -> dict[str, Any]:
    return {
        "id": str(uuid4())[:8],
        "from_name": "label",
        "to_name": "image",
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
        "type": "brushlabels",
        "score": score,
        "readonly": False,
        "value": {
            "format": "rle",
            "rle": brush.mask2rle(mask.astype(np.uint8) * 255),
            "brushlabels": ["sundew"],
        },
    }


def save_overlay(image: np.ndarray, mask: np.ndarray, point: tuple[int, int], destination: Path) -> None:
    overlay = image.copy()
    tint = np.asarray([244, 63, 94], dtype=np.uint8)
    overlay[mask] = (0.58 * overlay[mask] + 0.42 * tint).astype(np.uint8)
    rendered = Image.fromarray(overlay)
    draw = ImageDraw.Draw(rendered)
    radius = max(5, min(rendered.size) // 100)
    draw.ellipse(
        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
        fill=(245, 158, 11),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered.thumbnail((640, 640), Image.Resampling.LANCZOS)
    rendered.save(destination, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--upload", action="store_true", help="Create Label Studio predictions")
    parser.add_argument("--replace", action="store_true", help="Replace predictions from this model version")
    parser.add_argument("--overlay-limit", type=int, default=12)
    parser.add_argument("--no-semantic-guide", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("data/reports/sam-preannotations/report.json"))
    args = parser.parse_args()

    credentials = json.loads(Path(".tools/label-studio-credentials.json").read_text(encoding="utf-8"))
    project = json.loads(Path(".tools/label-studio-project.json").read_text(encoding="utf-8"))
    token = credentials["token"]
    tasks_payload = request_json(f"/api/tasks?project={project['id']}&page_size=1000", token)
    tasks = tasks_payload.get("tasks") or tasks_payload.get("results") or tasks_payload
    tasks = sorted(tasks, key=lambda item: item["id"])
    if args.limit is not None:
        tasks = tasks[: args.limit]

    predictions_payload = request_json(f"/api/predictions?project={project['id']}&page_size=1000", token)
    predictions = predictions_payload.get("results", predictions_payload) if isinstance(predictions_payload, dict) else predictions_payload
    existing = {
        item["task"]: item
        for item in predictions
        if item.get("model_version") == MODEL_VERSION
    }
    if args.upload and args.replace:
        for item in existing.values():
            request_json(f"/api/predictions/{item['id']}/", token, method="DELETE")
        existing.clear()

    checkpoint = Path(".tools/MobileSAM/weights/mobile_sam.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device=device)
    predictor = SamPredictor(model)
    semantic_guide = None if args.no_semantic_guide else SemanticGuide(Path(".tools/huggingface"))
    proposal_root = Path("data/annotations/sam-proposals")
    overlay_root = Path("data/reports/sam-preannotations/overlays")
    rows = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=1):
        image_path = local_image_path(task["data"]["image"])
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        item_started = time.perf_counter()
        semantic_heat = semantic_guide.heatmap(image) if semantic_guide is not None else None
        mask, score, point = propose_mask(predictor, image, semantic_heat)
        split = task["data"]["split"]
        mask_path = proposal_root / split / f"{image_path.stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

        if index <= args.overlay_limit:
            save_overlay(image, mask, point, overlay_root / f"{index:03d}-{image_path.stem}.jpg")

        uploaded = False
        if args.upload and task["id"] not in existing:
            request_json(
                "/api/predictions/",
                token,
                method="POST",
                body={
                    "task": task["id"],
                    "project": project["id"],
                    "result": [prediction_result(mask, width, height, score)],
                    "score": score,
                    "model_version": MODEL_VERSION,
                },
            )
            uploaded = True
        row = {
            "task_id": task["id"],
            "image": image_path.as_posix(),
            "split": split,
            "score": round(score, 6),
            "foreground_fraction": round(float(mask.mean()), 6),
            "prompt_xy": list(point),
            "seconds": round(time.perf_counter() - item_started, 3),
            "uploaded": uploaded,
        }
        rows.append(row)
        print(
            f"[{index}/{len(tasks)}] task={task['id']} score={score:.3f} "
            f"area={mask.mean():.3f} seconds={row['seconds']:.2f}",
            flush=True,
        )

    if args.upload:
        request_json(
            f"/api/projects/{project['id']}",
            token,
            method="PATCH",
            body={"model_version": MODEL_VERSION, "show_collab_predictions": True},
        )
    report = {
        "model_version": MODEL_VERSION,
        "device": device,
        "tasks_processed": len(rows),
        "predictions_uploaded": sum(row["uploaded"] for row in rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "mean_score": round(float(np.mean([row["score"] for row in rows])), 6),
        "mean_foreground_fraction": round(float(np.mean([row["foreground_fraction"] for row in rows])), 6),
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
