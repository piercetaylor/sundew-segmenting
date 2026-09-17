"""Run one CPU MobileSAM point prompt and save a visual smoke test."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import argparse
import json

import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/curated/images/train/inat_322962690.jpg"),
    )
    parser.add_argument("--checkpoint", type=Path, default=Path(".tools/MobileSAM/weights/mobile_sam.pt"))
    parser.add_argument("--x", type=float, default=0.5, help="Prompt x coordinate as a 0-1 fraction")
    parser.add_argument("--y", type=float, default=0.5, help="Prompt y coordinate as a 0-1 fraction")
    parser.add_argument("--output", type=Path, default=Path("data/reports/mobilesam-smoke-test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise SystemExit(f"Could not read {args.image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    point = np.array([[args.x * width, args.y * height]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)

    started = perf_counter()
    model = sam_model_registry["vit_t"](checkpoint=str(args.checkpoint))
    model.to(device="cpu")
    predictor = SamPredictor(model)
    loaded_seconds = perf_counter() - started

    started = perf_counter()
    predictor.set_image(image)
    embedding_seconds = perf_counter() - started

    started = perf_counter()
    masks, scores, _ = predictor.predict(point_coords=point, point_labels=label, multimask_output=True)
    prediction_seconds = perf_counter() - started
    best_index = int(np.argmax(scores))
    mask = masks[best_index]

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / "mask.png")
    overlay = image.copy()
    tint = np.array([244, 63, 94], dtype=np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * tint).astype(np.uint8)
    cv2.circle(overlay, (int(point[0, 0]), int(point[0, 1])), 10, (245, 158, 11), -1)
    Image.fromarray(overlay).save(args.output / "overlay.jpg", quality=92)

    report = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "image": args.image.as_posix(),
        "image_width": width,
        "image_height": height,
        "prompt_xy": point[0].tolist(),
        "model_load_seconds": loaded_seconds,
        "embedding_seconds": embedding_seconds,
        "prediction_seconds": prediction_seconds,
        "score": float(scores[best_index]),
        "foreground_pixels": int(mask.sum()),
        "foreground_fraction": float(mask.mean()),
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
