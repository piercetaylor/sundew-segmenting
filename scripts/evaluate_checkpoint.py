"""Evaluate a saved segmentation checkpoint and create validation overlays."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sundew_segmentation.baseline import (
    SEGFORMER_B0,
    UNET_RESNET34,
    add_growth_form_metadata,
    binary_metrics,
    paired_samples,
    resolve_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--images", type=Path, default=Path("data/curated/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/curated/masks"))
    parser.add_argument("--metadata", type=Path, default=Path("data/curated/metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/checkpoint-evaluation"))
    return parser.parse_args()


def make_tile(image_path: Path, target: np.ndarray, prediction: np.ndarray, title: str, size: int = 256) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size - 36), Image.Resampling.LANCZOS)
    target_image = Image.fromarray(target.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    prediction_image = Image.fromarray(prediction.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
    truth = np.asarray(target_image) > 0
    pred = np.asarray(prediction_image) > 0
    array = np.asarray(image, dtype=np.float32)
    colors = np.zeros_like(array)
    colors[np.logical_and(truth, pred)] = (0, 255, 80)
    colors[np.logical_and(~truth, pred)] = (255, 215, 0)
    colors[np.logical_and(truth, ~pred)] = (255, 0, 180)
    marked = np.logical_or(truth, pred)
    array[marked] = 0.55 * array[marked] + 0.45 * colors[marked]
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(Image.fromarray(array.astype(np.uint8)), ((size - image.width) // 2, 0))
    ImageDraw.Draw(tile).text((5, size - 31), title[:42], fill="black", font=ImageFont.load_default())
    return tile


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as error:
        raise SystemExit("Install baseline dependencies with: pip install -e '.[baseline]'") from error

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["config"]
    base = UNET_RESNET34 if saved["name"] == "unet-resnet34" else SEGFORMER_B0
    config = replace(base, **{key: saved[key] for key in base.to_dict() if key in saved})
    model = resolve_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = add_growth_form_metadata(
        paired_samples(args.images, args.masks, args.split, require_masks=False),
        args.metadata,
    )
    metadata = {}
    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            metadata[Path(row["curated_path"]).name] = row

    rows = []
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
    for sample in samples:
        with Image.open(sample["image"]) as source:
            image = source.convert("RGB").resize((config.image_size, config.image_size), Image.Resampling.BILINEAR)
        with Image.open(sample["mask"]) as source:
            target = np.asarray(
                source.convert("L").resize((config.image_size, config.image_size), Image.Resampling.NEAREST)
            ) > 0
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(((image_array - mean) / std).copy()).unsqueeze(0)
        with torch.no_grad():
            prediction = (torch.sigmoid(model(tensor))[0, 0].numpy() >= 0.5)
        metrics = binary_metrics(prediction, target)
        image_name = Path(sample["image"]).name
        row_metadata = metadata.get(image_name, {})
        rows.append({
            "image": Path(sample["image"]).as_posix(),
            "mask": Path(sample["mask"]).as_posix(),
            "species": row_metadata.get("taxon_name", "unknown"),
            "growth_form": sample["growth_form"],
            **metrics,
            "_target": target,
            "_prediction": prediction,
        })

    ordered = sorted(rows, key=lambda row: row["dice"])
    columns = 4
    tiles = [
        make_tile(
            Path(row["image"]), row["_target"], row["_prediction"],
            f"{Path(row['image']).stem}  D={row['dice']:.2f}  {row['growth_form']}",
        )
        for row in ordered
    ]
    sheet = Image.new("RGB", (columns * 256, ((len(tiles) + columns - 1) // columns) * 256), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 256, (index // columns) * 256))
    args.output.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output / f"{args.split}-overlays.jpg", quality=92)

    serializable_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    report = {
        "checkpoint": args.checkpoint.as_posix(),
        "split": args.split,
        "samples": len(rows),
        "mean_metrics": {
            name: float(np.mean([row[name] for row in rows]))
            for name in ("iou", "dice", "precision", "recall", "pixel_accuracy")
        },
        "rows": serializable_rows,
    }
    (args.output / f"{args.split}-metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
