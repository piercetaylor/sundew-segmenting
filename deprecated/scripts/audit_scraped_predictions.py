"""Run a checkpoint over unlabelled scraped images and render review sheets.

Scraped images have no masks, so this cannot report IoU. It reports the
distribution statistics that *are* checkable without ground truth, and it
renders contact sheets, because the failure modes that matter on uncurated
images are ones the statistics do not catch.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sundew_segmentation.baseline import SEGFORMER_B0, UNET_RESNET34, resolve_model

MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="directory of scraped .jpg files")
    parser.add_argument("--metadata", type=Path, default=None, help="acquisition metadata.jsonl, for species labels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("unet-resnet34", "segformer-b0"), default="segformer-b0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tile", type=int, default=256)
    return parser.parse_args()


def components(pred: np.ndarray, min_frac: float = 0.002) -> int:
    """Count connected foreground blobs above a size floor."""
    size = pred.shape[0]
    seen = np.zeros_like(pred, dtype=bool)
    counts = []
    for y in range(0, size, 4):
        for x in range(0, size, 4):
            if pred[y, x] and not seen[y, x]:
                queue = deque([(y, x)])
                seen[y, x] = True
                total = 0
                while queue:
                    cy, cx = queue.popleft()
                    total += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < size and 0 <= nx < size and pred[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
                counts.append(total)
    return sum(1 for c in counts if c >= min_frac * size * size)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base = UNET_RESNET34 if args.model == "unet-resnet34" else SEGFORMER_B0
    config = replace(base, **{k: v for k, v in checkpoint["config"].items()
                              if k in ("image_size", "batch_size", "epochs", "learning_rate")})
    model = resolve_model(config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    size = config.image_size

    labels: dict[str, str] = {}
    if args.metadata and args.metadata.exists():
        for line in args.metadata.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[Path(row["local_path"]).name] = str(row.get("taxon_name") or "?")

    args.output.mkdir(parents=True, exist_ok=True)
    records, tiles = [], []
    for path in sorted(args.images.glob("*.jpg")):
        with Image.open(path) as source:
            original = source.convert("RGB")
            resized = original.resize((size, size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        with torch.no_grad():
            probability = torch.sigmoid(model(((tensor - MEAN) / STD)[None]))[0, 0].numpy()
        pred = probability >= args.threshold
        species = labels.get(path.name, "?")
        records.append({
            "name": path.name,
            "species": species,
            "foreground_fraction": float(pred.mean()),
            "components": components(pred),
            "mean_confidence": float(probability[pred].mean()) if pred.any() else 0.0,
        })

        tile_size = args.tile
        thumb = original.copy()
        thumb.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        overlay = np.asarray(Image.fromarray((pred * 255).astype(np.uint8))
                             .resize(thumb.size, Image.Resampling.NEAREST)) > 0
        array = np.asarray(thumb, dtype=np.float32).copy()
        colour = np.zeros_like(array)
        colour[overlay] = (0, 255, 80)
        array[overlay] = 0.55 * array[overlay] + 0.45 * colour[overlay]
        tile = Image.new("RGB", (tile_size, tile_size + 14), "white")
        tile.paste(Image.fromarray(array.astype(np.uint8)), ((tile_size - thumb.width) // 2, 0))
        ImageDraw.Draw(tile).text(
            (2, tile_size + 1),
            f"{species[:26]} {pred.mean() * 100:.0f}%",
            fill="black",
        )
        tiles.append(tile)

    (args.output / "predictions.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    tile_size = args.tile
    height = tile_size + 14
    for start in range(0, len(tiles), 25):
        chunk = tiles[start:start + 25]
        sheet = Image.new("RGB", (5 * tile_size, 5 * height), "white")
        for index, tile in enumerate(chunk):
            sheet.paste(tile, ((index % 5) * tile_size, (index // 5) * height))
        sheet.save(args.output / f"contact-{start // 25 + 1}.png")

    fractions = [r["foreground_fraction"] for r in records]
    print(f"images                  {len(records)}")
    print(f"mean foreground         {np.mean(fractions) * 100:.1f}%")
    print(f"near-empty (<0.5%)      {sum(1 for f in fractions if f < 0.005)}")
    print(f"runaway (>60%)          {sum(1 for f in fractions if f > 0.60)}")
    print(f"fragmented (>=6 blobs)  {sum(1 for r in records if r['components'] >= 6)}")
    print()
    print("Statistics do not catch confident false positives on skin or bright")
    print("artificial objects. Review the contact sheets by eye.")


if __name__ == "__main__":
    main()
