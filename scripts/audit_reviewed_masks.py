"""Validate reviewed masks and create an overlay sheet for rapid visual QA."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_thumbnail(image_path: Path, mask_path: Path, title: str, size: int = 256) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.convert("L")
    image.thumbnail((size, size - 36), Image.Resampling.LANCZOS)
    mask = mask.resize(image.size, Image.Resampling.NEAREST)
    image_array = np.asarray(image, dtype=np.float32)
    mask_array = np.asarray(mask) >= 128
    red = np.zeros_like(image_array)
    red[..., 0] = 255
    image_array[mask_array] = 0.55 * image_array[mask_array] + 0.45 * red[mask_array]
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(Image.fromarray(image_array.astype(np.uint8)), ((size - image.width) // 2, 0))
    ImageDraw.Draw(tile).text((5, size - 31), title[:42], fill="black", font=ImageFont.load_default())
    return tile


def main() -> None:
    export = json.loads(Path("data/reports/reviewed-mask-export.json").read_text(encoding="utf-8"))
    rows = []
    errors = []
    for item in export["masks"]:
        mask_path = Path(item["mask"])
        image_path = Path("data/curated/images") / item["split"] / (mask_path.stem + ".jpg")
        with Image.open(mask_path) as source:
            mask = np.asarray(source.convert("L"))
        with Image.open(image_path) as source:
            image_size = source.size
        if image_size != (mask.shape[1], mask.shape[0]):
            errors.append(f"dimension mismatch: {mask_path}")
        values = set(np.unique(mask).tolist())
        if not values <= {0, 255}:
            errors.append(f"non-binary values in {mask_path}: {sorted(values)[:10]}")
        fraction = float((mask > 0).mean())
        if fraction == 0:
            errors.append(f"empty mask: {mask_path}")
        rows.append({**item, "image": image_path.as_posix(), "foreground_fraction": fraction})

    ordered = sorted(rows, key=lambda row: row["foreground_fraction"])
    selected = ordered[:12] + ordered[-12:]
    columns = 4
    tiles = [
        overlay_thumbnail(
            Path(row["image"]),
            Path(row["mask"]),
            f"task {row['task_id']}  {row['foreground_fraction']:.1%}  {row['growth_form']}",
        )
        for row in selected
    ]
    sheet = Image.new("RGB", (columns * 256, ((len(tiles) + columns - 1) // columns) * 256), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 256, (index // columns) * 256))
    output_dir = Path("data/reports/mask-audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(output_dir / "foreground-extremes.jpg", quality=92)

    report = {
        "masks": len(rows),
        "errors": errors,
        "flagged_small_under_0_2_percent": [row["task_id"] for row in rows if row["foreground_fraction"] < 0.002],
        "flagged_large_over_85_percent": [row["task_id"] for row in rows if row["foreground_fraction"] > 0.85],
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_growth_form": dict(Counter(row["growth_form"] for row in rows)),
        "foreground_fraction": {
            "minimum": min(row["foreground_fraction"] for row in rows),
            "median": float(np.median([row["foreground_fraction"] for row in rows])),
            "maximum": max(row["foreground_fraction"] for row in rows),
        },
        "rows": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if errors:
        raise SystemExit("Mask audit failed")


if __name__ == "__main__":
    main()
