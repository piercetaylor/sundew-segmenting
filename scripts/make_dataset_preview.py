"""Build a small CC0-only preview montage suitable for the repository README."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

from sundew_segmentation.acquisition import read_manifest


def main() -> None:
    rows = [row for row in read_manifest(Path("data/curated/metadata.jsonl")) if row["license_code"] == "cc0"][:12]
    tile = 300
    caption = 42
    canvas = Image.new("RGB", (tile * 4, (tile + caption) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    credits = ["# Dataset preview attribution", "", "All preview sources are CC0 1.0.", ""]
    for index, row in enumerate(rows):
        with Image.open(row["curated_path"]) as image:
            image = ImageOps.fit(image.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
            x = (index % 4) * tile
            y = (index // 4) * (tile + caption)
            canvas.paste(image, (x, y))
            draw.text((x + 8, y + tile + 6), row["taxon_name"], fill="black", font=font)
        credits.append(f"- {row['taxon_name']} — {row['creator']} — {row['source_page']}")
    output = Path("assets/dataset-preview.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)
    Path("assets/dataset-preview-attribution.md").write_text("\n".join(credits) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(rows)} CC0 images")


if __name__ == "__main__":
    main()
