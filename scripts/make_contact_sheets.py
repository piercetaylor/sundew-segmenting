from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for image curation.")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/inaturalist/metadata.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/contact_sheets"))
    parser.add_argument("--page-size", type=int, default=80)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=192)
    parser.add_argument("--font-size", type=int, default=12)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    rows = load_rows(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", args.font_size)
    except OSError:
        font = ImageFont.load_default()
    label_height = max(52, args.font_size * 3)
    cell_width = args.thumb_size
    cell_height = args.thumb_size + label_height

    for page_index, start in enumerate(range(0, len(rows), args.page_size), start=1):
        page_rows = rows[start : start + args.page_size]
        page_row_count = (len(page_rows) + args.columns - 1) // args.columns
        sheet = Image.new("RGB", (cell_width * args.columns, cell_height * page_row_count), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page_rows):
            image_path = Path(row["local_path"])
            with Image.open(image_path) as source:
                tile = ImageOps.fit(source.convert("RGB"), (args.thumb_size, args.thumb_size))
            x = (index % args.columns) * cell_width
            y = (index // args.columns) * cell_height
            sheet.paste(tile, (x, y))
            global_index = start + index
            label = (
                f"#{global_index:03d}  {row['photo_id']}  {row['license_code']}\n"
                f"{row['taxon_name'][:28]}"
            )
            draw.multiline_text((x + 3, y + args.thumb_size + 3), label, fill="black", font=font, spacing=2)
        sheet.save(args.output / f"contact_sheet_{page_index:02d}.jpg", quality=88)

    print(f"Created {(len(rows) + args.page_size - 1) // args.page_size} contact sheets for {len(rows)} images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
