"""Curation and dataset packaging utilities."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping
import json

from PIL import Image, ImageOps


def split_for_observer(observer_login: str, seed: str = "sundew-2026") -> str:
    """Assign all images from one observer to a stable dataset split."""
    digest = sha256(f"{seed}:{observer_login}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def select_core_set(
    rows: Iterable[Mapping[str, Any]],
    rejected_indices: set[int],
    target_count: int = 250,
    max_per_species: int = 25,
    max_per_observer: int = 8,
    seed: int = 20260916,
) -> tuple[list[int], list[int]]:
    """Select a diverse core set from images that passed the visual screen."""
    indexed = [(index, dict(row)) for index, row in enumerate(rows) if index not in rejected_indices]
    Random(seed).shuffle(indexed)
    species_counts: Counter[str] = Counter()
    observer_counts: Counter[str] = Counter()
    selected: list[int] = []
    reserve: list[int] = []

    for index, row in indexed:
        species = str(row.get("taxon_name") or "Drosera")
        observer = str(row.get("observer_login") or "unknown")
        if (
            len(selected) < target_count
            and species_counts[species] < max_per_species
            and observer_counts[observer] < max_per_observer
        ):
            selected.append(index)
            species_counts[species] += 1
            observer_counts[observer] += 1
        else:
            reserve.append(index)

    if len(selected) < target_count:
        needed = target_count - len(selected)
        selected.extend(reserve[:needed])
        reserve = reserve[needed:]

    return sorted(selected), sorted(reserve)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def normalize_image(source: Path, destination: Path, max_side: int = 1600) -> tuple[int, int]:
    """Create a consistently oriented RGB JPEG for annotation and training."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=92, optimize=True)
        return image.size
