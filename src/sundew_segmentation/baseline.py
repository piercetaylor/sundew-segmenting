"""Small, reproducible baseline utilities for the sundew mask task.

The module keeps evaluation independent of a deep-learning framework so dataset
work can proceed on a CPU before annotated masks are available. Optional model
names document the planned comparison and are resolved only when torch is
installed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from typing import Any, Iterable
from pathlib import Path
import json
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    architecture: str
    encoder: str
    image_size: int = 768
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


UNET_RESNET34 = BaselineConfig("unet-resnet34", "UNet", "resnet34")
SEGFORMER_B0 = BaselineConfig("segformer-b0", "SegFormer", "mit_b0")


def paired_samples(image_root: Path, mask_root: Path, split: str, *, require_masks: bool = True) -> list[dict[str, Path | str]]:
    """Index ``images/<split>/*.jpg`` with matching ``masks/<split>/<stem>.png``.

    Training callers should keep ``require_masks=True``: missing masks are a
    hard error so an accidental image-only run cannot produce a misleading model.
    """
    images = sorted((image_root / split).glob("*.jpg"))
    samples, missing = [], []
    for image in images:
        mask = mask_root / split / f"{image.stem}.png"
        if not mask.exists():
            missing.append(image.name)
            continue
        with Image.open(image) as source, Image.open(mask) as target:
            if source.size != target.size:
                raise ValueError(f"image/mask size mismatch for {image.name}: {source.size} != {target.size}")
        samples.append({"split": split, "image": image, "mask": mask})
    if require_masks and missing:
        raise FileNotFoundError(f"{len(missing)} masks missing for split '{split}' (first: {missing[0]})")
    return samples


def require_training_data(image_root: Path, mask_root: Path, splits: tuple[str, ...] = ("train", "validation")) -> dict[str, list[dict[str, Path | str]]]:
    """Return complete training pairs or fail before any model is constructed."""
    result = {split: paired_samples(image_root, mask_root, split) for split in splits}
    if not any(result.values()):
        raise FileNotFoundError("no reviewed masks found; annotation must be completed before training")
    return result


def add_growth_form_metadata(
    samples: list[dict[str, Path | str]], metadata_path: Path
) -> list[dict[str, Path | str]]:
    """Attach growth-form labels to image/mask pairs using the curated manifest."""
    by_name: dict[str, str] = {}
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_name[Path(row["curated_path"]).name] = str(row.get("growth_form") or "unknown")
    enriched = []
    for sample in samples:
        item = dict(sample)
        item["growth_form"] = by_name.get(Path(sample["image"]).name, "unknown")
        enriched.append(item)
    return enriched


def inverse_frequency_weights(labels: Iterable[str]) -> list[float]:
    """Return per-sample weights that give each represented group equal mass."""
    values = list(labels)
    counts = Counter(values)
    return [1.0 / counts[value] for value in values]


def binary_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Return IoU, Dice, precision, recall, and pixel accuracy."""
    pred = np.asarray(prediction) >= threshold
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} != {truth.shape}")
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    tn = float(np.logical_and(~pred, ~truth).sum())
    return {
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "pixel_accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def resolve_model(config: BaselineConfig, num_classes: int = 1, *, pretrained: bool = True):
    """Create a segmentation model, importing the optional framework on demand."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError("Install optional baseline dependencies to build models: pip install -e '.[baseline]'") from exc
    encoder_weights = "imagenet" if pretrained else None
    if config.architecture == "UNet":
        return smp.Unet(
            encoder_name=config.encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    if config.architecture == "SegFormer":
        return smp.Segformer(
            encoder_name=config.encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    raise ValueError(f"unknown architecture: {config.architecture}")
