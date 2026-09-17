"""Train a reproducible binary sundew segmentation baseline.

This entry point deliberately fails before importing the model stack when
reviewed masks are absent. Expected masks are PNG files under
``data/curated/masks/<split>/<image-stem>.png``.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from sundew_segmentation.baseline import (
    SEGFORMER_B0,
    UNET_RESNET34,
    require_training_data,
    resolve_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet-resnet34", "segformer-b0"), default="unet-resnet34")
    parser.add_argument("--images", type=Path, default=Path("data/curated/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/curated/masks"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        pairs = require_training_data(args.images, args.masks)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training data check failed: {error}") from error

    try:
        import torch
        import torch.nn.functional as functional
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("Install the baseline dependencies with: pip install -e '.[baseline]'") from error

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    class MaskDataset(Dataset):
        def __init__(self, samples, augment: bool):
            self.samples = samples
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            with Image.open(sample["image"]) as source:
                image = source.convert("RGB").resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
            with Image.open(sample["mask"]) as source:
                mask = source.convert("L").resize((args.image_size, args.image_size), Image.Resampling.NEAREST)
            image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)[None, ...]
            image_tensor = torch.from_numpy(image_array.copy())
            mask_tensor = torch.from_numpy(mask_array.copy())
            if self.augment and torch.rand(()) < 0.5:
                image_tensor = image_tensor.flip(-1)
                mask_tensor = mask_tensor.flip(-1)
            if self.augment and torch.rand(()) < 0.5:
                image_tensor = image_tensor.flip(-2)
                mask_tensor = mask_tensor.flip(-2)
            mean = torch.tensor((0.485, 0.456, 0.406), dtype=image_tensor.dtype)[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225), dtype=image_tensor.dtype)[:, None, None]
            image_tensor = (image_tensor - mean) / std
            return image_tensor, mask_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = UNET_RESNET34 if args.model == "unet-resnet34" else SEGFORMER_B0
    config = replace(
        config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
    )
    train_loader = DataLoader(
        MaskDataset(pairs["train"], augment=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        MaskDataset(pairs["validation"], augment=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = resolve_model(config, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    def loss_function(logits, targets):
        bce = functional.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return bce + dice_loss

    def validate():
        model.eval()
        totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        loss_total = 0.0
        with torch.no_grad():
            for images, targets in validation_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss_total += float(loss_function(logits, targets)) * images.shape[0]
                predictions = torch.sigmoid(logits) >= 0.5
                truth = targets >= 0.5
                totals["tp"] += int((predictions & truth).sum())
                totals["fp"] += int((predictions & ~truth).sum())
                totals["fn"] += int((~predictions & truth).sum())
                totals["tn"] += int((~predictions & ~truth).sum())
        tp, fp, fn, tn = (totals[key] for key in ("tp", "fp", "fn", "tn"))
        return {
            "loss": loss_total / len(validation_loader.dataset),
            "iou": tp / max(tp + fp + fn, 1),
            "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / f"{config.name}-best.pt"
    report_path = args.output / f"{config.name}-metrics.json"
    history = []
    best_iou = -1.0
    stale_epochs = 0
    started = perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * images.shape[0]
        metrics = validate()
        metrics.update(
            epoch=epoch,
            train_loss=train_loss / len(train_loader.dataset),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True))
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            stale_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "config": config.to_dict(), "metrics": metrics},
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            break

    report = {
        "config": config.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "weight_decay": args.weight_decay,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "elapsed_seconds": perf_counter() - started,
        "best_iou": best_iou,
        "checkpoint": checkpoint_path.as_posix(),
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {checkpoint_path} and {report_path}")


if __name__ == "__main__":
    main()
