"""Train the species classifier on full frames or on segmentation crops.

The two arms differ ONLY in which image file is read. Same architecture, same
observer-grouped split, same augmentation, same schedule, same seed. That is the
whole design: reports/field-compare.md showed how easily an uncontrolled flag
(growth-form balancing, worth +0.052 on its own) can be mistaken for the effect
under study.

The question is whether the crop earns its place. If it does not, the crop
failure rate measured in step 1 stops being a risk to the species project.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
from collections import Counter

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--labels", type=pathlib.Path, required=True)
    p.add_argument("--arm", choices=("full", "crop"), required=True)
    p.add_argument("--crop-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=7)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    labels = json.loads(args.labels.read_text())
    index = {lab: i for i, lab in enumerate(labels)}
    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]

    def resolve(row) -> pathlib.Path | None:
        if args.arm == "full":
            return pathlib.Path(row["image"])
        c = args.crop_dir / f"inat_{row['photo_id']}.jpg"
        return c if c.exists() else None

    kept, missing = [], 0
    for r in rows:
        p = resolve(r)
        if p is None or not p.exists():
            missing += 1
            continue
        r["_path"] = str(p)
        kept.append(r)
    if missing:
        # Both arms must see the same images or the comparison is not paired.
        raise SystemExit(f"{missing} images unavailable for arm '{args.arm}'; refusing to run an unpaired comparison")

    train = [r for r in kept if r["split"] == "train"]
    val = [r for r in kept if r["split"] == "validation"]
    print(f"arm={args.arm}  train={len(train)}  val={len(val)}  classes={len(labels)}")

    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.image_size * 1.14)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), norm,
    ])

    class DS(Dataset):
        def __init__(self, rows, tf): self.rows, self.tf = rows, tf
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            with Image.open(r["_path"]) as im:
                x = self.tf(im.convert("RGB"))
            return x, index[r["label"]]

    g = torch.Generator(); g.manual_seed(args.seed)
    tl = DataLoader(DS(train, train_tf), batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True, generator=g)
    vl = DataLoader(DS(val, eval_tf), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model = model.to(device)

    # Class weights: the split is near-balanced but not exactly, and a species
    # advantage from frequency would confound the arm comparison.
    counts = Counter(r["label"] for r in train)
    w = torch.tensor([len(train) / (len(labels) * counts[lab]) for lab in labels],
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_epoch, history, since = 0.0, -1, [], 0
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        sched.step()

        model.eval()
        correct = 0
        per_class = Counter(); per_class_ok = Counter()
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device, non_blocking=True)
                pred = model(x).argmax(1).cpu()
                correct += (pred == y).sum().item()
                for t, p_ in zip(y.tolist(), pred.tolist()):
                    per_class[t] += 1; per_class_ok[t] += int(t == p_)
        acc = correct / len(val)
        balanced = float(np.mean([per_class_ok[i] / per_class[i] for i in per_class]))
        history.append({"epoch": epoch, "train_loss": tot / len(train),
                        "val_accuracy": acc, "val_balanced_accuracy": balanced})
        print(json.dumps(history[-1]), flush=True)
        if balanced > best:
            best, best_epoch, since = balanced, epoch, 0
            torch.save({"model_state": model.state_dict(), "labels": labels,
                        "arm": args.arm, "seed": args.seed},
                       args.output / "classifier-best.pt")
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at epoch {epoch}"); break

    (args.output / "classifier-metrics.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "classes": labels,
        "train_samples": len(train), "validation_samples": len(val),
        "image_size": args.image_size, "epochs_run": len(history),
        "best_balanced_accuracy": best, "best_epoch": best_epoch,
        "best_accuracy": max(h["val_accuracy"] for h in history),
        "elapsed_seconds": time.time() - start, "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nbest balanced accuracy {best:.4f} at epoch {best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
