"""Fine-tune a pretrained backbone on the 110-species classifier, one arm, one seed.

Stage 2 of docs/species-classifier-plan.md: the frozen screen picked the
backbones, this trains them end to end. Same rule as train_species_classifier.py:
the arms differ ONLY in which image is read and how it is fitted to a square;
backbone, recipe, split and seed are held fixed within a comparison.

Arms (geometry matches screen_species_backbones.py at evaluation):
- crop         the segmentation crop, Resize + CenterCrop
- full         the full frame, Resize + CenterCrop, as in the ResNet-18 baseline
- full-square  the full frame squashed to a square, nothing cut away

The recipe is the reviewer's, not ResNet-18's. lr 3e-4 on every layer would
wreck pretrained ViT or CLIP weights, so:
- AdamW, lr on the top block, layer-wise decay below it, the new head at
  --head-lr-mult times the top-block lr
- linear warmup, then cosine to zero, stepped per iteration
- drop-path (timm models; open_clip's ViT has none), bf16 autocast, grad clip 1.0
- hue jitter OFF; pigment is a species character. Brightness, contrast and
  saturation jitter stay, as in the baseline.
Class weighting and label smoothing are the baseline's, so thin classes count
the same way in every run.

Model selection is on validation, as before. Rows whose split is "test" are
dropped before anything is read: the held-out test split is scored once, by a
separate script, after every configuration is fixed.

Preemption: the full state is written to last.pt after every epoch and picked up
on restart. Data order and augmentation are seeded per epoch, so a resumed run
sees the same batches it would have seen.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import time

import numpy as np
from PIL import Image

from screen_species_backbones import MODELS, class_weights, metrics

ARMS = ("crop", "full", "full-square")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--labels", type=pathlib.Path, required=True)
    p.add_argument("--sections", type=pathlib.Path, required=True)
    p.add_argument("--crop-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-5, help="Top block; lower blocks get less.")
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--head-lr-mult", type=float, default=10.0,
                   help="The head starts from random, so it gets a larger step than the top block.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test: a fixed random N images per split; outputs get a -smoke suffix.")
    return p.parse_args()


def build(tag: str, n_classes: int, drop_path: float, image_size: int, device: str):
    """Return (backbone + head, mean, std, parameter groups without lr set)."""
    import torch
    from torch import nn
    lib, name = MODELS[tag]
    if lib == "timm":
        import timm
        from timm.optim import param_groups_layer_decay
        kw = {"img_size": image_size} if "vit_" in name else {}
        if drop_path and not name.startswith("resnet"):
            kw["drop_path_rate"] = drop_path
        backbone = timm.create_model(name, pretrained=True, num_classes=0, **kw)
        cfg = timm.data.resolve_model_data_config(backbone)
        mean, std, dim = cfg["mean"], cfg["std"], backbone.num_features

        def groups(wd, decay):
            return param_groups_layer_decay(backbone, weight_decay=wd, layer_decay=decay,
                                            no_weight_decay_list=getattr(backbone, "no_weight_decay", set)())
    else:
        import open_clip
        clip, _, pre = open_clip.create_model_and_transforms(name)
        norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
        mean, std = tuple(norm.mean), tuple(norm.std)
        backbone = clip.visual  # the text tower is not needed once the head is learned
        dim = backbone.output_dim
        del clip

        def groups(wd, decay):
            return open_clip_groups(backbone, wd, decay)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(dim, n_classes)

        def forward(self, x):
            return self.head(self.backbone(x))

    net = Net().to(device)

    def param_groups(wd, decay, head_mult):
        gs = groups(wd, decay)
        gs.append({"params": [net.head.weight], "weight_decay": wd, "lr_scale": head_mult})
        gs.append({"params": [net.head.bias], "weight_decay": 0.0, "lr_scale": head_mult})
        return gs
    return net, mean, std, param_groups


def open_clip_groups(visual, wd, decay):
    """Layer-wise lr decay for an open_clip VisionTransformer, mirroring timm's:
    patch embedding and tokens are layer 0, resblock i is layer i+1, the final
    norm and projection sit at the top with scale 1."""
    n_blocks = len(visual.transformer.resblocks)
    top = n_blocks + 1
    buckets: dict[tuple[int, bool], list] = {}
    for name, p in visual.named_parameters():
        if not p.requires_grad:
            continue
        m = re.match(r"transformer\.resblocks\.(\d+)\.", name)
        if m:
            layer = int(m.group(1)) + 1
        elif name.startswith(("conv1", "class_embedding", "positional_embedding", "ln_pre")):
            layer = 0
        else:
            layer = top
        no_wd = p.ndim <= 1 or name in ("class_embedding", "positional_embedding")
        buckets.setdefault((layer, no_wd), []).append(p)
    return [{"params": ps, "weight_decay": 0.0 if no_wd else wd, "lr_scale": decay ** (top - layer)}
            for (layer, no_wd), ps in sorted(buckets.items())]


def main() -> int:
    args = parse_args()
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    labels = json.loads(args.labels.read_text())
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    sec_map = json.loads(args.sections.read_text())["species"]
    section_of = np.array([sec_map[lab]["section"] for lab in labels])

    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    unknown = {r["split"] for r in rows} - {"train", "validation", "test"}
    if unknown:
        raise SystemExit(f"unexpected split values {sorted(unknown)}")
    held_out = sum(r["split"] == "test" for r in rows)
    rows = [r for r in rows if r["split"] != "test"]
    for r in rows:
        r["_path"] = str(args.crop_dir / f"inat_{r['photo_id']}.jpg") if args.arm == "crop" else r["image"]
    # Every arm must see the same images, so check the crops even on a full-frame
    # arm: a photo without a crop would otherwise be in one arm and not the other.
    missing = [r["photo_id"] for r in rows
               if not (args.crop_dir / f"inat_{r['photo_id']}.jpg").exists() or not pathlib.Path(r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from an arm; refusing to run an unpaired comparison")

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    if args.limit:
        rng = np.random.default_rng(0)
        train = [train[i] for i in sorted(rng.choice(len(train), args.limit, replace=False))]
        val = [val[i] for i in sorted(rng.choice(len(val), args.limit, replace=False))]
    ty = np.array([index[r["label"]] for r in train]); vy = np.array([index[r["label"]] for r in val])
    print(f"model={args.model}  arm={args.arm}  seed={args.seed}  train={len(train)}  val={len(val)}  "
          f"test_held_out={held_out}  classes={n}  device={device}", flush=True)

    net, mean, std, param_groups = build(args.model, n, args.drop_path, args.image_size, device)
    s = args.image_size
    bicubic = transforms.InterpolationMode.BICUBIC
    norm = transforms.Normalize(mean, std)
    square = [transforms.Resize((s, s), interpolation=bicubic)] if args.arm == "full-square" else []
    train_tf = transforms.Compose(square + [
        transforms.RandomResizedCrop(s, scale=(0.7, 1.0), interpolation=bicubic),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.0),
        transforms.ToTensor(), norm,
    ])
    eval_tf = transforms.Compose(
        (square if square else [transforms.Resize(int(s * 1.14), interpolation=bicubic), transforms.CenterCrop(s)])
        + [transforms.ToTensor(), norm])

    class DS(Dataset):
        def __init__(self, rows, tf): self.rows, self.tf = rows, tf
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            with Image.open(r["_path"]) as im:
                return self.tf(im.convert("RGB")), index[r["label"]]

    g = torch.Generator()
    tl = DataLoader(DS(train, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, generator=g, persistent_workers=False)
    vl = DataLoader(DS(val, eval_tf), batch_size=args.batch_size * 2, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    groups = param_groups(args.weight_decay, args.layer_decay, args.head_lr_mult)
    for gr in groups:
        gr["lr"] = args.learning_rate * gr.pop("lr_scale", 1.0)
    opt = torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.999))
    steps_per_epoch = len(tl)
    total, warmup = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights(ty, n), device=device),
                                    label_smoothing=args.label_smoothing)
    autocast = lambda: torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")

    def evaluate():
        net.eval()
        out = []
        with torch.no_grad(), autocast():
            for x, _ in vl:
                out.append(net(x.to(device, non_blocking=True)).float().cpu().numpy())
        return np.concatenate(out)

    suffix = "-smoke" if args.limit else ""
    best, best_epoch, since, history, start_epoch = -1.0, -1, 0, [], 1
    last = args.output / f"last{suffix}.pt"
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        best, best_epoch, since, history = st["best"], st["best_epoch"], st["since"], st["history"]
        start_epoch = st["epoch"] + 1
        print(f"resumed after epoch {st['epoch']} (best {best:.4f} at {best_epoch})", flush=True)

    ids = np.array([r["photo_id"] for r in val])
    observers = np.array([r["observer_login"] for r in val])
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if since >= args.patience:
            break
        # Seeded per epoch so a resumed run replays the same order, augmentation and drop-path.
        g.manual_seed(args.seed * 1000 + epoch); torch.manual_seed(args.seed * 1000 + epoch)
        net.train()
        tot, t0 = 0.0, time.time()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast():
                loss = criterion(net(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * x.size(0)

        scores = evaluate()
        m = metrics(vy, scores, n, section_of)
        history.append({"epoch": epoch, "train_loss": tot / (steps_per_epoch * args.batch_size),
                        "val_balanced_accuracy": m["balanced_accuracy"], "val_accuracy": m["accuracy"],
                        "val_top5_accuracy": m["top5_accuracy"],
                        "val_section_balanced_accuracy": m["section_balanced_accuracy"],
                        "epoch_seconds": round(time.time() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)
        if m["balanced_accuracy"] > best:
            best, best_epoch, since = m["balanced_accuracy"], epoch, 0
            torch.save({"model_state": net.state_dict(), "model": args.model, "labels": labels,
                        "arm": args.arm, "seed": args.seed, "image_size": s, "epoch": epoch},
                       args.output / f"classifier-best{suffix}.pt")
            np.savez_compressed(args.output / f"predictions-best{suffix}.npz", val_photo_id=ids, val_label=vy,
                                val_observer=observers, scores=scores.astype(np.float32))
        else:
            since += 1
        np.savez_compressed(args.output / f"predictions-last{suffix}.npz", val_photo_id=ids, val_label=vy,
                            val_observer=observers, scores=scores.astype(np.float32))
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best": best, "best_epoch": best_epoch, "since": since, "history": history,
                    "epoch": epoch}, last.with_suffix(".tmp"))
        last.with_suffix(".tmp").replace(last)  # atomic, so preemption mid-write cannot corrupt it
    if since >= args.patience:
        print(f"early stop after epoch {history[-1]['epoch']}")

    (args.output / f"classifier-metrics{suffix}.json").write_text(json.dumps({
        "model": args.model, "checkpoint": MODELS[args.model][1], "arm": args.arm, "seed": args.seed,
        "train_samples": len(train), "validation_samples": len(val), "test_held_out": held_out,
        "recipe": {k: getattr(args, k) for k in ("image_size", "epochs", "warmup_epochs", "batch_size",
                   "learning_rate", "layer_decay", "head_lr_mult", "weight_decay", "drop_path",
                   "label_smoothing", "patience")},
        "epochs_run": len(history), "best_balanced_accuracy": best, "best_epoch": best_epoch,
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device == "cuda" else None,
        "last": history[-1], "elapsed_seconds_this_attempt": round(time.time() - started, 1),
        "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    last.unlink(missing_ok=True)
    peak = f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device == "cuda" else ""
    print(f"\nbest balanced accuracy {best:.4f} at epoch {best_epoch}{peak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
