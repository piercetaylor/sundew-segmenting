"""Frozen-backbone screen for the 110-species classifier.

One pretrained backbone per invocation, weights frozen. Extract one embedding per
image, then score three cheap readouts on the validation split:

- linear  class-weighted multinomial logistic regression on standardised
          features; the L2 strength is picked by observer-grouped 3-fold
          cross-validation on TRAIN ONLY, so validation is touched once
- knn     cosine k=10, votes weighted exp(sim/0.07) as in DINO; no tuning
- zeroshot  "a photo of Drosera <epithet>." (open_clip models only)

The question is which backbone is worth a full fine-tune, not what the final
accuracy will be. A frozen probe is a floor on what fine-tuning reaches, so a
backbone that beats the fine-tuned ResNet-18 while frozen is a clear candidate.

Resolution is held at 224 for every model, including those pretrained larger,
so the screen varies the backbone and nothing else. Resolution is a later,
separate experiment. See docs/species-classifier-plan.md.

Arms, all read at 224:
- crop         the segmentation crop, Resize(255) + CenterCrop(224), as trained
- full         the full frame, same geometry as the fine-tuned full arm
- full-square  the full frame squashed to 224x224, nothing cut away; the
               untested control from reports/species-crop-comparison.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import zlib

import numpy as np
from PIL import Image

# tag -> (library, checkpoint). The tag is the directory name and report label.
MODELS = {
    "resnet18-in1k": ("timm", "resnet18.tv_in1k"),
    "convnext-b-in22k": ("timm", "convnext_base.fb_in22k_ft_in1k_384"),
    "dinov2-b": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "dinov2-l-reg": ("timm", "vit_large_patch14_reg4_dinov2.lvd142m"),
    "dinov3-vit-b": ("timm", "vit_base_patch16_dinov3.lvd1689m"),
    "dinov3-convnext-b": ("timm", "convnext_base.dinov3_lvd1689m"),
    "siglip2-so400m": ("timm", "vit_so400m_patch14_siglip_378.v2_webli"),
    "bioclip": ("open_clip", "hf-hub:imageomics/bioclip"),
    "bioclip-2": ("open_clip", "hf-hub:imageomics/bioclip-2"),
    # Small backbones, candidates for an on-device (browser / phone) student.
    # Added 2026-09-24 after the main screen; see the small-model section of
    # docs/species-classifier-plan.md.
    "dinov2-s": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2-s-reg": ("timm", "vit_small_patch14_reg4_dinov2.lvd142m"),
    "dinov3-vit-s": ("timm", "vit_small_patch16_dinov3.lvd1689m"),
    "dinov3-vit-s-plus": ("timm", "vit_small_plus_patch16_dinov3.lvd1689m"),
    "tinyvit-21m-in22k": ("timm", "tiny_vit_21m_224.dist_in22k"),
    "convnext-nano-in12k": ("timm", "convnext_nano.in12k"),
    "convnext-t-in22k": ("timm", "convnext_tiny.fb_in22k"),
    "mobilenetv4-conv-m-in12k": ("timm", "mobilenetv4_conv_medium.e250_r384_in12k"),
    "mobilenetv4-hybrid-m-in12k": ("timm", "mobilenetv4_hybrid_medium.e200_r256_in12k"),
    "efficientnetv2-s-in21k": ("timm", "tf_efficientnetv2_s.in21k"),
    "mobileclip2-s2": ("timm", "fastvit_mci2.apple_mclip2_dfndr2b"),
}
ARMS = ("crop", "full", "full-square")
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--records", type=pathlib.Path, required=True)
    p.add_argument("--labels", type=pathlib.Path, required=True)
    p.add_argument("--sections", type=pathlib.Path, required=True)
    p.add_argument("--crop-dir", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True,
                   help="Per-model directory; features are cached here so a preempted task resumes.")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: a fixed random N images per split; writes results-smoke.json.")
    return p.parse_args()


def load_backbone(tag: str, device: str):
    """Return (image -> embedding module, mean, std, text encoder or None)."""
    import torch
    lib, name = MODELS[tag]
    if lib == "timm":
        import timm
        # Plain ViTs only; tiny_vit_ and fastvit_ also contain "vit_".
        kw = {"img_size": IMAGE_SIZE} if name.startswith("vit_") else {}
        model = timm.create_model(name, pretrained=True, num_classes=0, **kw)
        cfg = timm.data.resolve_model_data_config(model)
        return model.eval().to(device), cfg["mean"], cfg["std"], None
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(name)
    norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
    tokenizer = open_clip.get_tokenizer(name)
    model = model.eval().to(device)

    class Image_(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m.encode_image(x)

    def encode_text(prompts):
        with torch.no_grad():
            return model.encode_text(tokenizer(prompts).to(device)).float()
    return Image_(model), tuple(norm.mean), tuple(norm.std), encode_text


def extract(model, rows, arm, mean, std, args, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    bicubic = transforms.InterpolationMode.BICUBIC
    geometry = ([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=bicubic)] if arm == "full-square"
                else [transforms.Resize(int(IMAGE_SIZE * 1.14), interpolation=bicubic), transforms.CenterCrop(IMAGE_SIZE)])
    tf = transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize(mean, std)])

    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            path = rows[i]["_crop"] if arm == "crop" else rows[i]["image"]
            with Image.open(path) as im:
                return tf(im.convert("RGB"))

    out = []
    loader = DataLoader(DS(), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for x in loader:
            out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def fit_logreg(x, y, weights, lam, n_classes, device):
    """Class-weighted multinomial logistic regression, full batch L-BFGS."""
    import torch
    x = torch.as_tensor(x, device=device); y = torch.as_tensor(y, device=device)
    w = torch.zeros(x.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    ce = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(weights, device=device))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ce(x @ w + b, y) + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def logits(x, wb, device):
    import torch
    w, b = wb
    return (torch.as_tensor(x, device=device) @ w + b).cpu().numpy()


def balanced_accuracy(y, pred, n_classes) -> float:
    per = [np.mean(pred[y == c] == c) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(per))


def class_weights(y, n_classes) -> np.ndarray:
    # Same formula as train_species_classifier.py, so frozen and fine-tuned runs
    # weight the thin classes identically.
    counts = np.bincount(y, minlength=n_classes)
    return (len(y) / (n_classes * np.maximum(counts, 1))).astype(np.float32)


def observer_folds(observers, k=3) -> np.ndarray:
    # Stable across runs and models: fold = crc32(observer) mod k.
    return np.array([zlib.crc32(o.encode()) % k for o in observers])


def knn_scores(train_x, train_y, val_x, n_classes, device, k=10, temperature=0.07):
    import torch
    tx = torch.nn.functional.normalize(torch.as_tensor(train_x, device=device), dim=1)
    vx = torch.nn.functional.normalize(torch.as_tensor(val_x, device=device), dim=1)
    ty = torch.as_tensor(train_y, device=device)
    scores = torch.zeros(len(vx), n_classes, device=device)
    for i in range(0, len(vx), 1024):
        sim, idx = (vx[i:i + 1024] @ tx.T).topk(k, dim=1)
        scores[i:i + 1024].scatter_add_(1, ty[idx], (sim / temperature).exp())
    return scores.cpu().numpy()


def metrics(y, scores, n_classes, section_of) -> dict:
    top5 = np.argsort(-scores, axis=1)[:, :5]
    pred = top5[:, 0]
    sec_hit = np.array([section_of[p] == section_of[t] for p, t in zip(pred, y)])
    return {
        "balanced_accuracy": balanced_accuracy(y, pred, n_classes),
        "accuracy": float(np.mean(pred == y)),
        "top5_accuracy": float(np.mean([t in row for t, row in zip(y, top5)])),
        # Averaged per true species, like balanced accuracy, so the common
        # species do not decide it.
        "section_balanced_accuracy": float(np.mean(
            [sec_hit[y == c].mean() for c in range(n_classes) if np.any(y == c)])),
    }


def main() -> int:
    args = parse_args()
    import torch
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    labels = json.loads(args.labels.read_text())
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    sec_map = json.loads(args.sections.read_text())["species"]
    section_of = np.array([sec_map[lab]["section"] for lab in labels])

    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    for r in rows:
        r["_crop"] = str(args.crop_dir / f"inat_{r['photo_id']}.jpg")
    missing = [r["photo_id"] for r in rows if not pathlib.Path(r["_crop"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} crops missing; every arm must see the same images")
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    if args.limit:
        # Records are sorted by species, so a head slice is a single class.
        rng = np.random.default_rng(0)
        train = [train[i] for i in sorted(rng.choice(len(train), args.limit, replace=False))]
        val = [val[i] for i in sorted(rng.choice(len(val), args.limit, replace=False))]
    ty = np.array([index[r["label"]] for r in train]); vy = np.array([index[r["label"]] for r in val])
    folds = observer_folds([r["observer_login"] for r in train])
    cw = class_weights(ty, n)
    print(f"model={args.model}  {MODELS[args.model][1]}  train={len(train)}  val={len(val)}  "
          f"classes={n}  device={device}", flush=True)

    model, mean, std, encode_text = load_backbone(args.model, device)
    # Image tower only; a CLIP model's text tower does not see the photograph.
    params = sum(p.numel() for p in getattr(getattr(model, "m", None), "visual", model).parameters())
    text = None
    if encode_text is not None:
        text = torch.nn.functional.normalize(
            encode_text([f"a photo of {lab}." for lab in labels]), dim=1).cpu().numpy()

    results = {"model": args.model, "checkpoint": MODELS[args.model][1], "library": MODELS[args.model][0],
               "parameters": params, "image_size": IMAGE_SIZE, "train": len(train), "validation": len(val),
               "lambdas": LAMBDAS, "arms": {}}
    predictions = {"val_photo_id": np.array([r["photo_id"] for r in val]),
                   "val_label": vy, "val_observer": np.array([r["observer_login"] for r in val])}
    for arm in args.arms:
        start = time.time()
        cache = args.output / f"features-{arm}.npz"
        if cache.exists() and not args.limit:
            z = np.load(cache)
            tx, vx = z["train"], z["validation"]
            print(f"[{arm}] features loaded from cache", flush=True)
        else:
            tx = extract(model, train, arm, mean, std, args, device)
            vx = extract(model, val, arm, mean, std, args, device)
            if not args.limit:
                np.savez(cache, train=tx.astype(np.float16), validation=vx.astype(np.float16),
                         train_photo_id=[r["photo_id"] for r in train],
                         validation_photo_id=[r["photo_id"] for r in val])
        extract_s = time.time() - start

        mu, sd = tx.mean(0), tx.std(0) + 1e-6
        txs, vxs = (tx - mu) / sd, (vx - mu) / sd
        cv = {}
        for lam in LAMBDAS:
            accs = []
            for f in range(3):
                tr, te = folds != f, folds == f
                wb = fit_logreg(txs[tr], ty[tr], class_weights(ty[tr], n), lam, n, device)
                accs.append(balanced_accuracy(ty[te], logits(txs[te], wb, device).argmax(1), n))
            cv[lam] = float(np.mean(accs))
        best_lam = max(cv, key=cv.get)
        wb = fit_logreg(txs, ty, cw, best_lam, n, device)
        lin = logits(vxs, wb, device)
        knn = knn_scores(tx, ty, vx, n, device)
        arm_res = {"extract_seconds": round(extract_s, 1), "feature_dim": int(tx.shape[1]),
                   "cv_balanced_accuracy": {str(k): v for k, v in cv.items()}, "chosen_lambda": best_lam,
                   "linear": metrics(vy, lin, n, section_of), "knn": metrics(vy, knn, n, section_of)}
        predictions[f"{arm}/linear"] = lin.astype(np.float32)
        predictions[f"{arm}/knn"] = knn.astype(np.float32)
        if text is not None:
            vn = vx / np.linalg.norm(vx, axis=1, keepdims=True)
            zs = vn @ text.T
            arm_res["zeroshot"] = metrics(vy, zs, n, section_of)
            predictions[f"{arm}/zeroshot"] = zs.astype(np.float32)
        results["arms"][arm] = arm_res
        print(f"[{arm}] " + json.dumps({k: v for k, v in arm_res.items() if k != "cv_balanced_accuracy"}), flush=True)

    suffix = "-smoke" if args.limit else ""
    (args.output / f"results{suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez_compressed(args.output / f"predictions{suffix}.npz", **predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
