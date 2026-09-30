"""Export the plant-outline U-Net to ONNX for the site, and check it against PyTorch.

docs/reports/segmentation-site-prereg.md, step 4. The graph takes `pixels`
float32 [1, 3, S, S] in [0, 1] (the whole photo resized to S x S, bilinear,
stretched, as in training) with the ImageNet normalisation inside, and returns
`prob` [1, 1, S, S] = sigmoid(logits); the site thresholds it.

Exports fp32 and a dynamic int8 copy (onnxruntime quantize_dynamic, QInt8).
Each is compared with PyTorch on the 19 validation and 17 field-eval images:
a variant passes only if its mask agrees with PyTorch's at IoU >= 0.99 on every
image, at the shipped threshold. Also times one image on one CPU thread.

Run from the repo root:
  PYTHONPATH=src python scripts/export_outline_onnx.py --checkpoint <unet-resnet34-best.pt> \
      --threshold 0.6 --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sundew_segmentation.baseline import binary_metrics

BASE = Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor")
IMAGES = [*sorted((BASE / "field-train-root/images/validation").glob("*.jpg")),
          *sorted((BASE / "field-probe-eval/images/field-eval").glob("*.jpg"))]
PARITY_BAR = 0.99


class Outline(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pixels):
        return torch.sigmoid(self.model((pixels - self.mean) / self.std))


def pixels(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)[None]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    import importlib.util
    spec = importlib.util.spec_from_file_location("s", Path(__file__).with_name("score_field_unet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model, cfg = mod.load(args.checkpoint)
    size = cfg.image_size
    wrapped = Outline(model).eval()

    fp32, int8 = args.out / "outline-fp32.onnx", args.out / "outline-int8.onnx"
    probe = torch.from_numpy(pixels(IMAGES[0], size))
    torch.onnx.export(wrapped, probe, str(fp32), opset_version=args.opset, input_names=["pixels"],
                      output_names=["prob"], do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sessions = {p.stem: ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"]) for p in (fp32, int8)}
    per_image = {k: {} for k in sessions}
    max_diff = {k: 0.0 for k in sessions}
    for path in IMAGES:
        x = pixels(path, size)
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(x))[0, 0].numpy()
        for k, s in sessions.items():
            got = s.run(None, {"pixels": x})[0][0, 0]
            max_diff[k] = max(max_diff[k], float(np.abs(got - ref).max()))
            a, b = got >= args.threshold, ref >= args.threshold
            per_image[k][path.name] = 1.0 if not (a.any() or b.any()) else float(binary_metrics(a, b)["iou"])

    timing = {}
    x = pixels(IMAGES[0], size)
    for k, s in sessions.items():
        s.run(None, {"pixels": x})
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            s.run(None, {"pixels": x})
            ts.append(time.perf_counter() - t0)
        timing[k] = float(np.median(ts))

    report = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "image_size": size, "threshold": args.threshold, "opset": args.opset, "parity_bar": PARITY_BAR,
              "images": len(IMAGES), "variants": {}}
    for k, path in (("outline-fp32", fp32), ("outline-int8", int8)):
        ious = per_image[k]
        report["variants"][path.name] = {
            "sha256": sha256(path), "bytes": path.stat().st_size, "max_prob_diff": max_diff[k],
            "min_mask_iou": min(ious.values()), "mean_mask_iou": float(np.mean(list(ious.values()))),
            "passes": min(ious.values()) >= PARITY_BAR, "ms_one_thread": round(1000 * timing[k]),
            "per_image_mask_iou": ious}
        v = report["variants"][path.name]
        print(f"{path.name}: {v['bytes'] / 1e6:.1f} MB, min mask IoU {v['min_mask_iou']:.4f}, "
              f"max prob diff {v['max_prob_diff']:.2e}, {v['ms_one_thread']} ms/image (1 thread) -> "
              f"{'PASS' if v['passes'] else 'FAIL'}")
    (args.out / "export-report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
