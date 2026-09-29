"""Export the shipped species student to ONNX, quantise it to int8, fit its temperature.

Steps 2 and 3 of "Test protocol for the shipped student" in
docs/species-classifier-plan.md, on validation only (the held-out test split
is dropped with the other test rows before anything is read):

- export the checkpoint to ONNX, opset 17, batch 1, 224 px, and check ONNX
  Runtime's fp32 logits against PyTorch's on a few images;
- quantise it with onnxruntime.quantization.quantize_dynamic, QInt8 weights,
  as the bake-off's deployment tiebreaker defines. Dynamic: weights of MatMul
  and Gemm are stored as int8 (per tensor), activations are quantised on the
  fly per inference, and everything else (layer norm, softmax, GELU, the
  normalisation) stays fp32. No calibration data is used;
- score PyTorch fp32, ONNX fp32 and ONNX int8 on the full validation set.
  PyTorch fp32 must reproduce the checkpoint's own predictions-last.npz, which
  training wrote under bf16 autocast on a GPU: same photos and labels in the
  same order, top-1 agreement >= 0.98 and balanced accuracy within 0.005,
  or the script stops;
- the rule: int8 ships if it loses <= 0.01 balanced accuracy against fp32
  ONNX, else fp32 ships;
- one temperature for the shipped variant, grid 0.30-3.00 step 0.01 on
  validation NLL, ECE (15 bins) before and after, as diagnose_species_distill.py;
- latency of both variants: median of 200 runs after 20 warm-ups,
  CPUExecutionProvider, one thread, batch 1.

The graph input `pixels` is float32 1x3x224x224, RGB, NCHW, in [0, 1]
(torchvision ToTensor of the resized, centre-cropped image). The ImageNet
mean/std normalisation is INSIDE the graph, so a client only resizes, crops
and divides by 255. The output `logits` is 1x110 in labels.json order;
probabilities are softmax(logits / temperature).

Writes to --output: model-fp32.onnx, model-int8.onnx, labels.json,
val-logits.npz (all three variants, for the test-scoring step) and
release.json.

Usage: python scripts/export_species_onnx.py --checkpoint <classifier-last.pt> --output <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import statistics
import time

import numpy as np

from analyze_species_teacher import ece, softmax, tempered
from finetune_species_backbone import build, eval_transform
from screen_species_backbones import MODELS, metrics

BASE = pathlib.Path("/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-species-corpus")
MAX_INT8_LOSS = 0.01
GRID = np.round(np.arange(0.30, 3.001, 0.01), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--records", type=pathlib.Path, default=BASE / "split-110-test" / "species-records.jsonl")
    p.add_argument("--labels", type=pathlib.Path, default=BASE / "split-110-test" / "labels.json")
    p.add_argument("--sections", type=pathlib.Path, default=pathlib.Path("data/species-110-sections.json"))
    p.add_argument("--full-dir", type=pathlib.Path, default=BASE / "full-s576",
                   help="The cache the checkpoint was trained and validated on (CACHE=1).")
    p.add_argument("--threads", type=int, default=8, help="Threads for scoring; latency always uses one.")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    import onnx
    import onnxruntime as ort
    import timm
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.set_num_threads(args.threads)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text())
    n = len(labels)
    sec_map = json.loads(args.sections.read_text())["species"]
    section_of = np.array([sec_map[lab]["section"] for lab in labels])

    st = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if st["labels"] != labels:
        raise SystemExit("checkpoint labels differ from --labels")
    if st["arm"] != "full":
        raise SystemExit(f"arm {st['arm']}: this export reads the full-frame cache only")
    s = st["image_size"]
    net, mean, std, _ = build(st["model"], n, 0.0, s, "cpu")
    net.load_state_dict(st["model_state"])  # final saves use model_state; the resumable last.pt uses model
    net.eval()

    class Deployed(nn.Module):
        """The classifier with the normalisation in front, so the graph takes [0, 1] RGB."""
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, pixels):
            return self.net((pixels - self.mean) / self.std)

    model = Deployed().eval()

    # Validation rows only; test rows are dropped before any image is opened.
    rows = [json.loads(l) for l in open(args.records, encoding="utf-8") if l.strip()]
    val = [r for r in rows if r["split"] == "validation"]
    del rows
    index = {lab: i for i, lab in enumerate(labels)}
    y = np.array([index[r["label"]] for r in val])
    ids = np.array([r["photo_id"] for r in val])
    observers = np.array([r["observer_login"] for r in val])
    ref_path = args.checkpoint.parent / "predictions-last.npz"
    ref = np.load(ref_path)
    if not (np.array_equal(ref["val_photo_id"], ids) and np.array_equal(ref["val_label"], y)):
        raise SystemExit(f"{ref_path}: validation photos or labels differ from this run")
    tf = eval_transform("full", s, mean, std)
    tf.transforms = tf.transforms[:-1]  # Normalize lives in the graph

    class DS(Dataset):
        def __len__(self): return len(val)
        def __getitem__(self, i):
            with Image.open(args.full_dir / f"inat_{val[i]['photo_id']}.jpg") as im:
                return tf(im.convert("RGB"))

    # Export, then check ONNX Runtime fp32 against PyTorch fp32 on eight images.
    fp32, int8 = out / "model-fp32.onnx", out / "model-int8.onnx"
    probe = torch.stack([DS()[i] for i in range(8)])
    torch.onnx.export(model, probe[:1], str(fp32), opset_version=args.opset, input_names=["pixels"],
                      output_names=["logits"], do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(fp32)))
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    def session(path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    sess = {"fp32": session(fp32, args.threads), "int8": session(int8, args.threads)}
    with torch.no_grad():
        pt = model(probe).numpy()
    ox = np.concatenate([sess["fp32"].run(None, {"pixels": probe[i:i + 1].numpy()})[0] for i in range(8)])
    export_diff = float(np.abs(pt - ox).max())
    print(f"export check: max |ORT fp32 - PyTorch fp32| logit {export_diff:.2e} on 8 images", flush=True)
    if export_diff > 1e-3:
        raise SystemExit("ONNX fp32 does not match PyTorch fp32")

    # Full validation: PyTorch fp32 in batches, both ONNX variants at batch 1 as shipped.
    scores = {"torch_fp32": [], "onnx_fp32": [], "onnx_int8": []}
    t0 = time.time()
    for k, x in enumerate(DataLoader(DS(), batch_size=32, num_workers=args.workers)):
        with torch.no_grad():
            scores["torch_fp32"].append(model(x).numpy())
        for v in ("fp32", "int8"):
            scores[f"onnx_{v}"].append(np.concatenate(
                [sess[v].run(None, {"pixels": x[i:i + 1].numpy()})[0] for i in range(len(x))]))
        if k % 20 == 0:
            print(f"  {min((k + 1) * 32, len(val))}/{len(val)} images, {time.time() - t0:.0f} s", flush=True)
    scores = {k: np.concatenate(v).astype(np.float32) for k, v in scores.items()}
    res = {k: metrics(y, v, n, section_of) for k, v in scores.items()}
    recorded = metrics(y, ref["scores"], n, section_of)
    agree_ref = float(np.mean(scores["torch_fp32"].argmax(1) == ref["scores"].argmax(1)))
    print(f"PyTorch fp32 {res['torch_fp32']['balanced_accuracy']:.4f} against recorded (bf16, GPU) "
          f"{recorded['balanced_accuracy']:.4f}; top-1 agreement {agree_ref:.4f}", flush=True)
    if agree_ref < 0.98 or abs(res["torch_fp32"]["balanced_accuracy"] - recorded["balanced_accuracy"]) > 0.005:
        raise SystemExit("PyTorch fp32 does not reproduce predictions-last.npz; preprocessing or weights differ")
    loss = res["onnx_fp32"]["balanced_accuracy"] - res["onnx_int8"]["balanced_accuracy"]
    shipped = "int8" if loss <= MAX_INT8_LOSS else "fp32"
    for k, m in res.items():
        print(f"{k:11s} balanced {m['balanced_accuracy']:.4f}  accuracy {m['accuracy']:.4f}  "
              f"top-5 {m['top5_accuracy']:.4f}", flush=True)
    print(f"int8 loses {loss:+.4f} balanced accuracy against fp32 ONNX; ships: {shipped}", flush=True)

    # Temperature for the shipped variant, as diagnose_species_distill.py fits it.
    p = softmax(scores[f"onnx_{shipped}"].astype(np.float64))
    logp = np.log(np.clip(p, 1e-12, None))
    nll = [-np.mean(np.log(np.clip(tempered(logp, g)[np.arange(len(y)), y], 1e-12, None))) for g in GRID]
    temp = float(GRID[int(np.argmin(nll))])
    pt_ = tempered(logp, temp)
    calib = {"temperature": temp, "grid": [0.30, 3.00, 0.01], "nll_before": float(nll[list(GRID).index(1.0)]),
             "nll_after": float(min(nll)), "ece_before": ece(p, y), "ece_after": ece(pt_, y),
             "mean_confidence_before": float(p.max(1).mean()), "mean_confidence_after": float(pt_.max(1).mean()),
             "at_grid_edge": temp in (GRID[0], GRID[-1])}
    print(f"temperature {temp:.2f}: ECE {calib['ece_before']:.4f} -> {calib['ece_after']:.4f}", flush=True)

    # Latency: one thread, batch 1, 20 warm-ups then the median of 200.
    latency = {}
    for v, path in (("fp32", fp32), ("int8", int8)):
        one = session(path, 1)
        x = probe[:1].numpy()
        for _ in range(20):
            one.run(None, {"pixels": x})
        ts = []
        for _ in range(200):
            a = time.perf_counter(); one.run(None, {"pixels": x}); ts.append((time.perf_counter() - a) * 1e3)
        latency[v] = {"median_ms": statistics.median(ts), "p10_ms": float(np.percentile(ts, 10)),
                      "p90_ms": float(np.percentile(ts, 90))}
        print(f"latency {v}: median {latency[v]['median_ms']:.1f} ms", flush=True)

    shutil.copyfile(args.labels, out / "labels.json")
    np.savez_compressed(out / "val-logits.npz", val_photo_id=ids, val_label=y, val_observer=observers,
                        **{f"scores_{k}": v for k, v in scores.items()})
    ckpt_metrics = json.loads((args.checkpoint.parent / "classifier-metrics.json").read_text())
    import platform, socket
    release = {
        "model": st["model"], "backbone": MODELS[st["model"]][1], "arm": st["arm"], "seed": st["seed"],
        "epoch": st["epoch"], "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "shipped": shipped, "shipped_file": f"model-{shipped}.onnx",
        "files": {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in (fp32, int8, out / "labels.json")},
        "input": {"name": "pixels", "shape": [1, 3, s, s], "dtype": "float32", "layout": "NCHW", "channels": "RGB",
                  "range": "[0, 1] (pixel / 255)", "normalisation_in_graph": True, "mean": list(mean), "std": list(std),
                  "resize": f"short side to {int(s * 1.14)} px, bicubic, then centre crop {s} px",
                  "training_source": f"full frames pre-shrunk to short side 576 px (LANCZOS), {args.full_dir}",
                  "exif_rotation": "none (training ignored EXIF orientation)"},
        "output": {"name": "logits", "shape": [1, n], "classes": "labels.json order",
                   "probabilities": "softmax(logits / temperature)"},
        "opset": args.opset,
        "quantisation": {"method": "onnxruntime.quantization.quantize_dynamic", "weight_type": "QInt8",
                         "per_channel": False, "calibration": None},
        "temperature": temp, "calibration": calib,
        "validation": {"images": len(val), **res, "recorded_bf16_gpu": recorded,
                       "torch_vs_recorded_top1_agreement": agree_ref,
                       "classifier_metrics_last": ckpt_metrics["last"]["val_balanced_accuracy"],
                       "int8_minus_fp32_balanced_accuracy": -loss,
                       "int8_vs_fp32_top1_agreement": float(np.mean(scores["onnx_int8"].argmax(1) == scores["onnx_fp32"].argmax(1))),
                       "int8_vs_fp32_max_abs_logit": float(np.abs(scores["onnx_int8"] - scores["onnx_fp32"]).max()),
                       "onnx_vs_torch_fp32_max_abs_logit": float(np.abs(scores["onnx_fp32"] - scores["torch_fp32"]).max())},
        "rule": {"max_int8_loss": MAX_INT8_LOSS, "int8_loss": loss, "fires_fp32": shipped == "fp32"},
        "export_check_max_abs_logit": export_diff,
        "latency": {"provider": "CPUExecutionProvider", "threads": 1, "batch": 1, "warmup": 20, "runs": 200,
                    "host": socket.gethostname(), "cpu": platform.processor() or None, **latency},
        "versions": {"torch": torch.__version__, "timm": timm.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "numpy": np.__version__},
        "test_split_read": False,
    }
    try:
        release["latency"]["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    (out / "release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'release.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
