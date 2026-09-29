# Shipped student: int8 export and temperature

Steps 2 and 3 of "Test protocol for the shipped student (written
2026-09-29 ...)" in `docs/species-classifier-plan.md`: export seed 17 of
KDw-100c+T (`dinov2-s-kdw-e100-t-c576`, last epoch) to ONNX, quantise it to
int8 as the bake-off tiebreaker defines, apply the 0.01 rule, and fit the
shipped variant's temperature. Validation only; the test split is not read.

Reproduce with `sbatch scripts/hellbender_species_export.slurm`, which runs
`python scripts/export_species_onnx.py --checkpoint models/species-110/finetune/dinov2-s-kdw-e100-t-c576/full/seed-17/classifier-last.pt --output models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17`.

The run:

- **Job**: 18080146, `general` partition, 8 CPUs, node c040 (AMD EPYC 7713),
  16 min, 2026-09-29.
- **Export**: opset 17, TorchScript exporter, input `pixels` 1x3x224x224,
  output `logits` 1x110. No op failed. ONNX Runtime fp32 matches PyTorch fp32
  to 8.6e-6 (max logit difference, 8 images), 3.2e-5 over all of validation.
- **Quantisation**: `quantize_dynamic`, QInt8 weights, per tensor, no
  calibration, no pre-processing pass. MatMul/Gemm weights are int8,
  activations are quantised per inference; the rest stays fp32.
- **Reproduction check** (hard): PyTorch fp32 on CPU against the checkpoint's
  own `predictions-last.npz` (bf16 autocast, GPU): same 3,959 photos and
  labels in order, top-1 agreement 0.997, balanced accuracy 0.7631 against
  0.7634. Passed (limits 0.98 and 0.005).

## Answer

Seed 17, last epoch, full validation (3,959 images), 224 px:

| Variant | Balanced accuracy | Plain accuracy | Top-5 | Section balanced |
| --- | ---: | ---: | ---: | ---: |
| recorded at training (PyTorch, bf16, GPU) | 0.7634 | 0.8012 | 0.9591 | 0.9107 |
| PyTorch fp32, CPU | 0.7631 | 0.8010 | 0.9591 | 0.9103 |
| ONNX fp32 | 0.7631 | 0.8010 | 0.9591 | 0.9103 |
| **ONNX int8** (ships) | **0.7635** | 0.8002 | 0.9566 | 0.9097 |
| int8 - fp32 | **+0.0003** | -0.0008 | -0.0025 | -0.0006 |

int8 and fp32 agree on top-1 for 0.955 of images; the largest logit change is 1.7.

| Shipped variant | Temperature | ECE before | ECE after | NLL before | NLL after | Mean confidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| int8 | **0.74** | 0.111 | **0.025** | 0.832 | 0.743 | 0.69 -> 0.82 |

Probabilities are `softmax(logits / 0.74)`. The fit is the one of
`diagnose_species_distill.py` (grid 0.30-3.00, step 0.01, validation NLL;
15-bin ECE), on the int8 logits.

| Latency, 1 thread, batch 1, CPUExecutionProvider | Median | p10-p90 | File |
| --- | ---: | --- | ---: |
| fp32 | 178 ms | 166-190 ms | 86.8 MB |
| int8 | **118 ms** | 110-127 ms | **22.4 MB** |

Median of 200 runs after 20 warm-ups, one AMD EPYC 7713 core. A diagnostic,
not a gate: int8 is 34% faster and 3.9x smaller.

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| int8 costs more than 0.01 validation balanced accuracy against its own fp32: fp32 ships | **no**: +0.0003 (int8 is not worse) |
| int8 ships otherwise | **yes**: `model-int8.onnx` |
| The model does not export or does not run in ONNX Runtime | no: exports and runs, fp32 matches PyTorch to 1e-5 |
| Temperature: fitted on validation for the shipped variant | **0.74**, not at a grid edge |

What remains before release is step 4: the single scoring of the test split,
with `model-int8.onnx` at temperature 0.74 as the shipped row.

## Release files

`models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/` (gitignored):
`model-int8.onnx`, `model-fp32.onnx`, `labels.json`, `val-logits.npz` (all
three variants' validation logits, for the test step), and `release.json`
(checkpoint and file sha256s, input spec, temperature, every metric above,
latencies, versions: torch 2.5.1, timm 1.0.29, onnx 1.23.1, onnxruntime
1.30.0). Checkpoint sha256 `1fb9ff7f...e81d2`; `model-int8.onnx` sha256
`d7ee150e...8af16`.

**Input contract** for a client: float32 NCHW RGB in [0, 1] (pixel / 255),
1x3x224x224, after resizing the short side to 255 px (bicubic) and centre
cropping 224. The ImageNet mean/std normalisation is **inside the graph**.

## Caveats

- **The temperature is fitted and scored on the same images**, so ECE 0.025
  is optimistic; the test split checks it. It is close to the seed-mean fit
  on PyTorch logits in `species-distill.md` (0.76), as expected.
- **int8 moves 4.5% of top-1 decisions** while leaving balanced accuracy
  unchanged: gains and losses cancel (plain accuracy -0.0008). Top-5 drops by 0.0025, to
  0.957, still above the 0.95 floor (the floor is a seed mean on PyTorch
  logits, so this is a reading, not a gate).
- **Preprocessing in a browser will not be pixel-identical.** Training and
  validation read a cache pre-shrunk to a 576 px short side (LANCZOS) and
  then resized to 255 bicubic, via PIL; a client resizing the original once
  in canvas uses a different filter. Training also ignored EXIF orientation,
  which browsers apply. Neither is measured here.
- **Latency is one server core**, a proxy for single-thread WASM as the
  tiebreaker defines it; a browser will be slower. PyTorch fp32 and ONNX
  fp32 on CPU differ from the recorded bf16 GPU run by 0.0003, which is why
  the reproduction check has a tolerance.
