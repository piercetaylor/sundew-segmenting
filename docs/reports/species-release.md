# Shipped student: int8 export and temperature

**In short.** Seed 17 of KDw-100c+T, exported to ONNX and quantised to int8,
loses nothing on validation (balanced accuracy 0.7635 against 0.7631 for
fp32), so int8 ships: 22.4 MB, 118 ms per photo on one CPU thread.
Probabilities are `softmax(logits / 0.74)`, a temperature fitted on
validation. The test split was not read here.

Steps 2 and 3 of "Test protocol for the shipped student (written 2026-09-29
...)" in the [plan](../species-classifier-plan.md): export seed 17 of
KDw-100c+T (`dinov2-s-kdw-e100-t-c576`, last epoch; see
[species-distill.md](species-distill.md)), quantise it to int8, apply the
0.01 rule, and fit the temperature of the variant that ships.

Reproduce with `sbatch scripts/hellbender_species_export.slurm`, which runs
`python scripts/export_species_onnx.py --checkpoint models/species-110/finetune/dinov2-s-kdw-e100-t-c576/full/seed-17/classifier-last.pt --output models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17`.
Job 18080146, `general` partition, 8 CPUs, node c040 (AMD EPYC 7713), 16 min,
2026-09-29.

- **Export**: opset 17, TorchScript exporter, input `pixels` 1x3x224x224,
  output `logits` 1x110. ONNX Runtime fp32 matches PyTorch fp32 to 8.6e-6
  (max logit difference, 8 images), 3.2e-5 over all of validation.
- **Quantisation**: `quantize_dynamic`, int8 weights for MatMul/Gemm, per
  tensor, no calibration data; activations quantised per inference; the rest
  stays fp32.
- **Reproduction check** (hard gate): PyTorch fp32 on CPU against the
  checkpoint's own recorded predictions (bf16, GPU): top-1 agreement 0.997,
  balanced accuracy 0.7631 against 0.7634. Passed (limits 0.98 and 0.005).

## Result

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

The temperature minimises validation NLL on the int8 logits over a grid
0.30-3.00, step 0.01 (as in `scripts/diagnose_species_distill.py`); ECE uses
15 bins. A temperature below 1 sharpens the probabilities: the raw model is
under-confident.

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

The last step, one scoring of the test split, is in [species-test.md](species-test.md).

## Release files

`models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/` (not in Git):
`model-int8.onnx`, `model-fp32.onnx`, `labels.json`, `val-logits.npz`
(validation logits of all three variants), and `release.json` (checksums,
input spec, temperature, every metric above, latencies, library versions:
torch 2.5.1, timm 1.0.29, onnx 1.23.1, onnxruntime 1.30.0). Checkpoint sha256
`1fb9ff7f...e81d2`; `model-int8.onnx` sha256 `d7ee150e...8af16`.

**Input contract** for a client: float32 NCHW RGB in [0, 1] (pixel / 255),
1x3x224x224, after resizing the short side to 255 px (bicubic) and centre
cropping 224. The ImageNet mean/std normalisation is **inside the graph**.
Output: 110 logits in `labels.json` order.

## Caveats

- **The temperature is fitted and scored on the same images**, so ECE 0.025
  is optimistic (on test it is 0.031). It is close to the seed-mean fit on
  PyTorch logits (0.76).
- **int8 changes 4.5% of top-1 decisions** while leaving balanced accuracy
  unchanged: gains and losses cancel. Top-5 drops by 0.0025, to 0.957, still
  above the 0.95 floor (the floor is a seed mean on PyTorch logits, so this is
  a reading, not a gate).
- **Browser preprocessing will not be pixel-identical.** Training read frames
  pre-shrunk to a 576 px short side (LANCZOS), then resized to 255 bicubic, in
  PIL; a browser resizing the original once uses a different filter. Training
  also ignored EXIF orientation, which browsers apply. Neither is measured.
- **Latency is one server core**, a stand-in for single-thread WebAssembly; a
  browser will be slower. PyTorch and ONNX fp32 on CPU differ from the
  recorded bf16 GPU run by 0.0003, which is why the reproduction check has a
  tolerance.
