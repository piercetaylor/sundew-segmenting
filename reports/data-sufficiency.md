# Do we need more images? And is segmentation ready for species prediction?

Two questions, three experiments, and they do not have the same answer.

## 1. More *training* images: no measurable return

SegFormer-B0 trained on 25/50/75/100% of the 144-image train split, three seeds
each, scored on the same fixed 19 validation images
(`scripts/hellbender_data_curve.slurm`).

| Train images | Fraction | Mean val IoU | SD | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.25 | 0.5428 | 0.0045 | |
| 72 | 0.50 | 0.6191 | 0.0119 | +0.0763 |
| 108 | 0.75 | 0.6482 | 0.0116 | +0.0291 |
| 144 | 1.00 | 0.6408 | 0.0223 | **-0.0074** |

The curve saturates. Going 36 -> 144 images (4x) bought +0.098 IoU, but the
last 36 images bought nothing distinguishable from zero. Annotating another
hundred images from the same distribution should not be expected to move IoU.

This is a statement about *this recipe*, not about the task: 768 px, flip-only
augmentation, 30 epochs, fixed hyperparameters. A plateau can mean the recipe
has run out of headroom rather than the data has. The practical reading is that
recipe work is now a better investment than annotation work.

## 2. More *evaluation* images: yes, this is the real gap

The 19-image validation split is too small to measure what the project needs to
decide.

Per-image validation IoU has SD 0.1025, so at n=19 the standard error is 0.0235
and the 95% interval on the headline number is about +/-0.046.

| To reliably detect a gap of | You need about |
| ---: | ---: |
| 0.05 IoU | 65 validation images |
| 0.03 IoU | 180 validation images |
| 0.02 IoU | 404 validation images |

The measured SegFormer-vs-U-Net gap was 0.0215. A single 19-image split cannot
see an effect that size; only pairing across seeds made it visible, and that
trick does not extend to comparing against a future model trained differently.

The 28-image test split has the same problem and has never been spent.

**More annotation effort should go to validation and test, not train.**

## 3. Calibration is free accuracy

The project's research question is projected plant area and percent cover, so
area error matters more than IoU. On validation, SegFormer-B0 at the default
0.5 threshold gives:

- mean absolute relative area error **21.7%**, median 17.9%
- **systematic +14.0% over-prediction**
- R^2 between true and predicted area **0.84**

The bias is not mysterious. The training loss is BCE + Tversky weighted 0.3 on
false positives and 0.7 on false negatives, which deliberately buys recall at
the cost of over-segmentation. Sweeping the decision threshold:

| Threshold | IoU | Mean abs area err | Signed area err |
| ---: | ---: | ---: | ---: |
| 0.40 | **0.6453** | 26.2% | +21.5% |
| 0.50 (default) | 0.6442 | 21.7% | +14.0% |
| 0.60 | 0.6376 | 18.9% | +6.9% |
| 0.70 | 0.6225 | **17.3%** | **-0.7%** |
| 0.80 | 0.5912 | 18.1% | -9.8% |

Threshold 0.70 removes the area bias almost exactly and cuts mean area error by
a fifth, costing 0.022 IoU. IoU-optimal and area-optimal thresholds are not the
same point, and the deliverable is area.

Caveat: this threshold is fitted on the same 19 images it is scored on, so the
gain is optimistic. It needs a larger held-out set to trust — see section 2.

## 4. Species prediction is blocked by labels, not by masks

The dataset carries `taxon_name`, so the species ceiling can be measured
directly. 191 images span **51 species**, median **1 image per species**.

| | |
| --- | ---: |
| Distinct species | 51 |
| Species with exactly 1 image | 26 |
| Species with < 5 images | 38 |
| Species with >= 5 *train* images | 9 (90 of 144 train images) |
| Species with >= 10 *train* images | 4 |
| Test species never seen in train | 6 |
| Validation species never seen in train | 4 |

Splits are grouped by observer (139 observers, zero cross-split leakage) — good
design for segmentation, but it means species fall where they fall, and ten
species exist only in a split where they cannot be learned.

A 51-way classifier is not trainable here. A coarse classifier over the ~9
best-represented species is arguably reachable, but that covers 90 training
images and discards a third of the data.

This is a labelling-budget problem, not a segmentation-quality problem. Better
masks will not create species coverage. Note also that the project plan lists
species identification as explicitly out of scope for the first release.
