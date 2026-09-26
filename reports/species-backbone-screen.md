# Frozen-backbone screen: the backbone was the ceiling

Nine pretrained backbones, weights frozen, one linear probe each, 110 species.
Design, prior and decision rule were fixed beforehand in
`docs/species-classifier-plan.md`; the full generated table, with k-NN,
zero-shot and every interval, is `species-backbone-screen/results.md`.

Reproduce with
`jid=$(sbatch --parsable scripts/hellbender_backbone_screen.slurm)` then
`sbatch --dependency=afterok:$jid scripts/hellbender_backbone_screen_summary.slurm`.
Nine A100 tasks of about six minutes each (job 17927437, 2026-09-23).

## Answer

Linear-probe validation balanced accuracy on the crop arm, 95% observer-grouped
bootstrap interval. The reference is the fine-tuned ResNet-18: **0.546**.

| Backbone | Params | Frozen, crop | vs fine-tuned R18 | Top-5 | Section |
| --- | ---: | --- | ---: | ---: | ---: |
| `bioclip-2` | 304M | **0.750** [0.728, 0.772] | +0.204 | 0.957 | 0.912 |
| `dinov2-l-reg` | 303M | **0.729** [0.706, 0.753] | +0.183 | 0.942 | 0.893 |
| `dinov2-b` | 86M | 0.675 [0.657, 0.699] | +0.129 | 0.915 | 0.854 |
| `dinov3-vit-b` | 86M | 0.649 [0.628, 0.674] | +0.103 | 0.915 | 0.833 |
| `bioclip` | 86M | 0.595 [0.575, 0.621] | +0.049 | 0.884 | 0.803 |
| `dinov3-convnext-b` | 88M | 0.580 [0.554, 0.614] | +0.034 | 0.876 | 0.794 |
| `convnext-b-in22k` | 88M | 0.570 [0.552, 0.596] | +0.024 | 0.874 | 0.779 |
| `siglip2-so400m` | 428M | 0.415 [0.391, 0.439] | -0.131 | 0.744 | 0.644 |
| `resnet18-in1k` (anchor) | 11M | 0.301 [0.282, 0.322] | -0.245 | 0.619 | 0.550 |

- **Seven of nine frozen backbones beat the fine-tuned ResNet-18**, and the
  top two by about 0.2, with no training beyond a linear layer. Every
  backbone beats the frozen anchor with P(delta <= 0) = 0.000.
- **The top two are close and neither is a fluke.** Their intervals overlap.
  DINOv2-L is not exposed to the BioCLIP contamination caveat and lands within
  0.021 of BioCLIP-2, so the jump does not rest on BioCLIP having seen these
  photos.
- **Most misses are now near-misses.** For BioCLIP-2, the correct species is in
  the top five 96% of the time, and the prediction is in the right section
  91% of the time.
- **Size matters within a family.** DINOv2-L beats DINOv2-B by 0.054 and
  BioCLIP-2 beats BioCLIP by 0.155.

## The crop question changes answer with the backbone

| Backbone | Crop - full [95% CI] | Crop - full-square [95% CI] |
| --- | --- | --- |
| fine-tuned ResNet-18 (for reference) | +0.021 (5 seeds) | not run |
| `bioclip-2` | -0.007 [-0.023, +0.009] | +0.006 [-0.009, +0.024] |
| `dinov2-l-reg` | -0.001 [-0.018, +0.011] | +0.028 [+0.004, +0.043] |
| `dinov2-b` | +0.011 [-0.005, +0.030] | +0.053 [+0.032, +0.073] |
| `resnet18-in1k` frozen | +0.011 [-0.006, +0.025] | +0.033 [+0.009, +0.050] |

**For the two leading backbones the crop no longer beats the full frame.** Both
intervals sit on zero. This was the reviewer's guess and part of the prior:
a strong backbone finds the plant without help. It is a frozen-probe result,
though, and the fine-tune has to confirm it before segmentation comes off the
species critical path.

**The centring hypothesis was not supported.** The prior expected
`full-square` (whole frame, nothing cut away) to beat `full` (centre-cropped).
It did the opposite for eight of nine backbones. Squashing distorts aspect
ratio, which is a confound of its own, so this does not rule centring out; it
means this control could not isolate it.

## Against the prior

| Prior | Outcome |
| --- | --- |
| Frozen ResNet-18 well below its fine-tuned 0.546 | held: 0.301 |
| At least one backbone beats the fine-tuned ResNet-18 frozen | held, by a wide margin: seven of nine |
| BioCLIP-2 and DINOv2-L lead | held |
| Crop - full gap shrinks for strong backbones | held: gone for the top two |
| `full-square` beats `full` | **wrong**: worse for eight of nine |

## Decision, applied as written

| Rule | Fires? |
| --- | --- |
| Best backbone >= 0.546 frozen: fine-tune the top two, 5 seeds, ViT recipe | **yes**: `bioclip-2` and `dinov2-l-reg` |
| Top two both BioCLIP: add the best non-BioCLIP | no: DINOv2-L is second |
| On the leader, the CI on crop - full-square includes 0 or is negative: add a `full-square` arm | **yes**: +0.006 [-0.009, +0.024] |

Given the crop - full result above, the fine-tune should carry the plain `full`
arm as well as `full-square`. It costs one more arm and it answers directly
whether segmentation is still needed.

## Caveats

- **SigLIP is probably under-measured.** It was pretrained at 378 px and read
  at 224, with position embeddings resampled to fit; the fixed-resolution rule
  may have hurt it more than the others. It is last among the modern models
  and nowhere near the leaders, so this does not change the decision. It
  should not be read as a verdict on SigLIP.
- **BioCLIP numbers may be optimistic** (possible training-set overlap with
  iNaturalist validation photos). Measured 2026-09-26 by comparing photos
  observed before and after BioCLIP-2's training data was collected
  (`species-finetune.md`, "BioCLIP-2 contamination"): no inflation detected,
  but the frozen `crop` bound is loose, DiD +0.017 [-0.030, +0.061], so
  inflation up to about 0.06 is not excluded and BioCLIP-2's 0.021 frozen
  lead is not proof of a better backbone. BioCLIP (v1) was not checked. DINOv2-L is the
  uncontaminated comparison.
- **The linear probe's L2 strength never landed on the grid edge**
  (chosen 1e-3 or 1e-2 everywhere), so the grid did not constrain any result.
- **Zero-shot BioCLIP-2 reaches 0.519** with no training at all, just
  "a photo of Drosera <species>". Diagnostic only.

## Small backbones for an on-device student

A second pass screened 11 backbones of 8-35M parameters under the same frozen
protocol, to pick a student small enough for a browser or phone. Design, prior
and decision rule are in `docs/species-classifier-plan.md` ("Small models for
an on-device student"); the generated table, which lists all 20 models, is
`species-backbone-screen-small/results.md`.

Reproduce with
`jid=$(sbatch --parsable --array=0-10 --export=ALL,MODEL_SET=small scripts/hellbender_backbone_screen.slurm)`
then
`sbatch --dependency=afterok:$jid --export=ALL,OUT=reports/species-backbone-screen-small scripts/hellbender_backbone_screen_summary.slurm`.
Eleven A100 tasks of about six minutes each (job 17943537, summary 17943538,
2026-09-25).

### Answer: DINOv2-S clears the bar; the student is viable as is

Linear probe, crop arm, 95% observer-grouped bootstrap interval. The bar set
beforehand is 0.58.

| Backbone | Params | Frozen, crop | Full | Full-square | k-NN, crop |
| --- | ---: | --- | ---: | ---: | ---: |
| `dinov2-s` | 22M | **0.637** [0.613, 0.664] | 0.616 | 0.574 | 0.486 |
| `dinov2-s-reg` | 22M | 0.622 [0.597, 0.655] | 0.599 | 0.549 | 0.451 |
| `tinyvit-21m-in22k` | 21M | 0.620 [0.597, 0.648] | 0.614 | 0.592 | 0.445 |
| `convnext-t-in22k` | 28M | 0.570 [0.544, 0.604] | 0.572 | 0.536 | 0.396 |
| `dinov3-vit-s-plus` | 29M | 0.563 [0.535, 0.594] | 0.544 | 0.496 | 0.362 |
| `convnext-nano-in12k` | 15M | 0.557 [0.537, 0.587] | 0.540 | 0.494 | 0.385 |
| `efficientnetv2-s-in21k` | 20M | 0.555 [0.530, 0.584] | 0.555 | 0.506 | 0.429 |
| `mobilenetv4-hybrid-m-in12k` | 10M | 0.551 [0.524, 0.586] | 0.557 | 0.509 | 0.404 |
| `dinov3-vit-s` | 22M | 0.543 [0.516, 0.574] | 0.520 | 0.483 | 0.363 |
| `mobileclip2-s2` | 35M | 0.538 [0.516, 0.569] | 0.523 | 0.487 | 0.344 |
| `mobilenetv4-conv-m-in12k` | 8M | 0.513 [0.488, 0.544] | 0.496 | 0.442 | 0.376 |

For scale: fine-tuned ResNet-18 0.546, frozen DINOv2-B 0.675, frozen
DINOv2-L 0.729.

- **Three small models clear 0.58**, and for `dinov2-s` the whole interval
  does. A 22M-parameter frozen backbone plus a linear layer beats the
  fine-tuned ResNet-18 by 0.091.
- **The lead is not separated.** `dinov2-s`, `dinov2-s-reg` and `tinyvit-21m`
  sit within 0.017 of each other with heavily overlapping intervals. The
  summary carries a paired test only against the frozen ResNet-18 anchor, so
  "top, not separated from TinyViT" is the honest reading.
- **k-NN gives the same order.** `dinov2-s` is again the best small model, so
  the ranking is not an artefact of the probe.
- **L2 strength never hit the grid edge** (1e-3 or 1e-2 for every small model).

### Without the crop, the choice can flip

DINOv2-S gains more from the crop than TinyViT does: crop - full-square is
+0.062 [+0.037, +0.083] against +0.028 [+0.009, +0.045]. On the full-square
arm TinyViT leads (0.592 vs 0.574, and DINOv2-S falls below the bar); on full
frames they tie (0.614 vs 0.616). So the student follows the app's pipeline:
if the phone runs SegFormer-B0 first and classifies the crop, `dinov2-s`; if
it classifies the whole photo, `tinyvit-21m-in22k` is at least as good. Unlike
the large leaders, every small backbone still gains from the crop over
full-square (all 11 intervals exclude zero); over the centre-cropped `full`
arm the gain is clear for only five of 11.

### Against the prior

| Prior | Outcome |
| --- | --- |
| DINOv2-S or DINOv3-S leads the small models | **half**: DINOv2-S leads; DINOv3-S is ninth of 11, below the fine-tuned ResNet-18 |
| MobileNetV4 and EfficientNetV2 trail | **mostly**: MobileNetV4-conv is last, but MobileCLIP2-S2 and DINOv3-S land below EfficientNetV2 and MobileNetV4-hybrid |
| Every small model below DINOv2-B (0.675) | held |

The DINOv3 small models underperform here (DINOv3-S+ 0.563, DINOv3-S 0.543),
so the pending check of Meta's DINOv3 licence no longer blocks the student.

### Decision, applied as written

| Best small backbone, frozen | Fires? |
| --- | --- |
| >= 0.58: on-device student is viable; that backbone is the student | **yes**: `dinov2-s`, 0.637 |
| 0.50-0.58: viable only with distillation | no |
| < 0.50: serve the teacher from a server | no |

`dinov2-s` (Apache-2.0 weights, same lineage as the DINOv2-L teacher) is the
student for a crop-first app, with `tinyvit-21m-in22k` (MIT) as the
alternative if the app skips segmentation. Distillation from the fine-tuned
teacher is still the route to close the gap to it; the rule only says the
student does not depend on it.

## Next

In order, per `docs/species-classifier-plan.md`:

1. Carve out an observer-grouped **held-out test split** before fine-tuning.
2. **Fine-tune `bioclip-2` and `dinov2-l-reg`**, 5 seeds, lr 2e-5 to 5e-5 with
   warmup, layer decay ~0.75, drop-path, bf16, hue jitter off; arms `crop`,
   `full` and `full-square`.
3. Resolution 224 -> 384; geo prior in parallel.
