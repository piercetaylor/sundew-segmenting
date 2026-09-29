# Frozen-backbone screen: the backbone was the ceiling

**In short.** With its weights frozen and only a linear layer trained on top,
BioCLIP-2 (0.750) and DINOv2-L (0.729) beat the fully fine-tuned ResNet-18
(0.546) by about 0.2. Both go on to fine-tuning. For these strong backbones,
cropping to the plant no longer helps. Among small models, DINOv2-S (0.637)
clears the 0.58 bar and becomes the on-device student candidate.

A frozen screen: each pretrained backbone turns every photo into one
embedding, and a linear probe (a single linear layer) is fitted on top. It is
cheap and ranks backbones without fine-tuning them. Design, prior and decision
rule were fixed beforehand in the [plan](../species-classifier-plan.md); the
full generated table, with k-NN, zero-shot and every interval, is
[species-backbone-screen/results.md](species-backbone-screen/results.md).

Reproduce with
`jid=$(sbatch --parsable scripts/hellbender_backbone_screen.slurm)` then
`sbatch --dependency=afterok:$jid scripts/hellbender_backbone_screen_summary.slurm`.
Nine A100 tasks of about six minutes each (job 17927437, 2026-09-23).

## Result: nine large backbones

Linear-probe validation balanced accuracy on the crop arm, 95% bootstrap
interval grouped by observer. Reference: fine-tuned ResNet-18, **0.546**.
"Section" is accuracy at the level of the taxonomic section, a coarser group
of related species.

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

- **Seven of nine frozen backbones beat the fine-tuned ResNet-18.** Every
  backbone beats the frozen anchor with P(delta <= 0) = 0.000.
- **The top two are close.** Their intervals overlap. DINOv2-L carries no risk
  of having seen these photos in pretraining (see caveats) and is within 0.021
  of BioCLIP-2, so the jump does not rest on BioCLIP.
- **Most misses are near-misses.** For BioCLIP-2 the right species is in the
  top five 96% of the time, and the right section 91%.
- **Size matters within a family**: DINOv2-L beats DINOv2-B by 0.054, BioCLIP-2
  beats BioCLIP by 0.155.

## The crop stops helping strong backbones

`full` is the photo centre-cropped to a square; `full-square` is the whole
photo squashed to a square, nothing cut away.

| Backbone | Crop - full [95% CI] | Crop - full-square [95% CI] |
| --- | --- | --- |
| fine-tuned ResNet-18 (for reference) | +0.021 (5 seeds) | not run |
| `bioclip-2` | -0.007 [-0.023, +0.009] | +0.006 [-0.009, +0.024] |
| `dinov2-l-reg` | -0.001 [-0.018, +0.011] | +0.028 [+0.004, +0.043] |
| `dinov2-b` | +0.011 [-0.005, +0.030] | +0.053 [+0.032, +0.073] |
| `resnet18-in1k` frozen | +0.011 [-0.006, +0.025] | +0.033 [+0.009, +0.050] |

For the two leaders, crop - full sits on zero: a strong backbone finds the
plant without help. This is a frozen result; the fine-tune had to confirm it.

The prior expected `full-square` to beat `full`. It did the opposite for eight
of nine backbones. Squashing distorts the aspect ratio, so this control could
not test whether centring matters.

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

Given the crop - full result, the fine-tune also carried the plain `full` arm,
which answers directly whether segmentation is still needed.

## Caveats

- **SigLIP is probably under-measured.** It was pretrained at 378 px and read
  at 224 with resampled position embeddings. It is not a verdict on SigLIP,
  and does not change the decision.
- **BioCLIP numbers may be optimistic**: its training data may include some of
  these iNaturalist validation photos. Checked 2026-09-26 by comparing photos
  observed before and after BioCLIP-2's data was collected
  ([species-finetune.md](species-finetune.md#bioclip-2-contamination-bounded-not-detected)):
  no inflation detected, but for the frozen `crop` arm the bound is loose,
  +0.017 [-0.030, +0.061]. Inflation up to about 0.06 is not excluded, so
  BioCLIP-2's 0.021 frozen lead does not prove it is the better backbone.
  BioCLIP (v1) was not checked. DINOv2-L is the clean comparison.
- **The probe's L2 strength never hit the grid edge** (1e-3 or 1e-2 everywhere).
- **Zero-shot BioCLIP-2 reaches 0.519** from the text prompt "a photo of
  Drosera <species>" alone. Diagnostic only.

## Small backbones for an on-device student

A second pass screened 11 backbones of 8-35M parameters the same way, to pick
a student small enough for a browser or phone. Design, prior and rule: the
plan's "Small models for an on-device student". Generated table (all 20
models): [species-backbone-screen-small/results.md](species-backbone-screen-small/results.md).

Reproduce with
`jid=$(sbatch --parsable --array=0-10 --export=ALL,MODEL_SET=small scripts/hellbender_backbone_screen.slurm)`
then
`sbatch --dependency=afterok:$jid --export=ALL,OUT=docs/reports/species-backbone-screen-small scripts/hellbender_backbone_screen_summary.slurm`.
Eleven A100 tasks of about six minutes each (job 17943537, summary 17943538,
2026-09-25).

### Result: DINOv2-S clears the bar

Linear probe, crop arm, 95% observer-grouped interval. The bar set beforehand
is 0.58.

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

For scale: fine-tuned ResNet-18 0.546, frozen DINOv2-B 0.675, frozen DINOv2-L 0.729.

- **Three small models clear 0.58**; for `dinov2-s` the whole interval does.
- **The lead is not separated.** `dinov2-s`, `dinov2-s-reg` and `tinyvit-21m`
  are within 0.017 of each other with overlapping intervals.
- **k-NN gives the same order**, so the ranking is not an artefact of the probe.
- L2 strength never hit the grid edge.

**Without the crop the choice can flip.** DINOv2-S gains more from the crop
than TinyViT (crop - full-square +0.062 [+0.037, +0.083] against +0.028
[+0.009, +0.045]). On full frames they tie (0.616 vs 0.614); on full-square
TinyViT leads (0.592 vs 0.574). Every small backbone gains from the crop over
full-square (all 11 intervals exclude zero); over `full`, only five of 11 do.
This tie on full frames is why the [student bake-off](species-student-bakeoff.md)
was run.

### Against the prior

| Prior | Outcome |
| --- | --- |
| DINOv2-S or DINOv3-S leads the small models | **half**: DINOv2-S leads; DINOv3-S is ninth of 11, below the fine-tuned ResNet-18 |
| MobileNetV4 and EfficientNetV2 trail | **mostly**: MobileNetV4-conv is last, but MobileCLIP2-S2 and DINOv3-S land below EfficientNetV2 and MobileNetV4-hybrid |
| Every small model below DINOv2-B (0.675) | held |

The DINOv3 small models underperform (0.563, 0.543), so the unresolved check
of Meta's DINOv3 licence no longer matters for the student.

### Decision, applied as written

| Best small backbone, frozen | Fires? |
| --- | --- |
| >= 0.58: on-device student is viable; that backbone is the student | **yes**: `dinov2-s`, 0.637 |
| 0.50-0.58: viable only with distillation | no |
| < 0.50: serve the teacher from a server | no |

`dinov2-s` (Apache-2.0 weights, same family as the DINOv2-L teacher) is the
student for a crop-first app, with `tinyvit-21m-in22k` (MIT) as the
alternative if the app skips segmentation. Next came the held-out test split,
then fine-tuning the two leaders ([species-finetune.md](species-finetune.md)).
