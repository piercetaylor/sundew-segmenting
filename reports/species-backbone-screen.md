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
- **BioCLIP numbers are optimistic by an unmeasurable amount** (possible
  training-set overlap with iNaturalist validation photos). DINOv2-L is the
  uncontaminated comparison.
- **The linear probe's L2 strength never landed on the grid edge**
  (chosen 1e-3 or 1e-2 everywhere), so the grid did not constrain any result.
- **Zero-shot BioCLIP-2 reaches 0.519** with no training at all, just
  "a photo of Drosera <species>". Diagnostic only.

## Next

In order, per `docs/species-classifier-plan.md`:

1. Carve out an observer-grouped **held-out test split** before fine-tuning.
2. **Fine-tune `bioclip-2` and `dinov2-l-reg`**, 5 seeds, lr 2e-5 to 5e-5 with
   warmup, layer decay ~0.75, drop-path, bf16, hue jitter off; arms `crop`,
   `full` and `full-square`.
3. Resolution 224 -> 384; geo prior in parallel.
