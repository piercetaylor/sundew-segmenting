# Fine-tuned teachers: DINOv2-L leads, and the crop stops helping

Stage 2 of `docs/species-classifier-plan.md`: the two leaders of the frozen
screen, `bioclip-2` and `dinov2-l-reg`, fine-tuned end to end at 224 px on
110 species, three arms (`full`, `crop`, `full-square`), five paired seeds.
Training split `split-110-test` (11,118 images; the held-out test split is not
read), validation unchanged (3,959 images, 308 observers). The full generated
table is `species-finetune/results.md`.

Reproduce with
`sbatch --array=0-14 --export=ALL,MODEL=bioclip-2 scripts/hellbender_species_finetune.slurm`,
the same with `MODEL=dinov2-l-reg`, then
`python scripts/summarize_species_finetune.py --out-md reports/species-finetune/results.md --out-json reports/species-finetune/summary.json`.
Jobs 17943517 (`bioclip-2`) and 17943518 (`dinov2-l-reg`), 30 A100 tasks of
27-53 min, 2026-09-25 02:29-05:51. Recipe: AdamW lr 5e-5, 2 warmup epochs,
25 epochs, patience 8, layer decay 0.85, drop-path 0.2 (DINOv2-L only;
open_clip ignores it), head lr x10, weight decay 0.05, label smoothing 0.05,
bf16. All 30 tasks completed; one (`dinov2-l-reg` crop, seed 17) stopped
early at epoch 23.

## Answer

Validation balanced accuracy, mean of five seeds at the best epoch. The anchor
is the ResNet-18 rerun on the same split: **full 0.499, crop 0.516**.

| Model | `full` | `crop` | `full-square` | Seed SD | 5-seed ensemble, `full` [95% CI] | Top-5 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `dinov2-l-reg` | **0.824** | 0.820 | 0.811 | 0.002 | **0.835** [0.814, 0.857] | 0.975 |
| `bioclip-2` | 0.802 | 0.799 | 0.795 | 0.006-0.008 | 0.816 [0.798, 0.836] | 0.967 |

- **Fine-tuning moves the best model from 0.516 to 0.824**, +0.31 over the
  ResNet-18 anchor on the same data, with a large backbone and nothing else
  changed.
- **DINOv2-L is the teacher.** It beats BioCLIP-2 on every arm in 5 of 5
  paired seeds, by +0.015 to +0.023 (all t intervals exclude zero; the
  ensemble difference on `full` is +0.019 [+0.008, +0.030]). Frozen, the
  order was the other way round (0.750 vs 0.729). DINOv2-L is also the model
  without the contamination caveat, so the lead does not rest on BioCLIP
  having seen these photos.
- **DINOv2-L is also the more stable**: seed SD 0.002 against 0.006-0.008.
- **Fine-tuning gained more for DINOv2-L**: crop arm 0.729 frozen -> 0.820
  fine-tuned (+0.091), against 0.750 -> 0.799 (+0.049) for BioCLIP-2, and on
  19% less training data than the frozen probes had.
- **Ensembling helps a little, mixing models does not.** Five DINOv2-L seeds
  add about 0.011 over the seed mean; averaging the DINOv2-L and BioCLIP-2
  ensembles on `full` gives 0.836, no better than DINOv2-L alone.
- **Misses are mostly near-misses.** The DINOv2-L ensemble has the right
  species in its top five 97.5% of the time.

## The crop question: answered, no

| Comparison, paired over seeds (best epoch) | DINOv2-L | BioCLIP-2 |
| --- | --- | --- |
| `crop` - `full` | -0.004 [-0.007, -0.000], 1/5 | -0.002 [-0.009, +0.005], 3/5 |
| `crop` - `full-square` | +0.010 [+0.005, +0.014], 5/5 | +0.004 [-0.005, +0.013], 3/5 |
| `full` - `full-square` | +0.013 [+0.010, +0.016], 5/5 | +0.006 [-0.001, +0.014], 4/5 |

**Once fine-tuned, a strong backbone does at least as well on the whole frame
as on the segmenter's crop.** For DINOv2-L the crop is slightly worse than
`full` (1 of 5 seeds; the last-epoch interval, -0.004 [-0.008, +0.001], just
reaches zero), and for BioCLIP-2 the two are indistinguishable. The frozen
screen predicted this; the fine-tune confirms it. For ResNet-18 the crop was
worth +0.017 on this split; the benefit belonged to the weak backbone.

`full-square` is the worst arm for both models, as it was frozen. The crop
does beat it for DINOv2-L, but `full` beats it by more, so the gap is about
squashing the aspect ratio, not about the crop.

**Segmentation comes off the species critical path.** The species classifier
takes the whole photo, resized and centre-cropped. The segmenter remains a
result in its own right, and a tool for later work on area or growth form.

## Against expectations

No prior for this stage was written into the plan beyond the choice of
models and arms. The one stated expectation was the second model's estimate
(`docs/species-classifier-plan.md`, "Small models for an on-device
student"):

| Expectation | Outcome |
| --- | --- |
| Teacher at 0.80-0.85 top-1 balanced accuracy | held: 0.824 seed mean, 0.835 ensemble |
| Crop - full gap gone for strong backbones (frozen-screen reading) | held, slightly negative for DINOv2-L |
| BioCLIP-2 and DINOv2-L close | held, but DINOv2-L now clearly ahead |

## What it means for the on-device student

The small-backbone rule (`species-backbone-screen.md`) picked `dinov2-s`
from the crop arm, where it leads TinyViT-21M by 0.017. If the teacher runs on
full frames, the app has no reason to segment first. On full frames the two
students tie (frozen 0.616 vs 0.614), and on `full-square` TinyViT leads.
The pre-registered rule still names DINOv2-S; whether to distil both from the
DINOv2-L `full` teacher before choosing is a decision to make, and to write
into the plan, before any distillation run.

## Caveats

- **Validation picked these models.** It chose the backbones in the screen and
  the best epoch here. Last-epoch seed means are within 0.004 of best-epoch
  ones, so epoch selection adds little, but the unbiased number is the
  held-out test split, scored once after the configuration is fixed.
- **Recomputed balanced accuracy differs from the training log by up to
  0.0008** per run (for example 0.8230 against 0.8233). Scores are bf16
  logits saved as float32, so exact ties are common; tie-breaking in argmax is the
  likely cause. No comparison here is that close.
- **224 px only.** Resolution is the next lever in the plan.
- **One recipe, set before any result and not tuned.** Neither model was given
  a learning-rate or layer-decay search, so the gap between them is for this
  recipe.

## Next

1. **Decide the student question above** and write it into the plan.
2. **Resolution 224 -> 384** on `dinov2-l-reg`, `full` arm (the 768 px crops
   no longer matter if the crop is dropped).
3. **Geo prior**, in parallel.
4. **Reporting, once**: score the held-out test split with the fixed
   configuration, alongside the ResNet-18 anchor; per-class accuracy against
   training count; most-confused species pairs.
5. **Distil** DINOv2-L into the chosen small student.
