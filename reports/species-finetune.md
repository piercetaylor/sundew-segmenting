# Fine-tuned teachers: DINOv2-L leads, and the crop stops helping

Stage 2 of `docs/species-classifier-plan.md`: the two leaders of the frozen
screen, `bioclip-2` and `dinov2-l-reg`, fine-tuned end to end at 224 px on
110 species, three arms (`full`, `crop`, `full-square`), five paired seeds.
Training split `split-110-test` (11,118 images; the held-out test split is not
read), validation unchanged (3,959 images, 308 observers). The full generated
table is `species-finetune/results.md`; per-species errors and calibration
are in `species-finetune/teacher-analysis.md`.

Reproduce with
`sbatch --array=0-14 --export=ALL,MODEL=bioclip-2 scripts/hellbender_species_finetune.slurm`,
the same with `MODEL=dinov2-l-reg`, then
`python scripts/summarize_species_finetune.py --out-md reports/species-finetune/results.md --out-json reports/species-finetune/summary.json`
and
`python scripts/analyze_species_teacher.py --out-md reports/species-finetune/teacher-analysis.md --out-json reports/species-finetune/teacher-analysis.json`.
Jobs 17943517 (`bioclip-2`) and 17943518 (`dinov2-l-reg`), 30 A100 tasks of
27-53 min, 2026-09-25 02:29-05:51. Recipe: AdamW lr 5e-5, 2 warmup epochs,
25 epochs, patience 8, layer decay 0.85, drop-path 0.2 (DINOv2-L only;
open_clip ignores it), head lr x10, weight decay 0.05, label smoothing 0.05,
bf16. All 30 tasks completed; one (`dinov2-l-reg` crop, seed 17) stopped
early at epoch 23.

## Answer

Validation balanced accuracy, mean of five seeds at the best epoch. The anchor
is the ResNet-18 rerun on the same split: **full 0.499, crop 0.516**.

Intervals on differences are observer-grouped bootstraps of the seed mean:
each resample of validation observers rescores all five seeds. They cover both
seed noise and which photos happen to be in validation. The seed-paired t on
4 df, used in earlier reports, holds validation fixed and is narrower; it is
kept in `results.md` as a secondary check. (Changed 2026-09-26, after the
audit below; the first version of this report used the t interval.)

| Model | `full` | `crop` | `full-square` | Seed SD | 5-seed ensemble, `full` [95% CI] | Top-5 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `dinov2-l-reg` | **0.824** | 0.820 | 0.811 | 0.002 | **0.835** [0.814, 0.857] | 0.975 |
| `bioclip-2` | 0.802 | 0.799 | 0.795 | 0.006-0.008 | 0.816 [0.798, 0.836] | 0.967 |

- **Fine-tuning moves the best model from 0.516 to 0.824**, +0.31 over the
  ResNet-18 anchor on the same data, with a large backbone and nothing else
  changed.
- **DINOv2-L is the teacher.** It beats BioCLIP-2 on every arm in 5 of 5
  paired seeds, by +0.015 to +0.023, and every interval excludes zero (`full`
  +0.022 [+0.013, +0.030]; ensemble +0.019 [+0.008, +0.030]). Frozen, the
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

| Seed-mean difference (best epoch), bootstrap 95% CI, seeds won | DINOv2-L | BioCLIP-2 |
| --- | --- | --- |
| `crop` - `full` | -0.004 [-0.012, +0.006], 1/5 | -0.002 [-0.011, +0.006], 3/5 |
| `crop` - `full-square` | +0.010 [-0.004, +0.022], 5/5 | +0.004 [-0.011, +0.017], 3/5 |
| `full` - `full-square` | +0.013 [+0.002, +0.022], 5/5 | +0.006 [-0.004, +0.016], 4/5 |

**Once fine-tuned, there is no evidence the crop helps a strong backbone.**
For both models `crop` - `full` sits on zero, and the upper end of the DINOv2-L
interval bounds any benefit at about +0.006. (The seed-paired t gave
-0.004 [-0.007, -0.000] for DINOv2-L, which read as "the crop is slightly
worse"; with validation sampling included, that is not supported.) The frozen
screen predicted this. For ResNet-18 the crop was worth +0.017 on this split;
the benefit belonged to the weak backbone.

`full-square` is the lowest arm for both models, but read that with care:
its training transform was handicapped. It squashed each photo to 224 px
*before* the random crop, so it trained on upsampled sub-crops of a 224-px
image while the other arms cropped from the original. The ranking of
`full-square` is therefore partly an artefact; no decision rests on it.

## Teacher errors and calibration

From `species-finetune/teacher-analysis.md`: DINOv2-L `full`, 5-seed
last-epoch ensemble, validation.

- **The ensemble is under-confident.** Mean confidence 0.74 against plain
  accuracy 0.857; ECE 0.118. One temperature, T = 0.72, brings ECE to 0.019
  and NLL from 0.639 to 0.539. T is fitted and scored on validation, so the
  test split has to confirm it. Class weighting, label smoothing and seed
  averaging all push confidence down, so this is expected.
- **Errors are mostly taxonomic near-misses.** 68% of 566 errors stay within
  the true section. The top pairs are known look-alike groups: *macrantha* /
  *planchonii* (16), *aberrans* / *whittakeri* (11), *dielsiana* /
  *natalensis* (11), *gunniana* / *hookeri* (10), *hirsuta* -> *macrantha*
  (10), and the *peltata* / *auriculata* / *lunata* complex, where some
  iNaturalist labels are likely wrong themselves.
- **Thin classes are weaker, but the link is loose.** Species with under 40
  training images average 0.76; those with 130 or more, 0.88 (Spearman 0.22
  across species). The worst is *D. dielsiana* at 0.15 (32 training images,
  13 validation), mostly called *natalensis*.
- **Weakest sections**: *Brasiliae* 0.71, *Luniferae* 0.76, *Macrantha* 0.77.
- **Balanced accuracy assumes every species is equally common.** Training
  weights classes toward a uniform prior; an app's users mostly photograph
  common species, so field accuracy will differ. Plain accuracy (0.857) is
  reported beside balanced accuracy from here on.

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
| Crop - full gap gone for strong backbones (frozen-screen reading) | held: on zero for both models |
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
  logits, so exact ties are common, and the training-time metric took top-1
  from an unstable `argsort` while the summary uses `argmax`. The metric now
  uses a stable sort, which matches `argmax`. No comparison here is that close.
- **The two recipes are not quite symmetric.** Layer decay matches (the last
  block at 0.85, the head at 1.0 in both), but BioCLIP-2 gets no drop-path
  (open_clip has none) against 0.2 for DINOv2-L.
- **The ResNet-18 anchor kept hue jitter 0.05**; the fine-tunes turned it off.
- **224 px only.** Resolution is the next lever in the plan.
- **One recipe, set before any result and not tuned.** Neither model was given
  a learning-rate or layer-decay search, so the gap between them is for this
  recipe.

## Audit, 2026-09-26

An independent review (a second model, read-only) checked the split and the
code. Verified: no observer, observation, identical-file (sha256) or
near-duplicate (dHash distance <= 6) overlap between any two of train,
validation and test; one photo per observation; the test split unread. Fixed
after it, in `scripts/finetune_species_backbone.py`:

- `img_size` was passed to any timm name containing `vit_`, which crashes on
  TinyViT and FastViT (a student candidate). Now plain ViTs only, as the
  screen already did.
- `full-square` now squashes to 268 px before the random crop, so crops are
  not upsampled. Runs already done keep the old transform.
- A task requeued after it finished no longer retrains from scratch.

## Next

In order of value per GPU-hour; each is pre-registered in
`docs/species-classifier-plan.md` before it runs.

1. **Student bake-off** (~10 A100-h): `dinov2-s` against
   `tinyvit-21m-in22k`, `full` arm, 224 px, 5 seeds, the same recipe.
2. **DINOv2-L at 384 px** (~25 A100-h), `full` arm, 5 seeds.
3. **Distil** the final teacher ensemble into the chosen student (~15 A100-h),
   paired against the bake-off's student trained without a teacher.
4. **Open-set set**: look-alike genera (*Pinguicula*, *Byblis*,
   *Drosophyllum*, *Utricularia*) and Drosera outside the 110, through the
   same licence pipeline, to set an abstain threshold for the app.
5. **Score the test split once**, under the protocol in the plan, then
   freeze. Geo prior after that (records carry no coordinates; needs a
   refetch).
