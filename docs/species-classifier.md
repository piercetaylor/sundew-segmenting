# Species classifier

Given a photo of a sundew, name its species: one of 110 *Drosera* species. The
shipped model is a 22 MB int8 ONNX file that runs in about 120 ms on one CPU
thread and scores **0.775** balanced accuracy on a held-out test set (the
right species in its top five 96% of the time). This page is the summary.
The decisions were [pre-registered](species-classifier-plan.md), and each
step has a report in [reports/](reports/).

## The pipeline

1. **Find the plant.** A SegFormer-B0 segmenter outlines the sundew and a crop
   is taken around it.
2. **Name the species.** A small image classifier (DINOv2-S) scores the 110
   species. It takes the **whole photo**, resized and centre-cropped to 224 px,
   not the crop: once a strong backbone is fine-tuned, the crop no longer helps
   ([species-finetune.md](reports/species-finetune.md)). So segmentation is not
   needed for species ID; it remains useful on its own.
3. **Report probabilities.** `softmax(logits / 0.74)`: a temperature fitted on
   validation, because the raw model is under-confident.

"Balanced accuracy" is the mean of per-species accuracy, so every species
counts equally however many photos it has. Chance is 1/110 = 0.009.

## Why a teacher and a student

Large models are accurate but too big for a phone or browser; small models fit
but are less accurate. So a large **teacher** (DINOv2-L, 303M parameters, an
ensemble of five) is fine-tuned for accuracy, and a small **student** (DINOv2-S,
22M parameters) is trained to copy the teacher's predicted probabilities
(knowledge distillation, KD) as well as the labels. The student recovers part of
the gap and ships alone.

## How the recipe was reached

Each step's rule was written before it ran. Validation balanced accuracy unless
noted; every comparison is five paired seeds.

1. **Baseline.** A fine-tuned ResNet-18 reaches 0.546 on crops, 0.525 on full
   frames; 0.516 / 0.499 on the split with a test set held back
   ([baseline](reports/species-110-baseline.md)).
2. **Backbone screen.** With frozen weights and a linear layer, BioCLIP-2
   (0.750) and DINOv2-L (0.729) beat the fine-tuned ResNet-18: the backbone was
   the ceiling ([screen](reports/species-backbone-screen.md)).
3. **Teacher: DINOv2-L.** Fine-tuned, it reaches 0.824 (5-seed ensemble 0.835)
   and beats BioCLIP-2 in 5 of 5 seeds; the crop stops helping
   ([fine-tune](reports/species-finetune.md)).
4. **Student: DINOv2-S.** Among 11 small backbones, DINOv2-S led the frozen
   screen. Fine-tuned without a teacher it beats TinyViT-21M, 0.712 against
   0.622 ([bake-off](reports/species-student-bakeoff.md)).
5. **Distillation**, each gain against the matching student without it
   ([distillation](reports/species-distill.md)):
   - plain KD at 25 epochs: -0.001, as it shifts accuracy from rare to common
     species;
   - KD weighted by class, so rare species count as in the label loss: +0.003;
   - 100 epochs instead of 25: +0.016 over label-only training at 100 epochs;
   - adding 12,126 unlabelled iNaturalist photos that the teacher labels (the
     transfer set): +0.033 over the previous step, to 0.768.

   This recipe is **KDw-100c+T**; seed 17 of it ships.
6. **Export.** int8 ONNX loses nothing on validation (0.7635 against 0.7631 for
   fp32), so int8 ships ([release](reports/species-release.md)).
7. **Test**, read once after everything was fixed ([test](reports/species-test.md)).

## Headline numbers

| Model | Validation | Test [95% CI] | Top-5 (test) |
| --- | ---: | --- | ---: |
| ResNet-18 baseline, full frame | 0.498 | 0.522 [0.503, 0.540] | 0.828 |
| DINOv2-S, labels only, 100 epochs (CE-100c), seed mean | 0.719 | 0.740 [0.721, 0.758] | 0.934 |
| DINOv2-S, KDw-100c+T, seed mean | 0.768 | 0.783 [0.763, 0.802] | 0.967 |
| **Shipped: KDw-100c+T seed 17, int8 ONNX** | **0.764** | **0.775 [0.752, 0.794]** | **0.962** |
| DINOv2-L teacher, 5-seed ensemble | 0.834 | 0.842 [0.822, 0.862] | 0.974 |

On test, the distilled student beats label-only training by +0.043 [+0.033,
+0.053], 5 of 5 seeds. Test scores run slightly above validation for every
model, which points to the split rather than to over-fitting on validation.

## The shipped artefact

`models/species-110/release/dinov2-s-kdw-e100-t-c576-seed17/` (not in Git):
`model-int8.onnx` (22.4 MB, median 118 ms on one AMD EPYC 7713 thread),
`model-fp32.onnx`, `labels.json` and `release.json` (checksums, metrics,
versions).

| | |
| --- | --- |
| Input `pixels` | float32, 1x3x224x224, NCHW, RGB, values in [0, 1] (pixel / 255) |
| Resize | short side to 255 px (bicubic), then centre crop 224 px |
| Normalisation | ImageNet mean/std, inside the graph |
| Output `logits` | 1x110, in `labels.json` order |
| Probabilities | `softmax(logits / 0.74)` |

## Known limits

- **Rare species are the weak spot.** The 23 species with under 40 training
  photos score 0.652 on test, against 0.837 for the 42 commonest; the teacher
  scores 0.777 on them.
- **No open-set handling.** The model always names one of the 110, including
  for other *Drosera*, other genera or no plant at all. There is no abstain
  threshold yet.
- **Browser behaviour is not measured.** Training read frames pre-shrunk with
  LANCZOS and ignored EXIF orientation, which browsers apply; int8 kernels
  differ between CPUs, and WebAssembly is untested. Latency was measured on one
  server core.
- **Balanced accuracy weights every species equally.** Users mostly photograph
  common species, so field accuracy will differ.

## Further reading

- [The plan](species-classifier-plan.md): the pre-registered decision record.
- Reports: [baseline](reports/species-110-baseline.md),
  [backbone screen](reports/species-backbone-screen.md),
  [teacher fine-tune](reports/species-finetune.md),
  [student bake-off](reports/species-student-bakeoff.md),
  [distillation](reports/species-distill.md),
  [release](reports/species-release.md),
  [held-out test](reports/species-test.md).
