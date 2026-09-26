# Improving the 110-species classifier

Written 2026-09-23, after `reports/species-110-baseline.md` put the fine-tuned
ResNet-18 at **0.546** validation balanced accuracy on crops (0.525 on full
frames). This document records where the project stands, what an independent
review of the setup found, and the next experiment with its prior and decision
rule fixed before any of its data is seen.

Since 2026-09-24 the fine-tunes are compared against **0.516 crop / 0.499
full**, the same ResNet-18 rerun on `split-110-test`, not 0.546 / 0.525; see
step 1 under "After the screen".

## Where things stand

| Step | Result | Record |
| --- | --- | --- |
| Crop failure rate on uncurated photos | 2.43%, 95% CI [1.40%, 4.21%] | `reports/crop-review-result.md` |
| Does the crop help? (10 species) | +0.025 balanced acc, 5/5 seeds; cropping, not resampling | `reports/species-crop-comparison.md` |
| Scaled scrape | 110 species, 17,678 images, CC0/BY/NC | `docs/crop-readiness-plan.md` |
| Does the crop help? (110 species) | +0.021, 5/5 seeds; crop 0.546, full 0.525 | `reports/species-110-baseline.md` |
| Which backbone? (frozen screen) | BioCLIP-2 0.750, DINOv2-L 0.729; fine-tune both | `reports/species-backbone-screen.md` |
| Held-out test split | 2,601 images carved from train, grouped by observer | `reports/split-110-test.txt` |
| ResNet-18 anchor on `split-110-test` | crop 0.516, full 0.499; +0.017, 5/5 seeds | `reports/species-110-baseline.md` |
| Fine-tune script smoke test | passed (job 17929407), resume from `last.pt` verified | `reports/species-finetune-smoke.md` |

The segment -> crop -> classify pipeline is justified twice over. The weak
link is now the classifier: an 11M-parameter ResNet-18 from 2015, pretrained
on ImageNet-1k (almost no plant species), fine-tuned at 224 px.

## Independent review

A second model reviewed the training script, split and results read-only on
2026-09-23 and was asked to challenge rather than confirm. The points that
changed the plan:

- **The backbone is the biggest lever, and it needs its own recipe.** Agreed
  with the diagnosis; added that lr 3e-4 would wreck pretrained ViT or CLIP
  weights. A ViT fine-tune wants lr 2e-5 to 5e-5, warmup, layer-wise decay
  (~0.75), drop-path and bf16. Ranked that above resolution.
- **Colour jitter is already on** (`train_species_classifier.py:83`, hue 0.05).
  Whether pigment is being augmented away is a pending A/B, not a future risk.
- **Resolution has a ceiling.** Crops are stored as 768 px thumbnails
  (median short side 576, 7% under 384), so 384 px is safe, 448 marginal, and
  anything above needs re-rendering from the originals.
- **There is no test split.** Every checkpoint is selected on validation. The
  bias is small today (best vs last epoch 0.000-0.008) but grows with every
  configuration compared on the same split. A held-out test split is needed
  before the fine-tune stage.
- **Geography is an unused signal.** *Drosera* is strongly regional (WA pygmy
  and tuberous endemics, the Cape, Brazil). A spatial prior fused with the
  image model is well established for iNaturalist species ID (Mac Aodha et
  al. 2019; Cole et al. 2023, SINR). The scrape did not keep coordinates, so
  this needs a re-fetch; its size here is unknown.
- **Section-level accuracy says more than top-5.** It separates near-misses
  inside a section from real confusion. Now possible: all 110 species have a
  subgenus and section in iNaturalist's taxonomy
  (`data/species-110-sections.json`, 16 sections).
- **The full-frame control is still unrun.** Full frames are read with a centre
  crop that can cut off an off-centre plant. Squashing the whole frame to a
  square without cropping would show whether segmentation is still needed
  once the backbone is strong. Cheap to add to the screen, so it is included.

Verified before acting on: the colour-jitter line, the per-class counts
(train 29-240, validation 9-76), and the crop sizes (1,500-crop sample: longest
side 768, median short side 576, 7.3% under 384).

## The experiment: a frozen-backbone screen

Fine-tuning nine backbones × five seeds would cost days and make the most of
the validation-reuse problem above. Instead, freeze each pretrained backbone,
extract one embedding per image, and fit cheap readouts on top. A frozen probe
underestimates what fine-tuning reaches, so it ranks candidates without
promising a number.

| | |
| --- | --- |
| Script | `scripts/screen_species_backbones.py` (one model per call) |
| Job | `scripts/hellbender_backbone_screen.slurm`, array 0-8, one A100 each |
| Summary | `scripts/summarize_backbone_screen.py` -> `reports/species-backbone-screen/` |
| Resolution | 224 px for every model, so only the backbone varies |
| Arms | `crop`; `full` (Resize + CenterCrop, as fine-tuned); `full-square` (whole frame squashed, nothing cut) |
| Readouts | linear probe (primary); k-NN k=10 cosine; zero-shot for BioCLIP |
| Metrics | balanced accuracy (primary), accuracy, top-5, section-level balanced accuracy |
| Intervals | observer-grouped bootstrap on validation, 2,000 resamples, paired across models and arms |

Backbones, all verified to load at 224 px in the `sundew-seg` environment:

| Tag | Checkpoint | Pretraining | Params |
| --- | --- | --- | ---: |
| `resnet18-in1k` | `resnet18.tv_in1k` | ImageNet-1k — the anchor | 11M |
| `convnext-b-in22k` | `convnext_base.fb_in22k_ft_in1k_384` | ImageNet-22k | 88M |
| `dinov2-b` | `vit_base_patch14_dinov2.lvd142m` | self-supervised, 142M images | 86M |
| `dinov2-l-reg` | `vit_large_patch14_reg4_dinov2.lvd142m` | same, ViT-L with registers | 304M |
| `dinov3-vit-b` | `vit_base_patch16_dinov3.lvd1689m` | self-supervised, 1.7B images | 86M |
| `dinov3-convnext-b` | `convnext_base.dinov3_lvd1689m` | same, ConvNeXt | 88M |
| `siglip2-so400m` | `vit_so400m_patch14_siglip_378.v2_webli` | image-text, WebLI | 428M |
| `bioclip` | `hf-hub:imageomics/bioclip` | tree-of-life image-text | 86M |
| `bioclip-2` | `hf-hub:imageomics/bioclip-2` | same, larger corpus | 304M |

Design choices that protect the comparison:

- **Validation is scored once per model.** The one tuned setting, the
  logistic-regression L2 strength, is chosen by 3-fold cross-validation on
  the training split with folds grouped by observer. k-NN (k=10,
  temperature 0.07, as in DINO) and zero-shot have nothing to tune.
- **Same class weighting as the fine-tuned runs**, so thin classes count the
  same way in both.
- **Every arm sees the same 17,678 images**, or the script refuses to run.

**Contamination caveat.** BioCLIP and BioCLIP-2 were trained on iNaturalist
dumps, which may include some of these validation photos (87% of validation
images were observed in 2020 or later). That is fine for deployment, but it
makes their screen numbers optimistic by an amount we cannot measure. Any
research claim that rests on a BioCLIP result must say so. Measured
2026-09-26 for BioCLIP-2 by splitting validation at its training-data dates
(`reports/species-finetune.md`, "BioCLIP-2 contamination"): not detected,
bounded at about +0.017 fine-tuned and +0.061 frozen (`crop`).

### Prior, written before the run

- Frozen ResNet-18 lands well below its fine-tuned 0.546. ImageNet-1k features
  were never trained to separate sundews.
- At least one modern backbone beats the fine-tuned ResNet-18 while frozen.
  If none does, the backbone is not the ceiling.
- BioCLIP-2 and DINOv2-L are the most likely leaders, the first for domain
  (partly contamination), the second for fine-grained features.
- The crop - full gap shrinks for strong backbones (the reviewer's guess), and
  `full-square` beats `full`, if the centring hypothesis in
  `reports/species-crop-comparison.md` is right.

### Decision rule

Primary number: **linear-probe balanced accuracy on the crop arm.**

| Screen result | Action |
| --- | --- |
| No backbone beats frozen ResNet-18 with the paired CI excluding 0 | the backbone is not the lever; turn to data (geo prior, label noise, thin classes) |
| Best backbone beats the anchor but is below 0.546 | fine-tune the top two anyway; frozen probes understate fine-tuning |
| **Best backbone >= 0.546 frozen** | **fine-tune the top two, 5 seeds, ViT recipe** |
| Always | if the top two are both BioCLIP models, add the best non-BioCLIP as a third, because of the contamination caveat |
| On the leader, the paired CI on `crop` - `full-square` includes 0 or is negative | fine-tune a `full-square` arm too, before calling segmentation necessary |

k-NN, zero-shot, top-5 and section accuracy are diagnostics, not inputs to
the rule.

**Amendment, 2026-09-23, made after four of nine models had reported.** The
rule above tests the crop only against `full-square`, but the plain `full`
arm was the stronger full-frame control: frozen DINOv2-L scored `full` 0.730
against `crop` 0.729, while `full-square` trailed both. Every fine-tuned
backbone therefore also gets a **`full` arm** (Resize + CenterCrop, as in
the ResNet-18 baseline), five seeds, paired with `crop`, whatever the
remaining screen results are. The `full-square` row still applies as
written. This is a change made after seeing data. It adds a control and
removes no comparison, so it cannot favour the crop.

## After the screen

1. **Carve out a held-out test split** before any fine-tuning, grouped by
   observer, so the fine-tune comparison is not selected on the data it is
   scored on. **Done 2026-09-23:** `sundew-species-corpus/split-110-test`, by
   `scripts/carve_test_split.py`, per-class table in
   `reports/split-110-test.txt`.
   - Carved from **train**, not validation. Validation already picked the
     screened backbones, so photos from it are not untouched; it stays
     byte-identical, so every validation number to date still reads the same.
   - 15% of each species' total, hard per-class quota: train 11,118 / validation
     3,959 / test 2,601. Test holds 6-45 images per species (median 23) from
     1,095 observers, none shared with other splits. Train minimum falls from 29
     to 23 images per species.
   - Observers of the species with the fewest observers go first, random order
     within that. Pure random order starved the Western Australian endemics,
     whose photos come mostly from a few prolific observers (*D. nitidula*
     got 2 of 6).
   - Cost: every model trained on `split-110`, including the ResNet-18 baseline
     and the screen's probes, has seen the test photos and must not be scored
     on them. The fine-tunes train on 19% less data, so their anchor is a
     ResNet-18 rerun on `split-110-test` (the same 10-task array, split path
     changed), not 0.546. **Done 2026-09-24, job 17928590:** crop 0.516,
     full 0.499, crop - full +0.017 [+0.005, +0.029], 5/5 seeds
     (`reports/species-110-baseline.md`). 19% less training data cost
     0.026-0.030 on both arms; the crop effect held.
   - The test split is scored once, after every configuration is fixed.
     `scripts/finetune_species_backbone.py` drops test rows before reading
     anything.
2. **Fine-tune the chosen backbones**, `crop` and `full` arms (see the
   amendment), at 224 px with the ViT recipe above and
   hue jitter off. Five seeds, paired, as before.
   - **Ready to launch (2026-09-24).** The smoke test passed on its second
     attempt (job 17929407, `reports/species-finetune-smoke.md`): every
     library and arm runs, resume after preemption works, and DINOv2-L at
     batch 64 peaks at 20.7 GiB. Attempt 1 failed in the test harness, not
     the training script: it killed the run before the first checkpoint
     was written.
   - Launch, one submission per backbone, all three arms (indices 0-4
     `full`, 5-9 `crop`, 10-14 `full-square`; the screen's `full-square` rule
     fired, crop - full-square +0.006 [-0.009, +0.024] on BioCLIP-2):
     `sbatch --array=0-14 --export=ALL,MODEL=bioclip-2 scripts/hellbender_species_finetune.slurm`
     and the same with `MODEL=dinov2-l-reg`. 30 tasks, up to 5 h each.
   - **Done 2026-09-25, jobs 17943517 / 17943518** (`reports/species-finetune.md`):
     DINOv2-L `full` 0.824, `crop` 0.820; BioCLIP-2 0.802 / 0.799; anchor
     0.499 / 0.516. DINOv2-L beats BioCLIP-2 on every arm, 5/5 seeds, and is
     the teacher. `crop` - `full` is -0.004 [-0.012, +0.006] for DINOv2-L
     (observer bootstrap of the seed mean), so the crop buys nothing and is
     dropped from the species path: later stages use `full`.
   - **Audit, 2026-09-26** (`reports/species-finetune.md`, "Audit"): split
     verified clean; intervals switched to the observer bootstrap of the
     seed mean; TinyViT crash, `full-square` upsampling and requeue-retrain
     fixed in `scripts/finetune_species_backbone.py`.
3. **Then resolution**, 224 -> 384, on the `full` arm (the crop is dropped).

**Test protocol, fixed 2026-09-26 before any test image is read.** The test
split (2,601 images, 1,095 observers) is scored once, after the teacher
configuration is final (224 or 384 px, whichever the resolution rule picks),
and the result is reported whatever it is.

- Scored: the DINOv2-L `full` 5-seed last-epoch ensemble (mean softmax), its
  five single seeds, and the ResNet-18 anchor on `split-110-test` (`full` and
  `crop`, 5 seeds each). Last epoch, not best, so no choice is made on
  validation at the checkpoint level.
- Metrics: balanced accuracy (primary), plain accuracy, top-5, section-level
  balanced accuracy, and ECE before and after the temperature fitted on
  validation (T = 0.72 at 224 px; refitted on validation if the
  configuration changes). Intervals: observer-grouped bootstrap, 2,000
  resamples.
- Nothing is retrained or re-tuned after the test is read. If the test number
  falls outside the validation interval, that is reported, not corrected.
4. **Geo prior**, in parallel, as it is independent of the image model:
   re-fetch observation coordinates and fuse a spatial prior late.
5. **Reporting, once**: ensemble the five seeds, per-class accuracy against
   training count, and the most confused species pairs.

## Small models for an on-device student

Goal, from 2026-09-24: a classifier small enough to run in a browser or on a
phone and be shared with the carnivorous-plant community. The route is to
fine-tune the large backbones as a teacher and distil them into a small
student. A second model (Fable) was asked for an independent view and
recommended DINOv2 ViT-S as the first student: same lineage as DINOv2-L,
Apache-2.0 weights. Its estimate: 0.80-0.85 top-1 balanced accuracy is
plausible for the teacher, a student 3-8 points below it with distillation on
extra unlabeled photos, and 0.90 unlikely without merging species that cannot
be told apart from a photo.

**Small-backbone screen**, the same frozen protocol as above (224 px, split
`split-110`, linear probe, crop / full / full-square), 11 backbones of 8-35M
parameters: `dinov2-s`, `dinov2-s-reg`, `dinov3-vit-s`, `dinov3-vit-s-plus`,
`tinyvit-21m-in22k`, `convnext-nano-in12k`, `convnext-t-in22k`,
`mobilenetv4-conv-m-in12k`, `mobilenetv4-hybrid-m-in12k`,
`efficientnetv2-s-in21k`, `mobileclip2-s2` (image tower only, so no
zero-shot). Job 17943537, summary 17943538 ->
`reports/species-backbone-screen-small/`, which lists all 20 models.

Prior, written before the run: DINOv2-S or DINOv3-S leads the small models,
MobileNetV4 and EfficientNetV2 trail, and every small model lands below
DINOv2-B (0.675).

Decision rule, primary number linear-probe balanced accuracy on the crop arm:

| Best small backbone, frozen | Action |
| --- | --- |
| >= 0.58 (beats the fine-tuned ResNet-18 frozen) | on-device student is viable; that backbone is the student |
| 0.50-0.58 | viable only with distillation; decide after the teacher fine-tunes |
| < 0.50 | drop on-device for now; serve the fine-tuned teacher from a server |

Before building on it: DINOv3 weights carry Meta's own licence, with conditions
on derived models. A DINOv3 student is used only after that licence is checked
for free public release.

**Result, 2026-09-25** (`reports/species-backbone-screen.md`): `dinov2-s`
0.637 [0.613, 0.664], so the first row fires and DINOv2-S is the student.
`tinyvit-21m-in22k` (0.620) is not separated from it and ties on `full`
(0.614 vs 0.616). With the teacher now on `full` (above), which student to
distil into is open and is to be decided here before any distillation run.
The DINOv3 small models trailed (0.563, 0.543), so the licence check no longer
blocks anything.

### Student bake-off (written 2026-09-26, before any run)

The frozen screen cannot separate the two students on the arm the species
path now uses (`full`: DINOv2-S 0.616, TinyViT-21M 0.614), and a frozen
probe says little about fine-tuning. So both are fine-tuned, without a
teacher, and the better one becomes the student that distillation targets.

| | |
| --- | --- |
| Models | `dinov2-s` (`vit_small_patch14_dinov2.lvd142m`, 22M), `tinyvit-21m-in22k` (`tiny_vit_21m_224.dist_in22k`, 21M) |
| Arm, size, split | `full` only, 224 px, `split-110-test` (test rows dropped) |
| Seeds | 17 101 202 303 404, paired across models |
| Recipe | the teachers': AdamW lr 5e-5, 2 warmup, 25 epochs, patience 8, head lr x10, wd 0.05, label smoothing 0.05, bf16, batch 64. Both fall into the `LAYER_DECAY=0.75`, `DROP_PATH=0.1` branch of `scripts/hellbender_species_finetune.slurm` |
| Launch | `sbatch --array=0-4 --export=ALL,MODEL=dinov2-s scripts/hellbender_species_finetune.slurm` and `sbatch --array=0-4 --export=ALL,MODEL=tinyvit-21m-in22k scripts/hellbender_species_finetune.slurm` (indices 0-4 are `full`) |
| Cost | 10 A100 tasks. Small models are bound by JPEG decoding, like the ResNet-18 `full` arm (45-56 min per task), so ~45-60 min each: **8-10 A100-h**, about 1 h of wall time if the tasks run together |
| Summary | `scripts/summarize_species_finetune.py` has the teachers' models and arms hard-coded; it gets `--models` / `--arms` options before the results are read |

**Layer decay differs between the two, and is accepted as is.** Checked by
building both on CPU with `build()` from `scripts/finetune_species_backbone.py`
(timm 1.0.29, decay 0.75), mapping each optimizer group back to parameter
names:

| | DINOv2-S | TinyViT-21M |
| --- | --- | --- |
| Blocks | 12 | 12 (stages of 2, 2, 6, 2) |
| Layer-decay groups (distinct lr scales) | **13**: stem, 12 blocks | **14**: stem, 12 blocks, final norm |
| Last block | 1.0 (shares the top id with the final norm) | 0.75 (the final norm `head.norm` takes the top id alone) |
| Lowest scale | stem, 0.75^12 = 0.032 | stem, 0.75^13 = 0.024 |
| Downsampling layers | none | each merged with the first block of its stage |

So every TinyViT block trains at 0.75x the lr of the matching DINOv2-S block.
This is timm's `param_groups_layer_decay` grouping, not a choice made here,
and "the same recipe" is the design; changing it for one model would be a
tuning step. It is recorded as a caveat on any TinyViT loss, and does not
entitle TinyViT to a rerun.

**Prior, written now.** Relative to the teacher, DINOv2-L `full` 0.824:

- `dinov2-s` fine-tuned: **0.73-0.77** (5-9 points below the teacher).
  Frozen-to-fine-tuned gains were +0.09 for DINOv2-L; a smaller model has
  more to gain but less capacity.
- `tinyvit-21m-in22k` fine-tuned: **0.72-0.78**. Supervised ImageNet-22k
  features and a convolutional stem usually fine-tune well at this size; the
  lower lr above pulls the other way.
- The difference: within +-0.02, and more likely than not a tie by the rule
  below. No strong view on the sign.

**Primary metric**: seed-mean validation balanced accuracy, **last epoch**,
`full` arm. Interval: observer-grouped bootstrap of the seed-mean difference
(`dinov2-s` - `tinyvit-21m-in22k`), 2,000 resamples, all ten runs rescored on
each resample.

**Deployment tiebreaker**, defined now: **median single-image latency** of
the int8 ONNX model. Export each model's seed-17 checkpoint to ONNX (opset
17, 224 px, batch 1), quantise dynamically to int8 with ONNX Runtime, and
time 200 runs after 20 warm-up runs with `onnxruntime` CPUExecutionProvider,
one thread (a proxy for ONNX Runtime Web's single-thread WASM backend, the
floor every browser has), on one Hellbender CPU node, both models in the same
job. Measured by a script written for it, run by Claude; architecture alone
decides it, so it may be run before the accuracies are read. Two
disqualifiers apply first: the model does not export or does not run in
ONNX Runtime; or int8 costs it more than 0.01 validation balanced accuracy
against its own fp32 checkpoint (then its fp32 latency is used instead).

**Licences**: DINOv2-S weights Apache-2.0 (Meta; timm card); TinyViT weights
Apache-2.0 on the timm card, MIT upstream (microsoft/Cream). Both permit
free public release, so **licence is not a tiebreaker.**

| Bake-off result | Action |
| --- | --- |
| CI on the difference excludes 0 | the higher model is the student |
| CI includes 0, one model's int8 median latency is >= 15% lower | the faster model is the student |
| CI includes 0, latencies within 15% | `dinov2-s`: the rule already named it, and it shares the teacher's lineage and patch features, which feature distillation can use |
| Better student < **0.70** (more than 12 points below the teacher) | no student ships without distillation; distillation becomes mandatory before release |
| Better student < **0.65** | stop the on-device line; serve the fine-tuned teacher from a server, and revisit after distillation and the open-set work |
| A model's runs fail or diverge (seed SD > 0.02) | reported as failed, not retuned; the other model is the student if it clears the floors |

What this run does not decide:

- **Distillation**: whether, and by how much, the teacher ensemble helps the
  student. That is the next pre-registered run, paired against this one.
- **Resolution**: the student stays at 224 px here; 384 px is for the
  teacher first.
- **Calibration of the student**: its temperature is fitted later, on
  validation, as for the teacher.
- **Browser behaviour on real devices**: WebGPU availability, memory and
  real phone latency are measured on the chosen student only, after this run.

Not planned yet: more data, or hierarchical losses (modest gains in the
literature). Revisit once per-class results show where the errors come from.
