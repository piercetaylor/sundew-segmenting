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
| Summary | `python scripts/summarize_species_finetune.py --models dinov2-s tinyvit-21m-in22k --arms full --title "Student bake-off" --report reports/species-student-bakeoff.md --out-md reports/species-student-bakeoff/results.md --out-json reports/species-student-bakeoff/summary.json` (options added before any result exists; the difference reads `dinov2-s` - `tinyvit-21m-in22k`) |

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

**Result, 2026-09-27** (`reports/species-student-bakeoff.md`, jobs 17987233 /
17987234): `dinov2-s` 0.712 [0.691, 0.736], `tinyvit-21m-in22k` 0.622
[0.602, 0.651], last-epoch seed means; difference +0.090 [+0.075, +0.102].
The first row fires: **DINOv2-S is the student**. It clears the 0.70 floor
by 0.012, so distillation is not mandatory by the rule. The latency
tiebreaker was not needed and not run. TinyViT under-fitted under the shared
recipe (train loss 1.44 vs 0.94 at epoch 25; +0.008 over its frozen probe),
as the accepted layer-decay handicap would predict; per the rule it is not
retuned. The prior missed: both students landed below their ranges, and the
expected tie was a 0.09 gap.

### Distillation into DINOv2-S (written 2026-09-27, before any run)

The student trained on labels alone reaches 0.712, 0.11 below the teacher.
This run asks whether the teacher closes that gap, and by how much. The
attribution is kept clean: the arms differ only in the training target, and
separately in schedule length. Fable reviewed the design read-only on
2026-09-27 and checked it against the literature; its recommendations are
adopted except where stated.

| | |
| --- | --- |
| Teacher | `dinov2-l-reg` `full`, the five seeds' `classifier-best.pt` (epochs 24, 18, 24, 24, 18), mean softmax, eval mode, bf16, no gradient. Validation 0.834 as an ensemble (0.835 from the last-epoch predictions; last-epoch weights were not kept) |
| Student | `dinov2-s` (`vit_small_patch14_dinov2.lvd142m`), `full` arm, 224 px, `split-110-test` (test rows dropped) |
| Seeds | 17 101 202 303 404, paired with the bake-off's `dinov2-s` runs |
| Recipe | the bake-off's, unchanged: AdamW lr 5e-5, 2 warmup epochs, cosine, layer decay 0.75, drop-path 0.1, head lr x10, wd 0.05, bf16, batch 64, the same augmentation and the same image pipeline (original JPEGs) |
| KD target | KL(teacher ‖ student) against the ensemble's mean softmax, tau = 1, no hard-label term (alpha = 0), no label smoothing, no class weights on the KD term. Loss in fp32 |
| Views | online and consistent: the same augmented batch goes to the five teachers and the student |
| Patience | off (= epochs) in every new arm, so "last epoch" always means the end of the schedule. No bake-off run stopped early, so the reused control is unaffected |

Arms. `CE` is the bake-off loss: class-weighted cross-entropy with label smoothing 0.05.

| Arm | Output tag | Target | Epochs | Status | Role |
| --- | --- | --- | ---: | --- | --- |
| CE-25 | `dinov2-s` | CE | 25 | **exists**, 0.712 | paired control for KD-25 |
| KD-25 | `dinov2-s-kd` | KD | 25 | new | **primary**: the effect of the teacher at a matched schedule |
| CE-100 | `dinov2-s-e100` | CE | 100 | new | matched-length control for KD-100 |
| KD-100 | `dinov2-s-kd-e100` | KD | 100 | new | the effect of the teacher when training runs longer; also the best available student |

**The pairing is tight.** Data order, augmentation and drop-path are seeded
per epoch (`seed * 1000 + epoch`), and teachers in eval mode draw no random
numbers. So a KD-25 seed sees the same batches and views as its CE-25 seed.
The teachers are built after the student, so the student's initial weights,
head included, are the same too. The smoke test checks this.

**Choices, and why:**

- **Pure KL, tau 1**, as in Beyer et al. 2022 (arXiv:2106.05237), who used no
  label term. A label term would pull the student toward iNaturalist labels
  the teacher doubts (the *peltata* / *auriculata* / *lunata* complex). It
  would also need its own class weighting, a second knob.
- **The under-confident teacher is not sharpened.** T = 0.72 was fitted on
  validation, so putting it into the training target would let validation
  shape training. Sharpening does not change the target's argmax. A
  better-calibrated target may help (Menon et al. 2021, arXiv:2005.10419), so
  it is a candidate for a later, separate run.
- **The KD term is not class-weighted.** The teacher was trained with class
  weights, so its posteriors already carry the balanced prior, and the student
  inherits it by matching them. Weighting again would count the prior twice.
  The imbalance is also mild, 23-195 training images per species. This is
  checked by a diagnostic, below.
- **Label smoothing in the teacher** (0.05) can make a teacher distil worse
  (Müller et al. 2019, arXiv:1906.02629; disputed by Shen et al. 2021,
  arXiv:2104.00676). It is mild here: the teacher's training loss, 0.72,
  stays well above the 0.43 floor. It is recorded, and the teacher is not
  retrained.
- **No mixup or CutMix.** Adding either to the KD arms alone would confound
  the target with the augmentation, and a CE + mixup control would then be
  needed. Function matching with mixup is deferred.
- **Logits only.** Patch-token distillation (ViTKD, arXiv:2209.02432) needs a
  1024 -> 384 projector, a way to handle the teacher's 4 registers, and a
  loss weight: three untuned choices. DINOv2-S is also already a
  feature-distilled child of DINOv2-g.
- **The 224 px teacher, now.** The recipe questions (loss, schedule) do not
  depend on teacher resolution. If the 384 px teacher wins its own rule, the
  final student is distilled once more from it, with the recipe that wins
  here and no further changes.
- **No extra photos.** The teacher saw every training image, so its targets
  there are sharper than on unseen photos. Unlabeled transfer data would
  likely help (Beyer et al. 2022; Stanton et al. 2021, arXiv:2106.05945), but
  that would change the data and the teacher at once. It gets its own run
  (below).
- **The original JPEG pipeline is kept**, although it bounds each epoch by
  decoding (about 105 s). A faster loader (`Image.draft`, or a 768 px cache)
  would change the downsampling path, so CE-25 could no longer be reused as
  the control and would have to be rerun.

**Recorded leak.** The teacher checkpoints were picked on validation (best
epoch). On each seed, best and last differ by at most 0.0012, and the two
ensembles by 0.0006, so the leak is negligible but real.

**Implementation, written and smoke-tested before launch.**
`scripts/finetune_species_backbone.py` gets `--teacher-checkpoints` (the
five paths) and `--kd-tau` (default 1.0). With teachers given, the loss is
the KD target above, and `classifier-metrics.json` records the teacher
paths, tau and the loss. `scripts/hellbender_species_finetune.slurm` gets
`KD=1` and `EPOCHS=` switches, which set the output tag and patience. The
walltime is 8 h for the 100-epoch arms, which resume from `last.pt` after
preemption. The smoke test (`--limit`) must show four things:

- the step-0 student weights equal the CE-25 seed's;
- the teacher ensemble, rebuilt from the checkpoints, scores 0.834 +- 0.002
  on validation;
- the KD loss falls;
- the measured seconds per epoch.

**Smoke test passed, 2026-09-27** (job 18012907,
`scripts/hellbender_species_distill_smoke.slurm`, 512 images per split). The
student init hash was identical with and without teachers. On those 512
images the rebuilt teacher ensemble agreed with the saved best-epoch
predictions on every top-1 prediction, with a maximum probability difference
of 0.0000. The KD loss fell from 4.88 to 4.27, last-epoch weights were
written, and a KD run killed after epoch 1 resumed at epoch 2. Peak GPU
memory was 9.5 GiB with teachers, against 3.3 GiB without. Seconds per epoch
come from the first real task, since 512 images is too few to time.
**KD-25 launched** as job 18012985 (`--array=0-4`, `KD=1`).

**Result, KD-25, 2026-09-27** (`reports/species-distill.md`): 0.711 [0.688,
0.736] against CE-25 0.712. **KD - CE -0.001 [-0.008, +0.006]**, 0 of 5
seeds. The first rule does not fire, and the prior (0.735) missed. The
per-class diagnostic shows why the net is zero:

- species with under 40 training images: KD -0.039 [-0.066, -0.022];
- species with 130 or more: KD +0.020 [+0.015, +0.024];
- plain accuracy +0.009, and single-model ECE 0.030 against 0.079.

The unweighted KD term shifted accuracy toward the common species, so the
**class-weighted KD follow-up named above is triggered**.

**Amendment, 2026-09-27, made after seeing KD-25 and before any further
run.**

1. **A fast full-frame cache.** Every run is bound by decoding the original
   JPEGs (105-120 s per epoch). `scripts/make_full_cache.py` writes train and
   validation frames once, short side 576 px:
   - LANCZOS resize, as the crops and the `full768` control used; that
     control changed ResNet-18 by +0.0001 (`reports/species-crop-comparison.md`);
   - no EXIF rotation, as the loader applies none;
   - JPEG quality 95, 4:4:4 chroma.

   The test split is cached only when it is scored. CE-25 is rerun on the
   cache (`CACHE=1`, tag `dinov2-s-c576`, 5 seeds, otherwise identical).
   **Equivalence rule, fixed now:**
   - the cache is adopted if the observer-bootstrap 95% CI of
     (`dinov2-s-c576` - `dinov2-s`), last-epoch seed means, lies inside
     **[-0.015, +0.015]**. The rerun then replaces CE-25 as the control for
     every later arm, and those arms run on the cache;
   - otherwise the cache is dropped and the originals stay.

   A CI covering 0 is not enough on its own, because a wide interval would
   pass. Cache build: job 18014841 (`scripts/hellbender_full_cache.slurm`).
   CE-25 rerun: job 18014847, which starts only if the build succeeds.
2. **The 100-epoch arms move onto the cache** if it is adopted. They are not
   launched on the originals: CE-100 plus KD-100 would cost about 32 A100-h
   there.
3. **Class-weighted KD** (the follow-up above) and a **transfer set** of
   photos the teacher never trained on are each added as arms. Each gets its
   design, prior and rule written here before it runs. The existing rule
   (KD-100 against CE-100; stop if both differences < +0.01) is unchanged.

The cache was built 2026-09-27 (job 18014841, 2.5 min): 15,077 train and
validation frames, 4.4 GB.

### Class-weighted KD (written 2026-09-27, before any run)

KD-25 lost 0.039 on the 23 species with under 40 training images and gained
0.020 on the 42 commonest. The pre-registration named class-weighted KD as
the follow-up for exactly that. Fable designed the arm (read-only review,
2026-09-27), and the design is adopted as below.

**Loss.** KDw is the KD loss with each image's KL weighted by the CE
baseline's class weight of its true label, w_c = N / (C n_c), normalised as
the weighted cross-entropy is:

    loss = sum_i w_{y_i} KL_i / sum_i w_{y_i}

- **It differs from CE-25 only in the target:** the teacher's mean softmax
  instead of the smoothed one-hot. Batches, views and starting weights stay
  paired seed for seed.
- **In expectation it equals class-balanced resampling**, without touching
  the seeded batches. Resampling would also repeat the 23-image species about
  8x, which Kang et al. 2020 (arXiv:1910.09217) show hurts the learned
  features.
- **Rejected alternatives:**
  - logit adjustment (Menon et al. 2021, arXiv:2007.07314): the teacher's
    posteriors already carry the balanced prior; the failure is how much
    gradient each species gets;
  - BKD (Zhang et al. 2021, arXiv:2104.10510): adds a label term and a
    second weighting;
  - KL plus weighted CE: brings back a label term and an alpha to choose.
- **Inverse frequency, not effective number** (Cui et al. 2019,
  arXiv:1901.05555): at 23-195 images per species the two are within 10%,
  and matching the baseline matters more.
- **It is class-balanced KD, not label-free KD.** The label enters only as a
  per-image scale, the same information CE's weighting uses.
- **Fixed now for the later transfer set.** Unlabeled images are weighted by
  the class weight expected under the teacher, w_i = sum_c t_ic w_c
  (`--kd-class-weight expected`). Weighting them by 1 would bring back
  iNaturalist's popularity skew, which is what just failed. Every KD run now
  logs how far this weight is from the label weight on the training images,
  and how often the teacher's top-1 disagrees with the label.

**Pipeline.** All arms here run on the 576 px cache and are **launched only
if the cache is adopted** by the equivalence rule above. If it is not, they
run on the originals and the costs roughly double. Before any KD arm on the
cache, the teacher ensemble must score 0.834 +- 0.002 on the cached
validation frames. Matching CE on the cache does not show that the teachers
respond the same way to the resampled pixels.

**Arms.** Five seeds, `full`, 224 px; the recipe is otherwise the
bake-off's.

| Arm | Tag | Target, weight | Epochs | A100-h | Role |
| --- | --- | --- | ---: | ---: | --- |
| CE-25c | `dinov2-s-c576` | class-weighted CE, LS 0.05 | 25 | ~1 (job 18014847) | control, and the cache's equivalence check |
| KD-25c | `dinov2-s-kd-c576` | KL, unweighted | 25 | ~3 | KD-25 on the cache: the base for the weighting contrast |
| **KDw-25c** | `dinov2-s-kdw-c576` | KL, weighted by w_{y_i} | 25 | ~3 | **primary**: class-balanced KD against class-weighted CE |
| CE-100c | `dinov2-s-e100-c576` | as CE-25c, patience off | 100 | ~4 | schedule control |
| KD(w)-100c | `dinov2-s-kd[w]-e100-c576` | the selected KD variant (below) | 100 | ~11 each | the teacher at length; best available student |

- **KD-25 is rerun on the cache** so that KDw - KD compares one pipeline.
- **Sequential design, fixed now.** The three 25-epoch arms run first. Then:
  - CE-100c always runs;
  - the KD variant with the higher 25-epoch seed mean goes to 100 epochs;
  - both go if the CI on their difference covers 0.

  Both 25-epoch results are reported whatever is selected.
- **Cost:** 25-epoch arms about 7 A100-h; with the 100-epoch arms, 22 (one
  KD variant) to 33 (both).

**Smoke test passed, 2026-09-27** (job 18014924,
`scripts/hellbender_species_kdw_smoke.slurm`):

- the teacher ensemble scores **0.8344** on the full cached validation,
  against 0.8341 on the originals;
- the KDw run has the same student init hash as every earlier seed-17 run,
  its loss falls (5.25 -> 4.76), and it records `class_weight: label`;
- the expected-weight mode runs;
- on the 512 training images of the smoke subset, the teacher's top-1 never
  disagrees with the label, and the mean |w_label - w_expected| is 0.13;
- epochs on the cache take about 7 s for 512 images. The CE-25c tasks take
  12 min, against 43 min on the originals.

Launch, once the cache is adopted:
`sbatch --array=0-4 --export=ALL,MODEL=dinov2-s,KD=1,CACHE=1 scripts/hellbender_species_finetune.slurm`
and the same with `KD_WEIGHT=label`.

**Prior, written now.** The honest central case is that weighting reverses
the reallocation and the net stays near 0.

| Arm | Point | Range |
| --- | ---: | --- |
| CE-25c | 0.712 | 0.70-0.725 |
| KD-25c | 0.711 | 0.70-0.72 |
| KDw-25c | 0.72 | 0.705-0.735 |
| CE-100c | 0.725 | 0.71-0.735 |
| KD-100c | 0.73 | 0.715-0.75 |
| KDw-100c | 0.74 | 0.72-0.76 |

Per bin, KDw - CE:

- **species with under 40 images: within +-0.015.** They recover to the CE
  level, not above it, since weighting adds no information there;
- **species with 130 or more: +0.005 to +0.015.** Part of KD's +0.020 was
  gradient mass, which weighting removes;
- **prediction share** of the commonest species back to about 0.60.

A rival reading, which the prior does not exclude: at 25 epochs the student
under-fits, and common species simply supply more views of a richer target;
weighting cannot add views to thin species. If KDw lands at CE, that is what
happened, and the remedy is the transfer set, capped per species.

**Decision rule.**

| Result | Action |
| --- | --- |
| KDw-25c - CE-25c: CI excludes 0, positive | class-balanced distillation is in the student recipe |
| KDw-25c - KD-25c on the under-40 bin: CI excludes 0, positive | the weighting is the fix for thin species; KDw is the KD form from here on |
| KDw-25c within +-0.01 of CE-25c | weighting is kept (the fair comparator for a balanced metric); the teacher is not the lever at 25 epochs |
| KD(w)-100c - CE-100c < +0.01 | distillation on the training set is closed; the transfer set is the last distillation arm, and if it too gains < +0.01, distillation is closed and the best CE arm is carried forward |
| Carried-forward recipe | the highest last-epoch seed mean among the cache arms; on a tie (CI covering 0), the KD variant, since it is better calibrated and free at inference |
| Shipped model | seed 17 of the carried-forward arm; release floor 0.75 and top-5 >= 0.95, unchanged |
| A run diverges (seed SD > 0.02) or fails | reported, not retuned |

**Diagnostics, fixed now, not inputs to the rule:**

- the four training-count bins, per arm and for KDw - KD and KDw - CE. Only
  the two extreme bins carry a claim, since the per-bin intervals are not
  corrected for four comparisons;
- single-model ECE before and after a validation temperature, and mean
  confidence. Expected: KDw between KD (0.030) and CE (0.079);
- the prediction share of the commonest species, against the true 0.599;
- fidelity: top-1 agreement, and the unweighted KL on validation, overall
  and per bin;
- the weight statistics above, and the training loss;
- int8 export of seed 17 of the carried-forward arm.

**Risks recorded:**

- **Weights reach 4.4 on the 23-image species.** The *peltata* complex
  (111-188 images) sits at 0.5-0.9, so it is not amplified. A mislabelled
  thin-species image is pulled toward the teacher's view under KDw, but
  toward the wrong label under CE.
- **The per-batch normalisation varies the effective step with batch
  composition.** CE has the same variation, so KDw and CE are matched. KDw
  and KD, however, differ in effective step as well as in weighting, and
  that contrast folds both in.
- **Choosing which KD variant goes to 100 epochs is a validation-informed
  step**, made legitimate by the rule fixed above.

**Cost**, from the decode-bound epoch (105 s). Five ViT-L forwards per step
roughly equal one ViT-L training step, which fit under the same bound, so a
KD epoch is estimated at 110-180 s.

| Arm | A100-h |
| --- | ---: |
| KD-25 | 4-6 |
| CE-100 | ~15 |
| KD-100 | 15-25 |
| **Total** | **34-46** |

This is more than the 15 A100-h pencilled in earlier; the matched-length
control is what costs. KD-25 answers the primary question alone for 4-6.

**Primary metric**: seed-mean validation balanced accuracy at the **last
epoch**, `full`. Intervals: observer-grouped bootstrap of the seed-mean
difference, 2,000 resamples, all runs rescored on each resample (as in the
bake-off). Summary:
`python scripts/summarize_species_finetune.py --models dinov2-s-kd dinov2-s --arms full ...`
and the same for `dinov2-s-kd-e100 dinov2-s-e100`.

**Prior, written now.** Last time both students landed below their ranges,
so the ranges are wide.

| Arm | Point | Range |
| --- | ---: | --- |
| CE-25 | 0.712 | (measured) |
| KD-25 | 0.735 | 0.72-0.75 |
| CE-100 | 0.725 | 0.71-0.735 |
| KD-100 | 0.76 | 0.74-0.78 |

- KD - CE is larger at 100 epochs than at 25 (the teacher is "patient"; Beyer
  et al.).
- A single student stays 5-9 points below the teacher's seed mean (0.824).
  Fable's estimate agrees, with points 0.74 and 0.765 for the KD arms.

**Decision rule.**

| Result | Action |
| --- | --- |
| KD-25 - CE-25: CI excludes 0, positive | the teacher helps at a matched schedule: distillation is in the student recipe |
| KD-100 - CE-100: CI excludes 0, positive | the teacher helps at the long schedule too |
| Both differences < +0.01 (point estimates) | the teacher is not the lever; the next student run is the transfer set, not a recipe change |
| Carried-forward recipe | the arm with the highest last-epoch seed mean among KD-25, CE-100 and KD-100; but if CE-100 is highest and the CI on its lead over the best KD arm includes 0, the KD arm is carried forward (the teacher is already paid for, and costs nothing at inference) |
| Shipped model | **seed 17** of the carried-forward arm, fixed now, not the best seed. Only one model ships, so the seed mean is the number that describes it; the 5-seed student ensemble is reported but not shipped |
| Release floor | the carried-forward seed mean >= **0.75** and top-5 >= 0.95. Below that, nothing ships until the transfer-set run |
| A run diverges (seed SD > 0.02) or fails | reported as failed, not retuned |

**Diagnostics, not inputs to the rule**:

- teacher-student top-1 agreement and mean KL on validation (fidelity; Stanton
  et al. 2021);
- section-level balanced accuracy and top-5;
- per-class accuracy against training count for the teacher, CE and KD. If
  KD loses to CE on species with under 40 training images, a class-weighted
  KD term is the follow-up;
- the student's ECE before and after a temperature fitted on validation;
- int8 ONNX export of seed 17 of the carried-forward arm, including the
  accuracy change from int8, since KD changes the logit scale.

**Not decided here, each for its own pre-registration:**

- a transfer set of unlabeled photos: beyond the 300 cap, "needs ID",
  cultivated (closer to what app users photograph), other *Drosera*. Every
  validation and test observer is excluded, and photos are dHash-deduplicated
  against both splits;
- mixup / function matching, with a CE + mixup control;
- a sharpened target;
- patch-token distillation, if the gap to the teacher stays above 0.05;
- re-distilling from the 384 px teacher, if it wins its rule. That fine-tune
  must save its last-epoch weights (see the test-protocol amendment).

**Amendment to the test protocol, 2026-09-27, before any test image is
read.** The protocol above scores the teacher's "last-epoch ensemble", but
`finetune_species_backbone.py` deletes `last.pt` on completion, so only
best-epoch weights exist. The test is therefore scored on the **best-epoch
ensemble**. Its validation optimism is bounded by best - last: at most 0.0012
per seed, and 0.834 against 0.835 for the ensemble. Retraining at 224 px to
recover last-epoch weights is not worth ~25 A100-h. If the 384 px teacher
replaces this one, its script saves last-epoch weights and the protocol
applies as first written. Distilled students save their last-epoch weights
too, since seed 17 is the model that ships.

Not planned yet: more data, or hierarchical losses (modest gains in the
literature). Revisit once per-class results show where the errors come from.
