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
research claim that rests on a BioCLIP result must say so.

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
3. **Then resolution**, 224 -> 384, using the existing 768 px crops.
4. **Geo prior**, in parallel, as it is independent of the image model:
   re-fetch observation coordinates and fuse a spatial prior late.
5. **Reporting, once**: ensemble the five seeds, per-class accuracy against
   training count, and the most confused species pairs.

Not planned yet: more data, or hierarchical losses (modest gains in the
literature). Revisit once per-class results show where the errors come from.
