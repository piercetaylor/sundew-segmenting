# Is the segmentation ready to drive a 10-20k species scrape?

Two experiments that answer that question, and a third that would only be needed
if the first two say no. Written after `reports/field-compare.md`, which left the
segmentation better than it had ever been and the scrape decision still unmade.

## The question, precisely

The species plan is crop-then-classify: segment a scraped photo, crop to the
predicted bounding box, hand the crop to a species classifier. The species label
comes from iNaturalist and is always correct, so a segmentation failure corrupts
the *input*, not the label. That is a milder failure than it first appears, and
it is why the decision turns on a rate rather than on mask quality.

`scrape-pipeline-readiness.md` measured 99.6% crop coverage on the curated test
split and concluded localisation was solved. `scrape-probe-50.md` showed that
conclusion does not transfer to uncurated images. The field measurement now
settles the middle ground:

| | Curated test split | `field-eval`, current best model |
| --- | ---: | ---: |
| Mean crop coverage | 99.6% | 95.4% |
| Coverage >= 90% | 28/28 | 95.3% of predictions |
| Bounding-box IoU | 0.79 | 0.69 |

Crop coverage is bimodal, which matters more than the mean. Sixteen of the 17
field-eval images score >= 95.9%; one scores 30.5%. There is no middle: the crop
either contains the plant or it does not.

**The blocker is not the rate, it is the uncertainty on the rate.** One failure
in 17 images gives a 95% Wilson interval of **[1.0%, 27.0%]** — between 104 and
2,698 unusable crops per 10,000. That is the difference between an affordable
nuisance and a ruined dataset, and the current evidence cannot tell them apart.

There are 21,360 research-grade, wild, CC0/CC-BY *Drosera* observations across
215 species, so 10-20k is close to the entire available pool rather than a
sample of it. 129 species have >= 10 observations; *D. rotundifolia* alone is
5,687 (27%).

## Step 1 — review 500 crops by eye

Estimating the failure rate does **not** require more masks. It requires more
*judgements*, and the judgement is binary: is the sundew inside this crop?

| Reviewed crops | Precision on the rate | Reviewer time at ~3 s each |
| ---: | --- | ---: |
| 200 | +/- 3.0 points | 10 min |
| **500** | **+/- 2.0 points** | **25 min** |

Against roughly 100 hours to annotate 500 masks, this is the same decision for
about 1/240th of the effort. Mask annotation cannot be justified until this
number exists, because the number determines whether more masks are needed at
all.

`scripts/hellbender_crop_review_scrape.slurm` pulls 500 fresh images with the
repository's own acquisition path under seed 20260921, with loose diversity caps
(25 per species, 3 per observer). The caps are deliberately looser than the
50-image probe's: the point is to sample what a real scrape would hit, and
capping hard would bias the failure rate away from the common species that
dominate the pool.

Then `scripts/render_crop_review.py` runs the current best checkpoint, builds
each padded crop exactly as a classifier pipeline would consume it, and renders
indexed review sheets. Review happens on the laptop; see
`docs/crop-review-handoff.md`.

**Decision rule, fixed before the data is seen:**

| Measured failure rate | Action |
| --- | --- |
| below 5% | start the scrape; the noise is affordable |
| 5-15% | judgement call against classifier tolerance; step 2 likely decides it |
| above 15% | fix hard negatives (step 3) before scraping |

## Step 2 — find out whether the crop is needed at all

The crop is a hypothesis, not a requirement. Nothing measured so far establishes
that cropping beats feeding the classifier a whole photograph. If it does not,
the crop failure rate stops mattering for the species project and segmentation
reverts to being purely about projected area.

This runs in parallel with step 1 because it can make step 1 moot.

`scripts/hellbender_species_scrape.slurm` acquires 200 images each of the ten
best-represented species (2,000 images) via `scripts/acquire_species_set.py`,
capped at 5 images per observer so the classifier split can be grouped by
observer exactly as the segmentation splits are. Then one classifier is trained
twice, changing only the input:

- **full frame** — the whole photograph, resized
- **crop** — the padded predicted bounding box from the segmentation model

Same architecture, same seeds, same observer-grouped split, paired across seeds.
`reports/field-compare.md` is the template: five seeds, paired, and report the
interval rather than a point value.

Expected readings:

| Result | Meaning |
| --- | --- |
| crop clearly ahead | the pipeline is justified; step 1's rate sets the risk |
| within noise | drop the crop; segmentation is not on the species critical path |
| full frame ahead | cropping is discarding useful context |

## Step 3 — hard negatives, only if step 1 says so

The one failure mode the field masks did not fix is bright artificial objects:
`inat_592910855`, IoU 0.000 at 0.957 confidence, unchanged by retraining. The
47-mask batch contained essentially one such example and the split placed it in
evaluation, so nothing taught the model about it.

If step 1 returns a rate above about 15%, annotate roughly 50 images containing
hands, tape measures, labels, pots, boots and neighbouring vegetation, where the
correct answer is often "almost nothing here". This is recommendation 2 of
`scrape-probe-50.md`, still unaddressed, now with a measured cost attached.

Target them at the failure mode rather than at volume. The evidence on that is
unusually clean:

| Added images | Effect on field IoU |
| --- | ---: |
| last 36 *curated* images (`data-sufficiency.md`) | -0.007 |
| 30 *field* images (`field-compare.md`) | **+0.061** |

The lever is distribution, not volume, and 50 targeted hard negatives are a
better use of a day than 250 more rosettes.

## What is deliberately not being done

- **Spending the curated test split.** It has been used once and stays locked.
- **More curated masks.** The curated scaling curve flattened at 108 images.
- **More seeds on the field comparison.** Five settled it; the eval set is now
  the limiting factor, not seed noise.
- **Trusting confidence as a filter.** It was 0.957 on the worst failure in the
  batch. This has now failed twice and should not be retried.

## Licensing

iNaturalist's terms prohibit using its data for commercial AI or machine
learning. Everything here stays personal noncommercial research. Accepting
NC/SA licences would raise the available pool from 21,360 to 130,193
observations, at the cost of a more restrictive downstream licence; that is a
deliberate decision, not a default.
