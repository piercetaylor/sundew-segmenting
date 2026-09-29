# 50-image scrape probe: the curated test set was hiding failures

Fifty fresh images were pulled from iNaturalist with the repository's own
acquisition script (`scripts/acquire_inaturalist.py`, CC0/CC-BY only,
research-grade, non-captive) under a new seed, then segmented with the best
available checkpoint (`combined` recipe, validation IoU 0.683).

Contamination check against the 191-image labelled set: **0 photo IDs, 0
SHA-256 hashes, 0 observation IDs** in common. 31 species, of which 16 (17
images) never appear in the train split.

## The statistics look excellent

| | |
| --- | ---: |
| Mean predicted foreground | 18.1% (labelled set averages ~18%) |
| Near-empty masks (<0.5%) | 0 / 50 |
| Runaway masks (>60%) | 0 / 50 |
| Low confidence (<0.75) | 0 / 50 |
| Mean confidence, species **seen** in train | 0.890 |
| Mean confidence, species **unseen** in train | 0.886 |

On these numbers the model transfers perfectly, including to unseen species.

## Looking at the images says otherwise

Reviewing all 50 overlays by eye found failures the statistics do not register
(`reports/figures/scrape-probe-failures.jpg`):

- **Human skin.** In `inat_329881443.jpg` (*D. gunniana*) the mask covers the
  palm and fingers almost entirely while missing the small sundew being held.
  Predicted foreground 24.8%, mean confidence **0.818** — squarely inside the
  normal range.
- **Bright artificial objects.** In `inat_686282548.jpg` (*D. capillaris*) the
  mask lands on a green Stanley tape measure and misses the sundew rosettes on
  the sand entirely. One connected component, no flag raised.
- **Surrounding vegetation.** Several dense-scene images (e.g. *D. intermedia*
  at 45.5% foreground) pull in neighbouring non-sundew greenery.

Roughly 6-8 of the 50 images contain a hand or finger. Most are handled
correctly — the model segments the plant and leaves the skin alone. It is the
minority that fail, and they fail confidently.

Counts here are from one reviewer's eye on 50 images and should be treated as
order-of-magnitude: on the order of 3 hard failures and 5-8 over-inclusive
masks, not a measured rate.

### A failed measurement, recorded on purpose

A YCbCr skin-tone detector was written to quantify the skin false positives
automatically. It reported that 46 of 50 predicted masks sat >25% on skin, and
it was wrong: it matches sandy soil, brown leaf litter and orange rock. Checking
its eight highest-scoring images by eye, four contained a hand and four did not.
The result was discarded. Any future automated version of this check needs
validating against labelled examples before its output means anything.

## What this changes

`scrape-pipeline-readiness.md` concluded from the curated test split that crop
coverage was 99.6% and localisation was effectively solved, including on unseen
species. That conclusion **does not transfer to uncurated scraped images**, and
the test split could not have revealed it: curation filtered these cases out
before they ever reached the split.

The generalisation gap that matters is not species. It is capture conditions.
Unseen *species* cost ~0.11 mask IoU with no loss of localisation. Unseen
*contexts* — a hand, a tape measure, a crowded bog — cause confident, total
failures.

The model has partly learned "green blob" rather than "sundew". Curated
plant-centred images never made it pay for that.

## Recommended before a large scrape

1. **Do not trust confidence as a filter.** It was 0.818 on the worst failure.
2. **Annotate hard negatives.** Hands, tape measures, labels, pots, boots,
   neighbouring vegetation. A few dozen masks where the correct answer is
   "almost nothing here" would attack the actual weakness. This is a better use
   of 250 masks than anything proposed in `data-sufficiency.md`, which was
   written before this failure mode was visible.
3. **Review by eye at every scale-up.** `scripts/audit_scraped_predictions.py`
   renders the contact sheets; the statistics alone would have passed all 50 of
   these images.
