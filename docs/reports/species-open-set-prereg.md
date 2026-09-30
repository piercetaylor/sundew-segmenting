# Open-set test of the species model: pre-registration

Written 2026-09-30, **before** any open-set photo was pulled or scored. The
pass bar was set by the project owner.

## Question

How often does the shipped species model (v1.0.0, int8 ONNX) give a confident
sundew name to a photo that is not one of its 110 species? "Confident" means
the site's rule names a species: top-1 probability p >= 0.65, with
probabilities softmax(logits / 0.74)
([species-abstain.md](species-abstain.md)).

## Sources

All from iNaturalist, research grade, one photo per observation, licences
CC0, CC BY and CC BY-NC, coordinates neither requested nor stored.

| Source | What | Photos |
| --- | --- | ---: |
| d | *Drosera* identified to species rank, not one of the 110 (hybrids and genus-only IDs excluded; `without_taxon_id` also drops varieties of the 110) | 300 |
| e1 | Carnivorous look-alikes: *Pinguicula*, *Byblis*, *Roridula*, *Drosophyllum*, *Dionaea*, *Utricularia*, filled as evenly as their pools allow | 200 |
| e2 | Any other plant: kingdom Plantae without Droseraceae, Lentibulariaceae, Byblidaceae, Roridulaceae, Drosophyllaceae, Sarraceniaceae, Nepenthaceae and Cephalotaceae | 200 |

"e" means e1 and e2 together (400 photos).

**Sampling:** observation ids below a ceiling fetched once and recorded;
random windows (`id_above` drawn uniformly below the ceiling, seed 20260930),
200 observations per window, pooled and de-duplicated by observation, then
shuffled with the same seed and filled under the caps. Caps: 3 photos per
observer per source; in d, 30 photos per species. If a source's pool is too
small, it is reported short rather than filled from elsewhere.

**Exclusions, in order:** every observer in the train, validation and test
splits or the transfer set; every observation and photo id in those sets;
photos whose sha256 matches, or whose dHash is within 6 bits of, any photo in
those sets or an earlier photo in this set. The 20 Wikimedia Commons photos of
[species-out-of-list-check.md](species-out-of-list-check.md) can't overlap
(different source).

## Metric and bar

- **Primary:** false-accept rate on e = share of e photos with p >= 0.65.
  **Pass if it is below 20%** (point estimate). Reported with a 95% interval
  from 2,000 bootstrap resamples of observers.
- **Reported, no bar:** the same rate on d, e1 and e2 separately; the share
  getting the section fallback ("Probably section X", section p >= 0.7); and,
  for comparison, the in-list coverage at the same threshold (77.8% on
  validation).
- Scored **once**, with the shipped `release/species-v1.0.0/model-int8.onnx`
  in Python ONNX Runtime and the site's preprocessing (EXIF rotation, short
  side 255 px bicubic, centre crop 224). The threshold is not changed.

## If it fails

In order, each needing its own held-out photos (never tune and score on the
same photos):

1. Other scores from the same logits (max logit, energy
   `-T * logsumexp(logits / T)`), with the threshold picked on one half of e
   (split by observer) and scored on the other half, keeping in-list coverage
   on validation near 77.8%.
2. A small "sundew / not a sundew" gate on the DINOv2-S embedding, trained on
   a separate open-set pull and the train split.

**Addendum (2026-09-30, after the primary result failed at 20.3%, before any
fallback was computed).** Step 1 in detail:

- Scores from the same int8 logits: top-1 probability (the current rule), max
  logit, and negative energy `T * logsumexp(logits / T)`; higher means
  "answer". Each score's threshold is set on validation so that in-list
  coverage stays 77.8% (the current rule's), so users with listed sundews see
  no change.
- e is split in two by observer (seed 20260930). The score with the lowest
  false-accept rate on half A is chosen; its rate on half B, next to the
  current rule's rate on half B, is the result. It passes if half B is below
  20%. d is reported, no bar.
- If no score beats the current rule on half A, step 1 fails and the rule
  stays.

Until one passes, the site keeps the disclaimer that it may name a sundew for
other plants, sometimes with high confidence.
