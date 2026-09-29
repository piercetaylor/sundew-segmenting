# Crop review: the crop misses the plant on 2.4% of uncurated photos

**In short.** On 493 uncurated scraped photos, the crop missed the sundew 12
times: 2.43%, 95% CI [1.40%, 4.21%]. The whole interval is below the 5% limit
set in advance, so the species scrape went ahead. A further 17 photos (3.45%)
are too poor to identify to species by eye; that is a property of the photo,
not a segmentation failure.

Crops came from the best checkpoint at the time
(`models/field-compare/field/seed-101`, threshold 0.6), exactly as a
crop-then-classify pipeline would use them, and were reviewed by eye on
2026-09-21. The raw review lists are in [crop-review/](crop-review/README.md).
Reproduce with `python scripts/crop_review_stats.py`.

The review returned **two** lists that answer different questions:

| List | Question it answers | n |
| --- | --- | ---: |
| `failures.txt` | is the sundew inside this crop? | 12 |
| `uncertain.txt` | could the reviewer identify the species from this photograph? | 17 |

Only the first is a segmentation failure.

| | |
| --- | ---: |
| Reviewed | 493 |
| Crop failures | **12** |
| **Rate** | **2.43%** |
| 95% Wilson CI | **[1.40%, 4.21%]** |
| Per 10,000 scraped images | ~243 bad crops (139-420) |

The earlier estimate, from one failure in the 17-image `field-eval` set, was
5.9% with a 95% interval of [1.0%, 27.0%]. The new interval is about eight
times tighter.

> A first pass of this report scored an incomplete 11-line `failures.txt` and
> said no `uncertain.txt` existed. Index 191 is the added failure. The headline
> moved from 2.23% to 2.43%; the conclusion did not change.

## The decision

[`deprecated/docs/crop-readiness-plan.md`](../../deprecated/docs/crop-readiness-plan.md)
fixed the rule before the data was seen:

| Measured rate | Action |
| --- | --- |
| **below 5%** | **start the scrape; the crop noise is affordable** |
| 5-15% | judgement call against classifier tolerance |
| above 15% | fix hard negatives first |

**2.43% clears it, and so does the upper bound (4.21%).**

The result holds for every reasonable treatment of the 17 uncertain images:

| Treatment | Rate | 95% CI | Clears 5%? |
| --- | ---: | --- | --- |
| Uncertain are not crop failures (**used**) | 12/493 = 2.43% | [1.40%, 4.21%] | yes, interval and all |
| Uncertain excluded from the denominator | 12/476 = 2.52% | [1.45%, 4.35%] | yes, interval and all |
| Uncertain counted as crop failures | 29/493 = 5.88% | [4.13%, 8.32%] | no |

The third row is not the right reading. The protocol in
[`deprecated/docs/crop-review-handoff.md`](../../deprecated/docs/crop-review-handoff.md)
defined uncertain as *"too blurry to tell whether a sundew is in the crop"*, a
question about containment. The reviewer used a different, more useful
criterion: *"those I would not be able to ID confidently from a photo...
because the photo was blurry, poor quality, or obscured by grass."* That is
about identifying the species. A crop can contain the plant and still be
unidentifiable, and 16 of the 17 have confidence and box geometry
indistinguishable from the passes. Counting them as crop failures would blame
the segmenter for the photographer's focus.

For scale:
[`deprecated/reports/species-crop-comparison.md`](../../deprecated/reports/species-crop-comparison.md)
found the crop is worth +0.0246 balanced accuracy over full frames. A 2%
failure rate is the price of a 2.5-point gain, and it corrupts the classifier's
*input* while its *label*, from iNaturalist, stays correct.

The review took about 25 minutes, against roughly 100 hours to annotate 500
masks.

## The twelve failures

Growth forms come from `src/sundew_segmentation/growth_forms.py`, not from hand
labels (the first pass hand-labelled two rows wrongly; the stratified counts
below were always computed from the code).

| Index | Species | Growth form | Box | Confidence |
| ---: | --- | --- | ---: | ---: |
| 58 | *D. stricticaulis* | unknown | 68% | 0.868 |
| 103 | *D. humilis* | unknown | 26% | 0.679 |
| 191 | *D. arcturi* | linear/forked | 25% | 0.943 |
| 339 | *D. erythrorhiza* | rosette | 100% | 0.865 |
| 354 | *D. filiformis* | linear/forked | 92% | 0.929 |
| 375 | *D. finlaysonii* | unknown | 69% | 0.900 |
| 435 | *D. linearis* | linear/forked | 82% | 0.870 |
| 442 | *D. filiformis* | linear/forked | 73% | 0.924 |
| 459 | *D. auriculata* | erect/branching | 95% | 0.928 |
| 462 | *D. anglica* | rosette | 100% | 0.934 |
| 480 | *D. intermedia* | erect/branching | 58% | 0.952 |
| 481 | *D. anglica* | rosette | 100% | 0.856 |

Four of the twelve boxes cover 92-100% of the frame: the model selected
everything and located nothing. Two (103, 191) cover under 30%: it locked onto
a fragment. These are the same two opposite failure modes that
[`deprecated/reports/field-probe-evaluation.md`](../../deprecated/reports/field-probe-evaluation.md)
found on `field-eval`, and they cancel out in any mean.

## By growth form

| Growth form | Failures | Rate | 95% CI |
| --- | ---: | ---: | --- |
| linear or forked | 4/78 | **5.13%** | [2.0%, 12.5%] |
| unknown | 3/110 | 2.73% | [0.9%, 7.7%] |
| erect or branching | 2/80 | 2.50% | [0.7%, 8.7%] |
| rosette | 3/214 | 1.40% | [0.5%, 4.0%] |
| dense mat | 0/11 | 0.00% | [0.0%, 25.9%] |

Linear or forked plants fail 3.7 times as often as rosettes. The intervals
overlap, so this data alone does not establish it, but it is the fourth
independent hint in the same direction:
[baseline-seed-sweep.md](baseline-seed-sweep.md) found `linear_or_forked` the
weakest form for both architectures,
[segmentation-recipe.md](segmentation-recipe.md) found both unseen-species test
failures were linear or forked, and it is the least represented form in
training (17 of 191 pairs).

The same row is also worst on identifiability (9 of 78 are a failure or
uncertain, 11.5%). That is likely morphology, not extra evidence: thin plants
are hard both to segment and to photograph clearly.

## The seventeen uncertain images

| | |
| --- | ---: |
| Unidentifiable photographs | 17/493 = **3.45%** |
| 95% Wilson CI | [2.16%, 5.45%] |
| Per 10,000 scraped | ~344 (216-545) |

This is a **ceiling on the species classifier, not a segmentation defect**:
about 3.5% of a research-grade CC-licensed *Drosera* scrape is too poor to
identify by eye, and no segmentation work will recover it. At 75% balanced
accuracy it is well inside the classifier's existing error.

### Loose cropping is not what they share

The reviewer suggested looseness was part of it (*"they are not cropping close
to the plant"*). The measurement does not support that:

| Group | n | Mean box fraction | vs passes | Mean confidence |
| --- | ---: | ---: | --- | ---: |
| Passes | 464 | 65.3% | — | 0.933 |
| Uncertain | 17 | 67.1% | +1.8 pts, permutation p = 0.78 | 0.911 |
| Failures | 12 | 74.2% | +8.9 pts, permutation p = 0.24 | 0.887 |

Uncertain images are cropped like the passes. Where box size does matter, it
helps: `deprecated/reports/species-crop-comparison.md` measured classifier
accuracy rising with box fraction (r = +0.108, t = +2.15; boxes over 90% of
the frame scored 0.802 against 0.697 for boxes under 50%). What the seventeen
share is photo quality (blur, clutter, occluding vegetation), which neither the
crop nor the segmenter controls.

## Two checks that came out clean

**Observer familiarity does not matter.** 170 of the 493 came from
photographers present in the labelled set. They were flagged rather than
removed, since a real scrape would include them too.

| | Failures | Rate |
| --- | ---: | ---: |
| Observer seen in labelled set | 4/170 | 2.35% |
| Novel observer | 8/323 | 2.48% |

The rates are the same, so the headline needs no adjustment.

**Confidence does not work as a filter** (the third time this has been
measured). Mean confidence on the failures is 0.887 against 0.933 on passes,
and the worst failures are confident: index 480 at **0.952**, index 191 at
0.943.

| Threshold | Images flagged | Failures caught | Precision |
| --- | ---: | ---: | ---: |
| < 0.85 | 2 | 1 of 12 | 0.50 |
| < 0.90 | 66 | 5 of 12 | 0.08 |

Catching under half the failures would mean reviewing 13% of the corpus and
being wrong 92% of the time.

## What follows

1. **Start the scrape.** The rate is cleared and the crop is justified.
2. **Expect about 2.4% unusable crops** (about 243 per 10,000) as input noise
   against correct labels, plus about 3.4% unidentifiable photos. Budget for
   both; no automated filter works.
3. **Hard negatives are worthwhile but no longer blocking.** They were gated on
   a rate above 15%. The better-aimed version targets linear and filiform
   plants, where four separate measurements agree the model is weakest.

## Caveats

- One reviewer, one pass, no second opinion. Eye-review counts are
  order-of-magnitude
  ([`deprecated/reports/scrape-probe-50.md`](../../deprecated/reports/scrape-probe-50.md)),
  though "is the plant in the box" is far less subjective than judging mask
  quality.
- The 464 images on neither list are treated as passes. This assumes the
  reviewer listed every case that gave them pause.
- Because the reviewer's `uncertain` criterion differs from the protocol's, how
  often a crop was too blurry to judge *containment* was not measured. That
  number is unknown, not zero.
- The rate is specific to this checkpoint at threshold 0.6. Retraining the
  segmenter invalidates it.
