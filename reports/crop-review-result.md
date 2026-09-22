# Step 1 result: the crop failure rate is 2.4%, and the scrape is cleared

493 uncurated scraped images, crops produced by the current best checkpoint
(`models/field-compare/field/seed-101`, threshold 0.6) exactly as a
crop-then-classify pipeline would consume them, reviewed by eye.

The completed review returned **two** lists, and they answer different
questions. Keeping them apart is the main thing this report does.

| List | Question it answers | n |
| --- | --- | ---: |
| `failures.txt` | is the sundew inside this crop? | 12 |
| `uncertain.txt` | could the reviewer identify the species from this photograph? | 17 |

Only the first is a segmentation failure. The second is a property of the
photograph — blur, poor quality, obscuring grass — that a better segmenter
cannot fix, and it is reported separately below.

| | |
| --- | ---: |
| Reviewed | 493 |
| Crop failures | **12** |
| **Rate** | **2.43%** |
| 95% Wilson CI | **[1.40%, 4.21%]** |
| Per 10,000 scraped images | ~243 bad crops (139-420) |

The prior estimate, from one failing image in the 17-image `field-eval` set, was
5.9% with a 95% interval of [1.0%, 27.0%]. The interval is now about **eight
times tighter** and sits **entirely below 5%**.

> This supersedes the first pass of this report, which scored an incomplete
> 11-line `failures.txt` returned before the review had finished and stated that
> no `uncertain.txt` existed. Index 191 is the added failure. The headline moved
> 2.23% → 2.43%; nothing else about the conclusion moved. Reproduce with
> `python scripts/crop_review_stats.py`.

## The decision

`docs/crop-readiness-plan.md` fixed the rule before the data was seen:

| Measured rate | Action |
| --- | --- |
| **below 5%** | **start the scrape; the crop noise is affordable** |
| 5-15% | judgement call against classifier tolerance |
| above 15% | fix hard negatives first |

**2.43% clears it, and the whole confidence interval clears it.** The upper
bound, 4.21%, is still below the threshold, so the conclusion does not depend on
the point estimate being exactly right.

It survives every defensible treatment of the seventeen uncertain images except
the one that is wrong on its face:

| Treatment | Rate | 95% CI | Clears 5%? |
| --- | ---: | --- | --- |
| Uncertain are not crop failures (**used**) | 12/493 = 2.43% | [1.40%, 4.21%] | yes, interval and all |
| Uncertain excluded from the denominator | 12/476 = 2.52% | [1.45%, 4.35%] | yes, interval and all |
| Uncertain counted as crop failures | 29/493 = 5.88% | [4.13%, 8.32%] | no |

The third row is the reason the distinction matters, and it is not the right
reading. The review protocol in `docs/crop-review-handoff.md` defined uncertain
as *"too blurry to tell whether a sundew is in the crop"* — a containment
judgement. The reviewer used a different and more useful criterion: *"those I
would not be able to ID confidently from a photo... because the photo was
blurry, poor quality, or obscured by grass."* That is a **species
identifiability** judgement. A crop can contain the plant perfectly and still be
unidentifiable, and 16 of the 17 have confidence and box geometry
indistinguishable from the passes. Folding them into the crop failure rate would
charge the segmentation model for the photographer's focus.

This also has to be read alongside `reports/species-crop-comparison.md`, which
established that the crop is worth +0.0246 balanced accuracy over full frames.
A failure rate of 2% is the price of an effect worth 2.5 points, and it corrupts
the classifier's *input* while leaving its *label* — which comes from
iNaturalist — correct.

Review cost: about 25 minutes, against roughly 100 hours to annotate 500 masks.
The cheap measurement was the right one, and annotating masks to answer this
would have been a mistake.

## The twelve failures

Growth forms are taken from `src/sundew_segmentation/growth_forms.py` rather
than assigned by hand — the first pass of this table hand-labelled two rows
wrong (`D. stricticaulis` is not in the table at all, and `D. erythrorhiza` is a
rosette), though the stratification counts below were always computed from the
code and were unaffected.

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

Four of the twelve produced a box covering 92-100% of the frame — the model
selected essentially everything and located nothing. Two more (103, 191)
produced a box under 30%, the opposite mode: it locked onto a fragment. The
same two opposite failure modes `reports/field-probe-evaluation.md` found on
`field-eval` are still here, in the same proportion, and they still cancel in
any mean.

## Stratification

| Growth form | Failures | Rate | 95% CI |
| --- | ---: | ---: | --- |
| linear or forked | 4/78 | **5.13%** | [2.0%, 12.5%] |
| unknown | 3/110 | 2.73% | [0.9%, 7.7%] |
| erect or branching | 2/80 | 2.50% | [0.7%, 8.7%] |
| rosette | 3/214 | 1.40% | [0.5%, 4.0%] |
| dense mat | 0/11 | 0.00% | [0.0%, 25.9%] |

The added failure was linear/forked, which moves that row from 3.85% to 5.13%
and widens the gap against rosettes from 2.7x to **3.7x**. The intervals still
overlap, so this is **not established by this data alone** — but it is the
fourth independent hint in the same direction. `baseline-seed-sweep.md` found
`linear_or_forked` the weakest growth form for both architectures,
`scrape-pipeline-readiness.md` found both unseen-species failures were
linear/forked, and it remains the least represented form in the training set at
17 of 191 pairs. Four weak signals pointing one way is worth acting on even
though no single one is conclusive.

The same row is also the worst on identifiability (9 of 78 are a failure or
uncertain, 11.5%), which is a coincidence of morphology rather than evidence:
filiform plants are thin, and thin plants are both hard to segment and hard to
photograph legibly.

## The seventeen uncertain images

| | |
| --- | ---: |
| Unidentifiable photographs | 17/493 = **3.45%** |
| 95% Wilson CI | [2.16%, 5.45%] |
| Per 10,000 scraped | ~344 (216-545) |

This is a **ceiling on the species classifier, not a segmentation defect.** It
says roughly 3.5% of a research-grade CC-licensed *Drosera* scrape is too poor
to identify by eye, and that no amount of segmentation work will recover those
images. Whether it is worth acting on depends on the classifier: at 75%
balanced accuracy, 3.5% unidentifiable input is well inside the existing error.

### Loose cropping is not what these images have in common

The reviewer's note offered looseness as part of the rationale — *"they are not
cropping close to the plant."* Measured, it is not there:

| Group | n | Mean box fraction | vs passes | Mean confidence |
| --- | ---: | ---: | --- | ---: |
| Passes | 464 | 65.3% | — | 0.933 |
| Uncertain | 17 | 67.1% | +1.8 pts, permutation p = 0.78 | 0.911 |
| Failures | 12 | 74.2% | +8.9 pts, permutation p = 0.24 | 0.887 |

The uncertain images are cropped essentially like the passes. And where box
size does move with an outcome, it moves the *helpful* way:
`reports/species-crop-comparison.md` measured classifier accuracy rising with
box fraction (r = +0.108, t = +2.15; boxes over 90% of frame scored 0.802
against 0.697 for boxes under 50%). Looser crops are not a defect in this
pipeline. What these seventeen share is photographic quality — blur, clutter,
occluding vegetation — which is exactly what the reviewer described, and which
neither the crop nor the segmenter controls.

Recorded because the intuition was reasonable and the measurement rejects it,
in the same way `field-probe-evaluation.md` records the rejected pigment
hypothesis.

## Two checks that came out clean

**Observer familiarity does not matter.** 170 of the 493 came from photographers
present in the labelled set, which was flagged rather than removed on the
grounds that a real scrape would hit them too.

| | Failures | Rate |
| --- | ---: | ---: |
| Observer seen in labelled set | 4/170 | 2.35% |
| Novel observer | 8/323 | 2.48% |

The rates are indistinguishable, so leaving those images in did not bias the
estimate, and the headline rate needs no adjustment.

**Confidence has now failed as a filter three times.** Mean confidence on the
twelve failures is 0.887 against 0.933 on the passes, and the worst offenders
are confident: index 480 failed at **0.952**, index 191 at 0.943.

| Threshold | Images flagged | Failures caught | Precision |
| --- | ---: | ---: | ---: |
| < 0.85 | 2 | 1 of 12 | 0.50 |
| < 0.90 | 66 | 5 of 12 | 0.08 |

Catching under half the failures means reviewing 13% of the corpus and being
wrong 92% of the time. This should not be attempted again.

## What follows

1. **Start the scrape.** The rate is cleared and the crop is justified.
2. **Expect ~2.4% unusable crops** — about 243 per 10,000 — as input noise
   against correct labels, plus ~3.4% photographs that are unidentifiable
   regardless. Budget for both rather than trying to filter them, since no
   automated filter works.
3. **Hard negatives remain worthwhile but are no longer blocking.** Step 3 of
   the plan was gated on a rate above 15%. At 2.4% it is optional, and the
   better-aimed version of it targets *linear and filiform* morphology, where
   four separate measurements now agree the model is weakest.

## Caveats

- One reviewer, one pass, no second opinion. `scrape-probe-50.md` already noted
  that eye-review counts are order-of-magnitude rather than precise, though the
  binary containment question is far less subjective than judging mask quality.
- The 464 images on neither list are treated as unambiguous passes. With both
  lists now returned that is the intended reading of the review, but it does
  assume the reviewer listed every case that gave them pause.
- The reviewer's `uncertain` criterion differs from the one the handoff
  specified. The criterion they used is the more informative of the two, but it
  means no measurement exists of how often a crop was too blurry to judge
  *containment* — that number is unknown rather than zero.
- The rate is specific to this checkpoint at threshold 0.6. Retraining the
  segmentation model invalidates it.
