# Step 1 result: the crop failure rate is 2.2%, and the scrape is cleared

493 uncurated scraped images, crops produced by the current best checkpoint
(`models/field-compare/field/seed-101`, threshold 0.6) exactly as a
crop-then-classify pipeline would consume them, reviewed by eye against one
binary question: is the sundew inside this crop?

| | |
| --- | ---: |
| Reviewed | 493 |
| Failures | **11** |
| **Rate** | **2.23%** |
| 95% Wilson CI | **[1.25%, 3.95%]** |
| Per 10,000 scraped images | ~223 bad crops (125-395) |

The prior estimate, from one failing image in the 17-image `field-eval` set, was
5.9% with a 95% interval of [1.0%, 27.0%]. The interval is now about **ten times
tighter** and sits **entirely below 5%**.

## The decision

`docs/crop-readiness-plan.md` fixed the rule before the data was seen:

| Measured rate | Action |
| --- | --- |
| **below 5%** | **start the scrape; the crop noise is affordable** |
| 5-15% | judgement call against classifier tolerance |
| above 15% | fix hard negatives first |

**2.23% clears it, and the whole confidence interval clears it.** The upper
bound, 3.95%, is still below the threshold, so the conclusion does not depend on
the point estimate being exactly right.

This also has to be read alongside `reports/species-crop-comparison.md`, which
established that the crop is worth +0.0246 balanced accuracy over full frames.
A failure rate of 2% is the price of an effect worth 2.5 points, and it corrupts
the classifier's *input* while leaving its *label* — which comes from
iNaturalist — correct.

Review cost: about 25 minutes, against roughly 100 hours to annotate 500 masks.
The cheap measurement was the right one, and annotating masks to answer this
would have been a mistake.

## The eleven failures

| Index | Species | Growth form | Box | Confidence |
| ---: | --- | --- | ---: | ---: |
| 58 | *D. stricticaulis* | linear/forked | 68% | 0.868 |
| 103 | *D. humilis* | unknown | 26% | 0.679 |
| 339 | *D. erythrorhiza* | unknown | 100% | 0.865 |
| 354 | *D. filiformis* | linear/forked | 92% | 0.929 |
| 375 | *D. finlaysonii* | unknown | 69% | 0.900 |
| 435 | *D. linearis* | linear/forked | 82% | 0.870 |
| 442 | *D. filiformis* | linear/forked | 73% | 0.924 |
| 459 | *D. auriculata* | erect/branching | 95% | 0.928 |
| 462 | *D. anglica* | rosette | 100% | 0.934 |
| 480 | *D. intermedia* | erect/branching | 58% | 0.952 |
| 481 | *D. anglica* | rosette | 100% | 0.856 |

Four of the eleven produced a box covering 92-100% of the frame — the model
selected essentially everything and located nothing.

## Stratification

| Growth form | Failures | Rate | 95% CI |
| --- | ---: | ---: | --- |
| linear or forked | 3/78 | 3.85% | [1.3%, 10.7%] |
| erect or branching | 2/80 | 2.50% | [0.7%, 8.7%] |
| unknown | 3/110 | 2.73% | [0.9%, 7.7%] |
| rosette | 3/214 | 1.40% | [0.5%, 4.0%] |
| dense mat | 0/11 | 0.00% | [0.0%, 25.9%] |

Linear and forked morphology fails about 2.7x as often as rosettes. The
intervals overlap heavily, so this is **not established by this data alone** —
but it is the fourth independent hint in the same direction. `baseline-seed-
sweep.md` found `linear_or_forked` the weakest growth form for both
architectures, `scrape-pipeline-readiness.md` found both unseen-species failures
were linear/forked, and it remains the least represented form in the training
set at 17 of 191 pairs. Four weak signals pointing one way is worth acting on
even though no single one is conclusive.

## Two checks that came out clean

**Observer familiarity does not matter.** 170 of the 493 came from photographers
present in the labelled set, which was flagged rather than removed on the
grounds that a real scrape would hit them too.

| | Failures | Rate |
| --- | ---: | ---: |
| Observer seen in labelled set | 4/170 | 2.35% |
| Novel observer | 7/323 | 2.17% |

The rates are indistinguishable, so leaving those images in did not bias the
estimate, and the headline rate needs no adjustment.

**Confidence has now failed as a filter three times.** Mean confidence on the
eleven failures is 0.882 against 0.932 on the passes, and the worst offenders
are confident: index 480 failed at **0.952**.

| Threshold | Images flagged | Failures caught | Precision |
| --- | ---: | ---: | ---: |
| < 0.85 | 2 | 1 of 11 | 0.50 |
| < 0.90 | 66 | 5 of 11 | 0.08 |

Catching half the failures means reviewing 13% of the corpus and being wrong
92% of the time. This should not be attempted again.

## What follows

1. **Start the scrape.** The rate is cleared and the crop is justified.
2. **Expect ~2% unusable crops** — about 223 per 10,000 — as input noise against
   correct labels. Budget for it rather than trying to filter it, since no
   automated filter works.
3. **Hard negatives remain worthwhile but are no longer blocking.** Step 3 of
   the plan was gated on a rate above 15%. At 2.2% it is optional, and the
   better-aimed version of it targets *linear and filiform* morphology, where
   four separate measurements now agree the model is weakest.

## Caveats

- One reviewer, one pass, no second opinion. `scrape-probe-50.md` already noted
  that eye-review counts are order-of-magnitude rather than precise, though the
  binary question here is far less subjective than judging mask quality.
- No `uncertain.txt` was returned, so this treats the 482 unlisted images as
  unambiguous passes.
- The rate is specific to this checkpoint at threshold 0.6. Retraining the
  segmentation model invalidates it.
