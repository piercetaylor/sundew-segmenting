# Which photograph licences this project accepts, and why

Every image here is someone else's photograph under its own licence. The
software licence in `LICENSE` covers the code and does not relicense them.
Above all of that sits iNaturalist's own terms, which prohibit using iNaturalist
data for commercial AI or machine-learning training regardless of what any
individual photo permits. This is personal, noncommercial research.

This records a decision taken on 2026-09-21, when the corpus grew from 10
species to 110 and the licence question stopped being theoretical.

## The policy

| Licence | Accepted | |
| --- | :---: | --- |
| CC0 | yes | |
| CC BY | yes | |
| CC BY-NC | **yes**, as of this decision | |
| CC BY-SA | no | ShareAlike; see below |
| CC BY-NC-SA | no | ShareAlike |
| CC BY-ND | no | NoDerivatives |
| CC BY-NC-ND | no | NoDerivatives |
| all rights reserved | no | never requested |

Enforced in two places that must agree: the `photo_license` parameter on the
API query, and `allowed_licenses` in `extract_candidates`. The second defaults
to the narrow CC0/CC-BY set, so widening the query alone yields nothing rather
than acquiring photographs by accident.

## What each choice cost, measured

Research-grade, wild, photographed *Drosera* observations, queried 2026-09-21:

| Licence set | Observations | Marginal |
| --- | ---: | ---: |
| CC0 only | 3,785 | |
| CC0 + BY | 21,386 | |
| + SA | 24,300 | +2,914 |
| **+ NC (adopted)** | **124,818** | **+103,432** |
| + NC + SA | 130,407 | +5,589 over NC |
| + ND as well | 133,263 | +2,856 |

The decisive fact is that these are not one choice. **NonCommercial is
essentially the entire benefit; ShareAlike is 4% of the expanded pool.** Treating
"NC/SA" as a package would have traded the hard problem for almost none of the
data.

In classes rather than images, at a 50-observation floor and a 300 cap:

| Pool | Classes | Images |
| --- | ---: | ---: |
| CC0 + BY | 48 | ~10,500 |
| CC0 + BY + NC | **110** | **~21,600** |

## NoDerivatives — declined on a plain reading

The pipeline produces segmentation masks and cropped derivatives and publishes
overlay figures from them. Those are derivative works. Under ND they could
arguably be trained on but never shown, which removes the ability to illustrate
or check the work. 2,856 observations, and not worth the ambiguity.

## ShareAlike — declined on a judgement call

ShareAlike requires adaptations to carry the same licence. Three consequences,
in increasing order of how much they matter:

1. Each mask or crop is an adaptation of one photograph and inherits BY-SA or
   BY-NC-SA. Manageable by segregating output directories.
2. BY-SA 4.0 and BY-NC-SA 4.0 are mutually incompatible, so no single derivative
   work can combine both. Composite figures would have to be split by licence.
3. **Whether trained model weights are an adaptation of the training images is
   unsettled.** A conservative reading says one SA photograph in the training
   set could oblige the weights to be released under SA.

The third is why this is declined. Contamination of that kind does not scale
with proportion: 4% of the images could implicate 100% of the weights. Accepting
an open legal question over the whole deliverable to gain 4.3% more data is a
bad trade, and the 4.3% is the part of this that is actually measured.

None of this is legal advice, and the weights question in particular is
genuinely unresolved — which is itself the argument for staying out of it.

## NonCommercial — accepted

**What it forbids:** any commercial use of the corpus, the derived crops and
masks, or, on a conservative reading, models trained on them. No paid product,
no ad-supported deployment, no sale of the dataset, no use inside an employer's
product.

**What it forbids that was permitted before: nothing, in practice.** iNaturalist
already prohibits commercial ML training on its data whatever the photo licence
says, so this corpus was never commercially usable. NC moves an existing
restriction out of a terms-of-service document and into the licences on the
photographs, where it is explicit and durable.

**What it genuinely forecloses:** if iNaturalist's terms ever loosened, a
CC0/CC-BY corpus could become commercially usable and this one could not.

**Why that is recoverable rather than a one-way door:** every manifest row
carries `license_code`, so the corpus can be filtered back to CC0/CC-BY at any
time and a clean-licence model retrained from the subset. The cost of changing
this decision later is one training run, not one re-scrape.

**What does not change:** attribution. BY-NC requires it exactly as BY does, and
the existing attribution tooling already covers it.

## Obligations this creates

- Attribution for every CC BY and CC BY-NC photograph, carried in
  `metadata.jsonl` (`attribution`, `creator`, `license_code`, `source_page`) and
  rendered into an `ATTRIBUTION.md` beside any published artifact.
- Licence tier recorded per image so the CC0/CC-BY subset stays separable.
- No commercial use of anything derived from the corpus.
- Exact coordinates continue to be neither requested nor stored.
