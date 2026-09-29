# Which photograph licences this project accepts, and why

**In short.** The species data accepts CC0, CC BY and CC BY-NC photos, and
declines ShareAlike and NoDerivatives. Adding NonCommercial took the pool from
48 to 110 species; ShareAlike would have added about 4% more photos at the risk
of binding the model weights. Decided 2026-09-21, when the corpus grew from 10
species to 110. The 191-mask segmentation set is CC0 and CC BY only.

Every image here is someone else's photograph under its own licence. The
code's Apache-2.0 licence (`LICENSE`, `NOTICE`) covers the code only and does not relicense them.
iNaturalist's own terms sit above all of that: they prohibit using iNaturalist
data for commercial AI or machine-learning training, whatever an individual
photo permits. This is personal, noncommercial research.

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

It is enforced in two places that must agree: the `photo_license` parameter
on the API query, and `allowed_licenses` in `extract_candidates`. The second
defaults to CC0/CC BY, so widening the query alone acquires nothing by
accident.

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

These are two separate choices. **NonCommercial is nearly all of the benefit;
ShareAlike is 4% of the expanded pool.** Taking "NC/SA" as a package would have
accepted the hard problem for almost no extra data.

In classes rather than images, at a 50-observation floor and a 300 cap:

| Pool | Classes | Images |
| --- | ---: | ---: |
| CC0 + BY | 48 | ~10,500 |
| CC0 + BY + NC | **110** | **~21,600** |

## NoDerivatives: declined on a plain reading

The pipeline makes masks and crops and publishes overlay figures from them.
Those are derivative works. Under ND they could arguably be trained on but
never shown, so the work could not be illustrated or checked. 2,856
observations are not worth that.

## ShareAlike: declined on a judgement call

ShareAlike requires adaptations to carry the same licence. Three consequences,
from least to most important:

1. Each mask or crop is an adaptation of one photograph and inherits BY-SA or
   BY-NC-SA. Manageable by segregating output directories.
2. BY-SA 4.0 and BY-NC-SA 4.0 are mutually incompatible, so no single derivative
   work can combine both. Composite figures would have to be split by licence.
3. **Whether trained model weights are an adaptation of the training images is
   unsettled.** A conservative reading says one SA photograph in the training
   set could oblige the weights to be released under SA.

The third is why it is declined. The risk does not scale with the share: 4%
of the images could bind 100% of the weights. An open legal question over the
whole deliverable is a bad trade for 4.3% more data.

None of this is legal advice. The weights question is unresolved, which is
itself the reason to stay out of it.

## NonCommercial: accepted

**What it forbids:** any commercial use of the corpus, the derived crops and
masks, or, on a conservative reading, models trained on them. No paid product,
no ad-supported deployment, no sale of the dataset, no use inside an employer's
product.

**What it forbids that was allowed before: nothing, in practice.** iNaturalist
already prohibits commercial ML training on its data, so this corpus was never
commercially usable. NC moves an existing restriction from the terms of service
into the photo licences, where it is explicit.

**What it genuinely forecloses:** if iNaturalist's terms ever loosened, a
CC0/CC-BY corpus could become commercially usable and this one could not.

**Why that is reversible:** every manifest row carries `license_code`, so the
corpus can be filtered back to CC0/CC BY at any time and a model retrained on
the subset. Changing this decision later costs one training run, not a new
scrape.

**What does not change:** attribution. BY-NC requires it exactly as BY does,
and the existing tooling covers it.

## Obligations this creates

- Attribution for every CC BY and CC BY-NC photograph, carried in
  `metadata.jsonl` (`attribution`, `creator`, `license_code`, `source_page`) and
  rendered into an `ATTRIBUTION.md` beside any published artifact.
- Licence tier recorded per image so the CC0/CC-BY subset stays separable.
- No commercial use of anything derived from the corpus.
- Exact coordinates continue to be neither requested nor stored.

## Trained weights

- **Species model, published** in [`release/species-v1.0.0/`](../release/species-v1.0.0/) (2026-09-29):
  CC BY-NC 4.0. The DINOv2 weights it starts from are Apache-2.0 (Meta, via
  timm), so nothing upstream blocks release; the training photos include
  CC BY-NC, and iNaturalist's terms rule out commercial AI training. The
  folder has a model card crediting DINOv2 and timm, the licence text, and
  `ATTRIBUTION.md` for all 23,244 training photos.
- **Segmentation model**: SegFormer-B0's encoder in segmentation-models-pytorch
  (`mit_b0`) comes from NVIDIA's SegFormer, under the NVIDIA Source Code
  Licence for research and noncommercial use only. Any SegFormer weights must
  carry that licence. The U-Net/ResNet-34 weights have no such restriction.

Not legal advice; check the NVIDIA licence again before sharing segmentation
weights.
