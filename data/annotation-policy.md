# Sundew mask annotation policy

**In short.** One binary mask per image, covering every visible pixel of living
sundew tissue, flowers and stalks included. Nothing inferred, nothing that is
not sundew. Mark unclear images `ambiguous` and unusable ones `reject`; only
masks a person has checked and accepted are training labels. Annotation
version `v1.0`.

## Target

The mask answers one question: “Which visible pixels belong to a sundew
plant?” It does not separate individual plants or identify species.

## Include

- living leaf blades, petioles, and visible tentacles;
- visible mucilage droplets attached to tentacles when their boundary is clear;
- sundew flower stalks, buds, open flowers, and seed capsules when they are visibly
  connected to a labeled plant;
- all visible sundews when several plants occur in one image;
- partly occluded plants up to the occluder boundary;
- plant tissue cut by the image border.

## Exclude

- moss, grasses, litter, soil, water, rocks, pots, labels, hands, and insects;
- shadows, reflections, glare, and unattached droplets;
- fully occluded portions inferred from context;
- detached or clearly dead brown tissue when it cannot be connected confidently to
  a living sundew;
- unrelated flowers or carnivorous plants.

## Difficult boundaries

Zoom in and preserve the spaces between tentacles when the resolution supports it.
At low resolution, trace the smallest stable visible plant region rather than
inventing hair-level detail. Mark `ambiguous` when sundew tissue cannot be separated
confidently from overlapping vegetation or when species identity is questionable.
Mark `reject` if less than a useful plant region remains after inspection.

## Quality control

1. Check every draft mask at full-image view and at high zoom.
2. Confirm that image and mask dimensions match and the mask is binary.
3. Double-label at least 25 images (10% of the core set) before either annotator sees
   the other's mask.
4. Compute inter-annotator Dice and IoU, discuss cases below 0.85 Dice, and record the
   adjudicated mask.
5. Keep model-generated proposals labeled as proposals until a person corrects and
   accepts them.

## Growth-form metadata

Each task carries a morphology prior for sampling and evaluation. It is not a
label class: still make one binary mask with all visible sundew tissue,
including mixed populations. `dense_mat` only means a colony-forming pygmy
taxon is likely; check it against the actual photo.
