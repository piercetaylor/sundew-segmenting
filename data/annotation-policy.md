# Sundew mask annotation policy

## Target

Create one binary semantic mask for **visible, living sundew plant tissue**. The
mask answers: “Which visible pixels belong to a sundew plant?” It does not separate
individual plants or identify species.

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

Annotation version: `v1.0`.

## Growth-form metadata

Each task carries a morphology prior for sampling and evaluation. It is not a
label class. Continue to create one binary mask containing all visible sundew
tissue, including mixed populations. The `dense_mat` value is only a likely
presentation for colony-forming pygmy taxa and should be checked against the
actual photograph.
