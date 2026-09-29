# Reviewed dataset freeze v0.3.0

**In short.** v0.3.0 is the frozen segmentation training set: 191 accepted
image-mask pairs out of 250 reviewed tasks, split 144 / 19 / 28 by observer.
The 47 ambiguous and 12 rejected tasks are left out. It is identified by the
SHA-256 digest below.

## Frozen split

| Split | Accepted pairs | Use |
| --- | ---: | --- |
| Train | 144 | Parameter fitting |
| Validation | 19 | Checkpoint and threshold selection |
| Test | 28 | Locked until model selection is complete |

The pairs cover 131 rosettes, 32 erect or branching plants, 17 linear or
forked plants, and 11 dense mats. Licences: 158 CC BY and 33 CC0 images, with
attribution and source URLs on every record.

## Artifact

`dist/sundew-segmenting-reviewed-v0.3.0.zip` (121.6 MB, local only). SHA-256:

```text
58fe6deb04da92ef7f52dd70bcbcb46e728bce8df69849212783ff8e48c5d78e
```

It contains the accepted images and masks, the full review and mask-audit
reports, the annotation policy, the dataset card, per-pair checksums, and a
complete file inventory. The build script refuses to overwrite an existing
snapshot directory, so this version cannot be changed by accident. See
[`docs/dataset-release.md`](../dataset-release.md) for how it was built.
