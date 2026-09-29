# Reviewed dataset freeze v0.3.0

The first reviewed training snapshot contains 191 accepted image-mask pairs
from 250 reviewed tasks. Forty-seven tasks were marked ambiguous and 12 were
rejected; neither group is included in model training.

## Frozen split

| Split | Accepted pairs | Use |
| --- | ---: | --- |
| Train | 144 | Parameter fitting |
| Validation | 19 | Checkpoint and threshold selection |
| Test | 28 | Locked until model selection is complete |

The accepted pairs cover 131 rosettes, 32 erect or branching plants, 17 linear
or forked plants, and 11 dense mats. Image licenses comprise 158 CC BY images
and 33 CC0 images; attribution and source URLs remain attached to every record.

The local transfer artifact is
`dist/sundew-segmenting-reviewed-v0.3.0.zip` (121.6 MB). Its SHA-256 digest is:

```text
58fe6deb04da92ef7f52dd70bcbcb46e728bce8df69849212783ff8e48c5d78e
```

The snapshot includes the accepted images and masks, complete review and mask
audit reports, the annotation policy, dataset card, per-pair checksums, and a
full file inventory. Rebuilding refuses to overwrite the existing directory,
which prevents accidental mutation of this version.
