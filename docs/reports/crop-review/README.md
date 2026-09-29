# Crop review, returned 2026-09-21

**In short.** These are the human review lists behind
[crop-review-result.md](../crop-review-result.md): 12 crops that miss the
sundew and 17 photos too poor to identify to species, out of 493. They are
kept in the repository because they are the one result here that cannot be
recomputed.

| File | Contents |
| --- | --- |
| `failures.txt` | 12 sheet indices where the sundew is **not inside the crop** |
| `uncertain.txt` | 17 indices the reviewer **could not identify to species** from the photograph (blur, poor quality, occluding grass) |
| `crop-review-manifest.jsonl` | index → image name, predicted box, box fraction, mean foreground confidence, for all 493 |

The two lists answer different questions and are not added together.

`failures.txt` was returned in two parts. The first was incomplete; this copy
replaces it and adds index 191. The reviewer's `uncertain` criterion is not the
one in the protocol,
[`deprecated/docs/crop-review-handoff.md`](../../../deprecated/docs/crop-review-handoff.md):
theirs is about identifying the species, the protocol's about whether the plant
is in the crop. The report treats it accordingly.

The images are not in the repository. They live at
`PierceTaylor/sundew-crop-review/` on Hellbender, with per-photo licences and
attribution in `metadata.jsonl` and `review/ATTRIBUTION.md`.

Recompute with `python scripts/crop_review_stats.py` and
`python deprecated/scripts/crop_review_boxtest.py`.
