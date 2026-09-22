# Crop review, returned 2026-09-21

The human review output behind `reports/crop-review-result.md`, kept in the
repository because it is the one artifact here that cannot be recomputed.

| File | Contents |
| --- | --- |
| `failures.txt` | 12 sheet indices where the sundew is **not inside the crop** |
| `uncertain.txt` | 17 indices the reviewer **could not identify to species** from the photograph — blur, poor quality, occluding grass |
| `crop-review-manifest.jsonl` | index → image name, predicted box, box fraction, mean foreground confidence, for all 493 |

The two lists answer different questions and are not summed. See the report.

`failures.txt` was returned in two parts; the first was incomplete and is
superseded by this copy, which adds index 191. The `uncertain` criterion the
reviewer used is not the one `docs/crop-review-handoff.md` specified — theirs
is about species identifiability, the handoff asked about crop containment —
and the report treats it accordingly.

The images themselves are not in the repository. They live at
`PierceTaylor/sundew-crop-review/` on Hellbender with their per-photograph
licences and attribution in `metadata.jsonl` and `review/ATTRIBUTION.md`.

Recompute with `python scripts/crop_review_stats.py` and
`python scripts/crop_review_boxtest.py`.
