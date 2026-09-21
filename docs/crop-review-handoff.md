# Laptop handoff: review 500 crops

Twenty-five minutes of eye review that decides whether the 10-20k species scrape
goes ahead. Background and the decision rule are in
`docs/crop-readiness-plan.md`; this file is only the procedure.

## What you are judging

Each tile is a **crop**, not a photograph — the region the segmentation model
selected, exactly as a species classifier would receive it. One binary question
per tile:

> **Is the sundew inside this crop?**

"Yes" means a classifier could plausibly identify the plant from this image. It
does not mean the crop is tight, well-centred or pretty. A loose crop with the
plant plus a lot of background is a **pass** — the classifier can cope with
that. Judge inclusion, not quality.

Mark it a **failure** when:

- the crop contains no sundew at all (landed on a hand, a tape measure, a label,
  bare soil, neighbouring vegetation);
- the sundew is cut off badly enough that you could not identify it;
- the tile is labelled `EMPTY PREDICTION` in yellow — the model found no
  foreground at all. These are automatic failures; you do not need to judge
  them, but confirm the label is not obviously wrong.

Do not agonise. A fast, consistent judgement across 500 tiles is worth far more
than a careful one across 100. If you genuinely cannot tell, count it as a pass
and note the index separately — systematic uncertainty is itself a finding.

## Getting the bundle

```bash
scp pmt5gt@hellbender.rnet.missouri.edu:/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-segmenting/dist/crop-review-v1.tar.gz* .
sha256sum -c crop-review-v1.tar.gz.sha256
tar -xzf crop-review-v1.tar.gz
```

If `sha256sum -c` complains about a file it cannot find, the checksum file has
Windows line endings — `tr -d '\r' < f.sha256 > f2 && mv f2 f.sha256`. This bit
the transfer in the other direction too.

Inside: `crop-review-01.jpg` through `crop-review-25.jpg` (20 crops per sheet,
**indices 1-493** running left-to-right, top-to-bottom), plus
`crop-review-manifest.jsonl` mapping every index to its source filename and
predicted box, `ATTRIBUTION.md` with per-image creator and licence, and a copy
of this file as `README.md`.

493 rather than 500: the contamination check found 7 images that were exact
duplicates (photo id, SHA-256 and observation id) of the frozen labelled set,
which the acquisition script could not catch because it only dedupes against its
own manifest. They are excluded and listed in
`sundew-crop-review/excluded-contaminated.jsonl` on the cluster. 493 still
clears the 457 needed for +/-2 points.

Expect many crops to be loose — the predicted box covers a median 69% of the
frame and more than 90% on about a fifth of them. **A loose crop still passes.**
You are judging whether the plant is inside, not whether the box is tight.

## Doing the review

Open the sheets in any image viewer. **Write down only the indices that fail.**
Most will pass, so recording failures rather than all 500 judgements turns this
into scanning plus a couple of dozen numbers.

Put them in a plain text file, one index per line or comma-separated:

```text
# failures.txt
17
48
122, 123
390
```

Add a second file if you had genuine uncertainty on any:

```text
# uncertain.txt
204
311
```

That is the whole deliverable. No annotation, no Label Studio, no masks.

## Sending it back

The result is a few dozen numbers, so the simplest path is to paste the contents
of `failures.txt` straight into the next cluster session. If you would rather
send files:

```bash
scp failures.txt uncertain.txt pmt5gt@hellbender.rnet.missouri.edu:/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-crop-review/
```

## What happens to the number

The failure rate over 500 tiles has a 95% interval of about +/- 2 points, which
is what the current estimate lacks — it rests on one failing image in 17, giving
an interval of [1.0%, 27.0%]. The decision rule is fixed in advance so the data
cannot be argued with after the fact:

| Measured rate | What follows |
| --- | --- |
| below 5% | start the scrape; the crop noise is affordable |
| 5-15% | judgement call; the step 2 classifier comparison likely decides it |
| above 15% | annotate ~50 hard negatives before scraping |

Note that a failure here corrupts the classifier's *input*, never its *label* —
the species comes from iNaturalist and stays correct. That is why a few percent
is tolerable at all.

## While you review

Step 2 is running on the cluster and does not need you: 2,000 images across the
ten best-represented species, then one classifier trained twice, on full frames
and on segmentation crops. If full frames match crops, the failure rate you are
measuring stops mattering for the species project entirely. That result and
yours arrive independently.
