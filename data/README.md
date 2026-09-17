# Dataset layout

The downloaded and derived images are intentionally excluded from Git.

```text
data/
├── raw/inaturalist/
│   ├── images/              # original licensed photographs
│   ├── metadata.jsonl       # image-level provenance and attribution
│   ├── rejected.jsonl       # failed, duplicate, or invalid downloads
│   └── acquisition.json     # query and run summary
├── curated/
│   ├── images/{train,validation,test}/
│   ├── metadata.jsonl       # provenance, grouped split, derivative details
│   ├── curation.jsonl       # decision for all 500 reviewed candidates
│   └── summary.json
└── reports/                 # contact sheets and audit output
```

Only photographs with an image-level `cc0` or `cc-by` license are accepted. The
train, validation, and test split is grouped by iNaturalist observer to reduce
photographer leakage. The visual screen is an initial suitability pass; confirm it
again while correcting segmentation masks.

Precise coordinates are not acquired or published. Do not add images manually
without equivalent provenance fields in the manifest.
