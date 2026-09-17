# Free online sundew image sources

Research date: 2026-09-16

## Recommendation

Use **iNaturalist as the primary source**, but admit only `CC0` and `CC BY` photographs into the publishable dataset. A live iNaturalist API query found **192,504 observations of _Drosera_ with photos**. Of those, **23,753 observations** had a CC0 or CC BY photo; **21,083** remained after requiring Research Grade and excluding captive/cultivated observations. These are observation counts rather than unique-photo counts, and an observation may contain several photographs. The [reproducible filtered query](https://api.inaturalist.org/v1/observations?taxon_id=51935&photos=true&quality_grade=research&captive=false&photo_license=cc0%2Ccc-by&per_page=1) returns the current count.

Supplement this pool with **Wikimedia Commons** photographs. Commons is smaller but easier to redistribute because its policy excludes files whose only license prohibits commercial use or derivatives. Use **Smithsonian Open Access** herbarium sheets only for an explicitly separate domain experiment; they do not resemble living sundews photographed from above.

Do not scrape arbitrary Google Images, Pinterest, nursery pages, social media, or Kaggle mirrors. No Kaggle sundew collection with a clearly traceable original source and image-by-image reuse license was identified. A downloadable file is not necessarily licensed for dataset redistribution.

The project can start with 300-500 screened candidates and 150-300 completed masks. Online biodiversity images are suitable for a small segmentation portfolio, but they are mostly close-up or oblique photographs. They cannot support a claim that the model has been validated on UAV or remote-sensing imagery.

## Source comparison

| Source | Availability and retrieval | Image and metadata quality | Redistribution status | Recommendation |
| --- | --- | --- | --- | --- |
| [iNaturalist](https://www.inaturalist.org/) | 192,504 photo observations for genus taxon `51935` in a live query; 21,083 Research Grade, non-captive observations with CC0/CC BY photos. Retrieve a curated sample through the REST API or query the monthly open-data snapshot. | Hosted originals are at most 2,048 px on the long edge. API records include photo ID, license, dimensions, attribution, observation ID, taxon, date, observer, quality grade, and public location/geoprivacy fields. | Yes for selected CC0/CC BY photos when their terms are followed. Other photos have different licenses or all rights reserved. iNaturalist additionally prohibits use of its data for **commercial** AI/ML training. | **Primary source for this personal, noncommercial project.** |
| [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Drosera) | The top-level _Drosera_ category currently contains 101 files and 163 subcategories; species and subject subcategories add many more. Traverse the category tree through the MediaWiki API. | Mixed resolutions, from small web images to originals above 6,000 px. `imageinfo` exposes dimensions, original URL, author/credit, source, and machine-readable license metadata. The category also contains maps, illustrations, herbarium sheets, PDFs, and audio that must be rejected. | Commons requires files to allow commercial reuse and derivatives. Individual files may be public domain, CC BY, CC BY-SA, or another free license; comply with each file page. | **Best supplement.** Prefer public-domain/CC0 and CC BY files; exclude or isolate ShareAlike files to keep licensing simple. |
| [GBIF](https://www.gbif.org/) | A live query found 202,153 _Drosera_ occurrence records with still images using taxon key `3190721`. Search through the occurrence API or request a Darwin Core Archive download. | `multimedia.txt` and API `media[]` records can expose the publisher URL, creator, rights holder, media license, source page, and date. GBIF's cache is limited to 1,200×1,200; the publisher URL may offer higher resolution. | The occurrence dataset's license does **not** necessarily govern its images. GBIF explicitly says media may have more restrictive terms, so validate every media-level license. | Useful for discovery, but heavily overlaps iNaturalist and other publishers. Deduplicate by publisher URL/photo ID. |
| [Smithsonian Open Access](https://www.si.edu/openaccess) | Search for _Drosera_ in the National Museum of Natural History botany collection, the public API, or the weekly JSON export. The Smithsonian does not publish a convenient genus-level image count without running the query. | High-resolution JPG and sometimes TIFF/IIIF assets with specimen, collection, place, collector, catalogue, and persistent identifier metadata. The U.S. National Herbarium reports 3.8 million digitized sheets overall. | Assets explicitly carrying the CC0 mark can be copied, modified, and redistributed for any purpose. Restricted assets are not interchangeable with the CC0 collection. | Optional domain-shift or herbarium experiment; poor match for live-plant segmentation. |
| [BOLD Systems](https://boldsystems.org/data/taxonomy/) | The taxonomy browser and API can locate barcode specimen records and associated images for _Drosera_. No reliable open count of usable _Drosera_ photographs was exposed. | BOLD accepts source images up to 20 MP but displays a greatly reduced thumbnail; the archived high-resolution image is not distributed without the submitter's explicit consent. Records include specimen/barcode metadata and per-image attribution/license fields. | BOLD owns neither the images nor their licenses. Uploaders may select licenses ranging from CC0-like “No Rights Reserved” through CC licenses to ordinary copyright, and BOLD says the owner may change the image license. | Poor acquisition source. Use only a clearly licensed original supplied at adequate resolution, after recording its current terms. |

## iNaturalist acquisition details

The genus taxon ID is `51935`. For a small, human-screened collection, page through this query at no more than the recommended request rate:

```text
https://api.inaturalist.org/v1/observations
  ?taxon_id=51935
  &photos=true
  &quality_grade=research
  &captive=false
  &photo_license=cc0,cc-by
  &per_page=200
```

The API's photo objects include `id`, `license_code`, `original_dimensions`, `url`, and `attribution`; observation objects provide the grouping and biological metadata needed to prevent leakage. iNaturalist asks clients to stay near one request per second and roughly 10,000 requests per day. It also states that downloading over 5 GB of media per hour or 24 GB per day can lead to a permanent block, and that the API is for application support rather than bulk scraping ([developer guidance](https://www.inaturalist.org/pages/developers), [API practices](https://www.inaturalist.org/pages/api%2Brecommended%2Bpractices)).

For a larger acquisition, use the official [iNaturalist Licensed Observation Images open-data snapshot](https://registry.opendata.aws/inaturalist-open-data/). Its metadata are updated monthly and its photo files have several standard sizes. The [iNaturalist open-data documentation](https://github.com/inaturalist/inaturalist-open-data) currently describes more than 400 million photos, images up to 2,048 px on the long edge, and metadata tables for photos, observations, observers, taxa, and projects. Each photo still retains its own Creative Commons license. iNaturalist's [research-download help](https://help.inaturalist.org/en/support/solutions/articles/151000223300-how-i-download-inaturalist-photos-to-use-for-research-) directs bulk researchers to this source.

iNaturalist does not own contributor photographs. Its [media reuse guidance](https://help.inaturalist.org/en/support/solutions/articles/151000169918-can-i-use-the-photos-and-sounds-that-are-posted-on-inaturalist-) says the uploader owns the media, the default is CC BY-NC, and unlicensed/all-rights-reserved media require explicit permission. Its [Terms of Use](https://www.inaturalist.org/pages/terms) separately prohibit using any iNaturalist data to train AI or ML systems **for commercial purposes**. This portfolio should therefore be documented as a personal, noncommercial research project. If the model or dataset is later intended for a commercial product or service, replace these images with owned/commissioned material or obtain explicit permission and appropriate legal review.

Location fields are unnecessary for segmentation and should be excluded from the published dataset. iNaturalist may obscure or hide coordinates chosen by observers or associated with taxa threatened by location disclosure ([geoprivacy documentation](https://help.inaturalist.org/en/support/solutions/articles/151000169938-what-is-geoprivacy-what-does-it-mean-for-an-observation-to-be-obscured-)). Retain only a broad country/region if it is scientifically useful and already public; do not attempt to infer a hidden site from dates, neighboring observations, image metadata, or backgrounds.

## Wikimedia Commons acquisition details

The [_Drosera_ category](https://commons.wikimedia.org/wiki/Category:Drosera) has 101 direct files and 163 subcategories at the time of research. Its species, leaves, flowers, seedlings, unidentified plants, and country subcategories make the usable pool larger, but the category is heterogeneous. Keep only raster photographs of living plants that show enough of the target plant to annotate.

Traverse the tree using the MediaWiki Action API's `categorymembers` generator, queueing namespace 14 subcategories and collecting namespace 6 files. For each file, request `imageinfo` with `url`, `dimensions`, and `extmetadata`. The official [`imageinfo` documentation](https://www.mediawiki.org/wiki/API:Imageinfo) describes original/thumbnail URLs, dimensions, MIME type, EXIF metadata, and extended metadata. The [CommonsMetadata documentation](https://www.mediawiki.org/wiki/Extension:CommonsMetadata/en) lists fields such as `Artist`, `Credit`, `DateTimeOriginal`, and license fields. Store the original file-description URL as the durable source of attribution and license terms.

Commons accepts only public-domain or freely licensed media that permit derivatives and commercial reuse; NC- and ND-only licenses are not accepted ([Commons licensing policy](https://commons.wikimedia.org/wiki/Commons:Project_scope#Must_be_freely_licensed_or_public_domain)). This makes Commons simpler than general image search, but it does not make every file license identical. CC BY-SA material can be adapted, but distributed adaptations must use a compatible ShareAlike license. For a uniform release, prefer public-domain/CC0 and CC BY files.

## GBIF, Smithsonian, and BOLD details

GBIF is an aggregator, not one photographic collection. The live query below returned 202,153 occurrence records with images, but it includes substantial iNaturalist overlap and does not mean there are 202,153 distinct, redistributable sundew photographs:

```text
https://api.gbif.org/v1/occurrence/search
  ?taxon_key=3190721
  &media_type=StillImage
  &limit=0
```

The search API pages at 300 records and has a 100,000-record query ceiling; larger retrievals should use an asynchronous occurrence download ([GBIF occurrence API](https://techdocs.gbif.org/en/openapi/v1/occurrence), [download guide](https://techdocs.gbif.org/en/data-use/api-downloads)). A Darwin Core Archive supplies `occurrence.txt` and `multimedia.txt`; the latter carries direct media and rights fields ([download-format documentation](https://techdocs.gbif.org/en/data-use/download-formats)). GBIF warns that occurrence images may be under terms more restrictive than the occurrence data and recommends checking the multimedia extension or publisher ([image API](https://techdocs.gbif.org/en/openapi/images), [citation guidance](https://www.gbif.org/citation-guidelines)). For this project, accept an image only when the media item itself is CC0 or CC BY. Use the publisher's URL for the best resolution, and deduplicate GBIF results against iNaturalist by canonical media URL, iNaturalist photo ID, checksum, and perceptual hash.

The Smithsonian's [Open Access FAQ](https://www.si.edu/openaccess/faq) permits copying, transforming, sharing, and commercial use of assets explicitly marked CC0. Search and bulk access are available through the [public API and weekly JSON repository](https://www.si.edu/openaccess/devtools). The Botany Department reports [3.8 million digitized herbarium images](https://naturalhistory.si.edu/research/botany/news-and-highlights/digitized) overall, and individual _Drosera_ sheets such as [_Drosera filiformis_](https://www.si.edu/object/drosera-filiformis-raf%3Anmnhbotany_2588767) show CC0 and detailed specimen metadata. Pressed, flattened specimens against standardized sheets are valuable for a domain-shift demonstration, but mixing them into a test set for living plants would inflate dataset size without measuring the intended task.

BOLD's [image-submission documentation](https://v3.boldsystems.org/index.php/resources/handbook?chapter=3_submissions.html) says it accepts images up to 20 MP, displays reduced thumbnails, does not distribute archived high-resolution files without the submitter's consent, assumes no ownership of uploaded images, and records a license chosen by the owner. Its [API](https://www.boldsystems.org/data/api/) supports taxonomic record retrieval, but the image licensing and resolution constraints make BOLD a weak source for this project.

## Existing plant-segmentation datasets

No open, pre-labeled sundew segmentation dataset was found in the official repositories checked. Two related datasets are useful only as benchmarks or transfer-learning references:

- [CVPPP 2017 Leaf Segmentation Challenge](https://www.plant-phenotyping.org/cvppp2017-challenge) provides top-down Arabidopsis and tobacco images with hand-labeled leaf masks. Its page says the challenge data may only be used to generate challenge submissions, so do not copy it into this repository or Hugging Face dataset.
- [PhenoBench](https://www.phenobench.org/) supplies high-resolution field/UAV images with semantic, plant-instance, and leaf-instance labels for sugar beet and weeds: more than 5,000 plants and 30,000 crop leaves. The [official development-kit FAQ](https://github.com/PRBonn/phenobench#frequently-asked-questions) states the dataset is CC BY-SA 4.0. It is a strong agricultural benchmark, but it contains no sundews and should remain a separately downloaded dependency rather than being merged into the new image archive.

These datasets can demonstrate transfer learning or establish that the pipeline works on a standard agricultural benchmark. Neither replaces a sundew-specific validation set.

## License rules for a publishable dataset

This is an operational licensing policy, not a legal opinion.

| Source-image license | Copy image to GitHub/Hugging Face? | Publish masks/overlays/crops? | Project policy |
| --- | --- | --- | --- |
| [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Yes | Yes | Accept. Credit is optional legally but still preserve provenance. |
| [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Yes, with attribution and license link | Yes, with attribution and an indication of changes | Accept. |
| [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Yes, with attribution | Yes, but distributed adaptations must use BY-SA or a compatible license | Exclude from v1 or publish in a clearly isolated ShareAlike subset. |
| [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) or other NC | Only within noncommercial terms | Same noncommercial constraint; SA may also apply | Exclude from the public training set to avoid downstream ambiguity. |
| [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/) or other ND | Unmodified redistribution may be allowed | Distribution of adapted images is prohibited | Exclude. |
| All rights reserved / missing / unclear | No | No | Exclude unless the copyright holder gives written permission. |

Do not place one blanket license over third-party photographs. The repository's software can be MIT or Apache-2.0, and the masks/annotation code can have a separate license, but each source image keeps its original license. Publish an image-level manifest with at least:

- internal image ID and SHA-256;
- source platform, source photo/media ID, observation/occurrence ID, and canonical source page;
- direct download URL and acquisition date;
- creator/photographer, required attribution text, rights holder if supplied;
- exact license code and license URL captured at acquisition;
- species/taxon label and identification quality;
- width, height, MIME type, and original-versus-resized status;
- a change notice such as “resized; binary segmentation annotation added”;
- annotation author, annotation version, review status, and mask checksum.

GitHub should contain code, small examples, and the manifest. Host the versioned image/mask archive on Hugging Face so it can have an explicit dataset card, per-row provenance, and a release tag. A dataset card must explain that third-party images retain their individual licenses and that the project's code license does not relicense them.

## Practical acquisition and curation plan

1. **Define the image domain first.** For v1, label the entire visible sundew plant or rosette as foreground. Accept living plants with a clear subject and enough surrounding background; reject flowers-only, macro tentacle shots, herbarium sheets, drawings, maps, screenshots, and images where the plant occupies too little of the frame.
2. **Pull 500-800 candidates from iNaturalist.** Require Research Grade, `captive=false`, and photo license CC0 or CC BY. Sample across species, observers, countries/regions, dates, backgrounds, plant sizes, and lighting. Do not treat Research Grade as perfect ground truth; screen every taxon label and image manually.
3. **Add 50-150 Commons candidates.** Traverse _Drosera_ subcategories, prefer public-domain/CC0 and CC BY photographs, and keep their complete attribution metadata.
4. **Deduplicate before annotation.** Collapse identical source IDs and URLs; then use SHA-256 for exact copies and a perceptual hash or embedding review for resized/cropped copies. GBIF records that point to iNaturalist are the same source, not new samples.
5. **Group before splitting.** Keep every photo from the same observation in one split. Also group by photographer and, when the public metadata support it, site/date or likely photographic session. A conservative first split is 70/15/15 by grouped observations, followed by an audit for near-duplicate backgrounds and repeated individual plants. Report performance both overall and by source platform/species/background type.
6. **Annotate 150-300 final images.** Use SAM-assisted polygons followed by human boundary correction. Mark ambiguous moss, neighboring vegetation, flower stalks, prey, and occluded leaves consistently in a written annotation guide. Double-review at least 10-20% and record inter-annotator IoU or boundary disagreement.
7. **Reserve a small owned test set.** Even 30-50 photographs taken by the project owner with a repeatable overhead protocol will provide a much stronger deployment test and eliminate source-license uncertainty. Record camera height, ground sampling distance or scale reference, lighting, plant identity, and date.
8. **Publish v1 as noncommercial research.** Keep precise occurrence coordinates out, retain image-level licenses and attribution, and document iNaturalist's commercial-AI restriction. A later UAV claim requires a separately collected nadir dataset with flight altitude, sensor, overlap, ground sampling distance, orthomosaic procedure, and field-trial split design.

## Main risks to document

- **Domain mismatch:** citizen-science closeups do not establish UAV performance.
- **Source leakage:** one photographer, observation, or field session can contribute several highly similar images.
- **Duplicate aggregation:** GBIF, Commons, and institutional portals may point to the same underlying photo.
- **Taxonomic noise:** community identifications can change; store acquisition date and source taxon ID.
- **License drift or deletion:** store the source page, exact license string, acquisition timestamp, and checksum; do not silently refresh assets without revalidating rights.
- **ShareAlike contamination:** overlays or other image adaptations may inherit SA obligations; isolate or exclude those files.
- **Sensitive locations:** strip coordinates and do not reverse-engineer obscured sites.
- **Photographer and site bias:** measure per-source and grouped-split performance instead of relying only on a random image split.
