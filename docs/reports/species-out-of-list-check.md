# Out-of-list spot check: 20 photos the species model doesn't know

The species model v1.0.0 only knows 110 *Drosera* species, and validation
([species-abstain.md](species-abstain.md)) says nothing about other photos.
Before the site launched, 20 photos outside the list went through the site as
it will ship, to check that its wording is honest. **This is a sanity check,
not a measurement:** 20 photos can't estimate a rate, and the threshold
(p >= 0.65) was not changed because of it.

**Result: on the site, 3 of 20 out-of-list photos got a confident species
name** (p >= 0.65). In Python ONNX Runtime it was 4 of 20: a coffee mug got
*D. binata* at 70% in Python and 60% ("Not sure") in the browser. The other
photos got "Not sure" in both; none got a section.

| Category | Photos | Named on the site (browser) | Named in Python |
| --- | ---: | --- | --- |
| Other carnivorous plants | 6 | 1: *Drosophyllum* as *D. filiformis*, 95% | the same |
| *Drosera* not in the 110 | 4 | 1: *D. regia* as *D. tracyi*, 69% | the same (68%) |
| Generic plants | 4 | 1: a grass as *D. finlaysonii*, 71% | the same (72%) |
| Not a plant | 4 | 0 (coffee mug: *D. binata* 60%, "Not sure") | 1: the coffee mug, 70% |
| Hard look-alikes (dew on grass, a rosette weed) | 2 | 0 | 0 |

## What this means for the site

The rule stays. The wording follows the plan for this outcome:

- A named result reads "Best match among 110 sundew species", never "This is".
- Under every result: "This tool only knows 110 sundew (*Drosera*) species. It
  cannot tell when a photo shows something else, and it may still suggest a
  sundew name, sometimes with high confidence."
- A non-plant came close to (Python: over) the line, so "Upload a clear photo
  of a sundew plant." sits above the upload button.

The confident mistakes are the ones the disclaimer predicts. *Drosophyllum* has
long thread leaves with sticky glands, much like *D. filiformis*. *D. regia*
has long strap leaves, like *D. tracyi*. The coffee mug shows that the model
has no "not a plant" answer.

## Method

- Photos: Wikimedia Commons, CC0, public domain or CC BY only, 1600 px
  thumbnails (iNaturalist's API was down for maintenance on the day). One
  search query per category, and every photo was looked at before scoring.
  Three first picks were replaced for not showing their category (a rice field
  for "hand", a single leaf for "rosette weed", barely visible dew). No photo
  was replaced after scoring. The list, with authors and licences, is in
  `data/out-of-list-check-photos.json`; the photos are not in the repository.
- Site: `site/` served locally, headless Chrome 154, onnxruntime-web 1.30.0
  WASM (1 thread), the page's own preprocessing and `decide()`; each photo
  picked through the file input, and the shown result read back
  (`site/test/privacy-check.mjs`, which also checked that nothing left the
  page).
- Python: the shipped `release/species-v1.0.0/model-int8.onnx`, ONNX Runtime
  1.30.0, one thread, EXIF rotation, short side 255 px bicubic, centre crop
  224, and `sundew_segmentation.species_decision`. Rerun with
  `PYTHONPATH=src python scripts/score_out_of_list_photos.py --cache <dir>`;
  raw output in [species-out-of-list-check.json](species-out-of-list-check.json).

## Limits

- 20 photos, one per query, chosen by the author: no rate can be read from
  them. A real open-set test (a few hundred photos per source, observer-disjoint,
  with the pass bar written down before scoring) is still to do.
- On these photos the browser and Python differ by up to about 10 percentage
  points (the mug) and swap close top-1 candidates twice (#1, #12), more than
  on the iNaturalist fixture used for browser parity. The cause (JPEG decoding
  of Commons thumbnails, int8 kernels, or both) was not looked into. A photo
  near 0.65 can land on either side.
- Commons photos are often tidier than phone snapshots.

## Photos and results

Browser percentages are what the page showed (whole numbers). Python
probabilities are softmax(logits / 0.74).

| # | Category | Photo (Wikimedia Commons) | Licence | Site (browser) says | Browser top-1 | Python top-1 (p) | Python top 5 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | other carnivorous plant | [Pinguicula Moranensis (15057446518)](https://commons.wikimedia.org/wiki/File:Pinguicula_Moranensis_(15057446518).jpg) | CC BY 2.0 | Not sure | *Drosera rotundifolia* 11% | *Drosera praefolia* (0.10, not sure) | praefolia 10%, rotundifolia 9%, squamosa 5%, modesta 4%, subhirtella 4% |
| 2 | other carnivorous plant | [Byblis Overflow (13552295935)](https://commons.wikimedia.org/wiki/File:Byblis_Overflow_(13552295935).jpg) | CC BY 2.0 | Not sure | *Drosera finlaysonii* 22% | *Drosera finlaysonii* (0.21, not sure) | finlaysonii 21%, indica 8%, serpens 6%, glabripes 5%, pallida 4% |
| 3 | other carnivorous plant | [Roridula gorgonias (2944506542)](https://commons.wikimedia.org/wiki/File:Roridula_gorgonias_(2944506542).jpg) | CC BY 2.0 | Not sure | *Drosera floridana* 30% | *Drosera floridana* (0.29, not sure) | floridana 29%, aquatica 13%, monticola 3%, tracyi 3%, zonaria 3% |
| 4 | other carnivorous plant | [Drosophyllum lusitanicum Habitat 2011-4-21 Si...](https://commons.wikimedia.org/wiki/File:Drosophyllum_lusitanicum_Habitat_2011-4-21_SierraMadrona.jpg) | Public domain | **species named** | *Drosera filiformis* 95% | *Drosera filiformis* (0.95, **named**) | filiformis 95%, floridana 1%, tracyi <1%, monticola <1%, binata <1% |
| 5 | other carnivorous plant | [Venus flytrap (Dionaea muscipula) Zagreb Bota...](https://commons.wikimedia.org/wiki/File:Venus_flytrap_(Dionaea_muscipula)_Zagreb_Botanical_Garden_1060.jpg) | CC BY 4.0 | Not sure | *Drosera bulbosa* 54% | *Drosera bulbosa* (0.51, not sure) | bulbosa 51%, whittakeri 13%, arcturi 5%, rupicola 3%, fulva 2% |
| 6 | other carnivorous plant | [Nepenthes rafflesiana ant](https://commons.wikimedia.org/wiki/File:Nepenthes_rafflesiana_ant.jpg) | CC BY 2.5 | Not sure | *Drosera murfetii* 20% | *Drosera murfetii* (0.18, not sure) | murfetii 18%, arcturi 9%, androsacea 5%, capensis 5%, magna 5% |
| 7 | Drosera not in the 110 | [Drosera regia - Botanischer Garten Braunschwe...](https://commons.wikimedia.org/wiki/File:Drosera_regia_-_Botanischer_Garten_Braunschweig_-_Braunschweig,_Germany_-_DSC04338.JPG) | CC0 | **species named** | *Drosera tracyi* 69% | *Drosera tracyi* (0.68, **named**) | tracyi 68%, ordensis 3%, dilatatopetiolaris 2%, capensis 2%, monticola 1% |
| 8 | Drosera not in the 110 | [Drosera adelae Exhibition of Carnivorous Plan...](https://commons.wikimedia.org/wiki/File:Drosera_adelae_Exhibition_of_Carnivorous_Plants_Prague_2016_1.jpg) | Public domain | Not sure | *Drosera latifolia* 28% | *Drosera latifolia* (0.29, not sure) | latifolia 29%, dilatatopetiolaris 18%, hilaris 7%, ericgreenii 5%, fulva 4% |
| 9 | Drosera not in the 110 | [Drosera paradoxa (2)](https://commons.wikimedia.org/wiki/File:Drosera_paradoxa_(2).JPG) | CC BY 3.0 | Not sure | *Drosera rotundifolia* 46% | *Drosera rotundifolia* (0.44, not sure) | rotundifolia 44%, petiolaris 9%, purpurascens 5%, dilatatopetiolaris 4%, fulva 4% |
| 10 | Drosera not in the 110 | [Drosera schizandra ne](https://commons.wikimedia.org/wiki/File:Drosera_schizandra_ne.jpg) | Public domain | Not sure | *Drosera collina* 27% | *Drosera collina* (0.25, not sure) | collina 25%, erythrorhiza 15%, tubaestylis 7%, bulbosa 4%, magna 3% |
| 11 | generic plant | [Macroscopic picture of Monstera Deliciosa](https://commons.wikimedia.org/wiki/File:Macroscopic_picture_of_Monstera_Deliciosa.jpg) | CC BY 4.0 | Not sure | *Drosera tracyi* 41% | *Drosera tracyi* (0.38, not sure) | tracyi 38%, fulva 13%, capensis 4%, burmanni 3%, lunata 3% |
| 12 | generic plant | [Bryum canariense (Canary bryum moss) (5575457...](https://commons.wikimedia.org/wiki/File:Bryum_canariense_(Canary_bryum_moss)_(5575457993).jpg) | CC BY 2.0 | Not sure | *Drosera spatulata* 19% | *Drosera rotundifolia* (0.19, not sure) | rotundifolia 19%, spatulata 19%, modesta 13%, anglica 10%, admirabilis 4% |
| 13 | generic plant | [Poa annua plant (7398667406)](https://commons.wikimedia.org/wiki/File:Poa_annua_plant_(7398667406).jpg) | CC BY 2.0 | **species named** | *Drosera finlaysonii* 71% | *Drosera finlaysonii* (0.72, **named**) | finlaysonii 72%, pallida 8%, tracyi 3%, dilatatopetiolaris 3%, fulva 2% |
| 14 | generic plant | [Echeveria strictiflora](https://commons.wikimedia.org/wiki/File:Echeveria_strictiflora.jpg) | CC BY 4.0 | Not sure | *Drosera gigantea* 43% | *Drosera gigantea* (0.41, not sure) | gigantea 41%, collina 29%, ordensis 12%, magna 5%, hilaris 1% |
| 15 | not a plant | [Two hands are open and facing upwards, showca...](https://commons.wikimedia.org/wiki/File:Two_hands_are_open_and_facing_upwards,_showcasing_a_vibrant_rainbow_reflection_on_their_palms.jpg) | CC BY 2.0 | Not sure | *Drosera cuneifolia* 15% | *Drosera cuneifolia* (0.15, not sure) | cuneifolia 15%, squamosa 13%, binata 11%, esterhuyseniae 6%, cistiflora 6% |
| 16 | not a plant | [White textured coat finish clean seamless bui...](https://commons.wikimedia.org/wiki/File:White_textured_coat_finish_clean_seamless_building_wall_texture.jpg) | CC0 | Not sure | *Drosera eneabba* 35% | *Drosera eneabba* (0.36, not sure) | eneabba 36%, minutiflora 9%, floridana 8%, micrantha 4%, pygmaea 4% |
| 17 | not a plant | [Coffee mugs cups](https://commons.wikimedia.org/wiki/File:Coffee_mugs_cups.jpg) | Public domain | Not sure | *Drosera binata* 60% | *Drosera binata* (0.70, **named**) | binata 70%, serpens 11%, modesta 8%, floridana 2%, humilis 1% |
| 18 | not a plant | [Domestic cat 2011 G02](https://commons.wikimedia.org/wiki/File:Domestic_cat_2011_G02.jpg) | Public domain | Not sure | *Drosera bulbosa* 13% | *Drosera bulbosa* (0.10, not sure) | bulbosa 10%, slackii 8%, collina 7%, macrophylla 5%, rupicola 5% |
| 19 | hard look-alike | [Grass at a lawn with morning dew 02](https://commons.wikimedia.org/wiki/File:Grass_at_a_lawn_with_morning_dew_02.jpg) | CC0 | Not sure | *Drosera modesta* 28% | *Drosera modesta* (0.31, not sure) | modesta 31%, indica 20%, finlaysonii 9%, arcturi 3%, serpens 3% |
| 20 | hard look-alike | [Hypochaeris radicata rosette4 (14609597536)](https://commons.wikimedia.org/wiki/File:Hypochaeris_radicata_rosette4_(14609597536).jpg) | CC BY 2.0 | Not sure | *Drosera finlaysonii* 25% | *Drosera finlaysonii* (0.23, not sure) | finlaysonii 23%, hilaris 17%, fulva 6%, collina 6%, ordensis 4% |
