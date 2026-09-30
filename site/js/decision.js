// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// What the site says for one photo: a species, a section, or "not sure".
//
// Browser copy of src/sundew_segmentation/species_decision.py for the species
// model v1.0.0; site/test/decision.test.mjs checks the two agree. The numbers
// behind the thresholds are in docs/reports/species-abstain.md (validation,
// 110 listed species only): at p >= 0.65, 77.8% of photos get a species name
// and 90.1% of those names are right.
//
// Usage, with probs = softmaxT(logits) from classify.js (T = 0.74 applied):
//   const d = decide(probs, clf.labels);
//   d.state: 'answer' | 'section' | 'not-sure'; d.label, d.section, d.top[].pct
//
// Thresholds compare the unrounded probabilities: p = 0.646 shows as "65%"
// but is still "not sure".

export const MODEL_VERSION = '1.0.0';
export const MODEL_SHA256 = 'd7ee150eaf5d1c7ddae00a2a94e66ddcfd48b55166b940f4dfa35f0e3cc8af16';
export const TEMPERATURE = 0.74;
export const P_ANSWER = 0.65;
export const P_SECTION = 0.7;
export const TOP_K = 5;

// Species -> section (iNaturalist infrageneric ranks, data/species-110-sections.json).
export const SECTIONS = Object.freeze({
  "Drosera aberrans": "Erythrorhiza",
  "Drosera admirabilis": "Ptycnostigma",
  "Drosera alba": "Ptycnostigma",
  "Drosera aliciae": "Ptycnostigma",
  "Drosera androsacea": "Bryastrum",
  "Drosera anglica": "Drosera",
  "Drosera aquatica": "Arachnopus",
  "Drosera arcturi": "Arcturia",
  "Drosera auriculata": "Luniferae",
  "Drosera barbigera": "Bryastrum",
  "Drosera binata": "Phycopsis",
  "Drosera brevicornis": "Lasiocephala",
  "Drosera brevifolia": "Drosera",
  "Drosera bulbosa": "Erythrorhiza",
  "Drosera burkeana": "Ptycnostigma",
  "Drosera burmanni": "Thelocalyx",
  "Drosera capensis": "Ptycnostigma",
  "Drosera capillaris": "Drosera",
  "Drosera cayennensis": "Drosera",
  "Drosera cistiflora": "Ptycnostigma",
  "Drosera collina": "Erythrorhiza",
  "Drosera collinsiae": "Ptycnostigma",
  "Drosera communis": "Drosera",
  "Drosera cuneifolia": "Ptycnostigma",
  "Drosera dielsiana": "Ptycnostigma",
  "Drosera dilatatopetiolaris": "Lasiocephala",
  "Drosera drummondii": "Macrantha",
  "Drosera eneabba": "Bryastrum",
  "Drosera eremaea": "Macrantha",
  "Drosera ericgreenii": "Ptycnostigma",
  "Drosera erythrogyne": "Macrantha",
  "Drosera erythrorhiza": "Erythrorhiza",
  "Drosera esterhuyseniae": "Ptycnostigma",
  "Drosera filiformis": "Drosera",
  "Drosera finlaysonii": "Arachnopus",
  "Drosera floridana": "Drosera",
  "Drosera fulva": "Lasiocephala",
  "Drosera gigantea": "Ergaleium",
  "Drosera glabripes": "Ptycnostigma",
  "Drosera glanduligera": "Coelophylla",
  "Drosera gunniana": "Luniferae",
  "Drosera heterophylla": "Ergaleium",
  "Drosera hilaris": "Ptycnostigma",
  "Drosera hirsuta": "Macrantha",
  "Drosera hookeri": "Luniferae",
  "Drosera huegelii": "Ergaleium",
  "Drosera humilis": "Stolonifera",
  "Drosera hyperostigma": "Bryastrum",
  "Drosera indica": "Arachnopus",
  "Drosera intermedia": "Drosera",
  "Drosera latifolia": "Brasiliae",
  "Drosera linearis": "Drosera",
  "Drosera lunata": "Luniferae",
  "Drosera macrantha": "Macrantha",
  "Drosera macrophylla": "Erythrorhiza",
  "Drosera madagascariensis": "Ptycnostigma",
  "Drosera magna": "Erythrorhiza",
  "Drosera menziesii": "Ergaleium",
  "Drosera micrantha": "Bryastrum",
  "Drosera miniata": "Bryastrum",
  "Drosera minutiflora": "Bryastrum",
  "Drosera modesta": "Macrantha",
  "Drosera montana": "Brasiliae",
  "Drosera monticola": "Stolonifera",
  "Drosera murfetii": "Arcturia",
  "Drosera natalensis": "Ptycnostigma",
  "Drosera neesii": "Ergaleium",
  "Drosera neocaledonica": "Drosera",
  "Drosera nitidula": "Bryastrum",
  "Drosera ordensis": "Lasiocephala",
  "Drosera pallida": "Macrantha",
  "Drosera pauciflora": "Ptycnostigma",
  "Drosera peltata": "Luniferae",
  "Drosera petiolaris": "Lasiocephala",
  "Drosera planchonii": "Macrantha",
  "Drosera platypoda": "Stolonifera",
  "Drosera platystigma": "Bryastrum",
  "Drosera porrecta": "Stolonifera",
  "Drosera praefolia": "Erythrorhiza",
  "Drosera pulchella": "Bryastrum",
  "Drosera purpurascens": "Stolonifera",
  "Drosera pygmaea": "Bryastrum",
  "Drosera ramentacea": "Ptycnostigma",
  "Drosera roseana": "Bryastrum",
  "Drosera rosulata": "Erythrorhiza",
  "Drosera rotundifolia": "Drosera",
  "Drosera rupicola": "Stolonifera",
  "Drosera schmutzii": "Erythrorhiza",
  "Drosera scorpioides": "Bryastrum",
  "Drosera serpens": "Arachnopus",
  "Drosera sessilifolia": "Thelocalyx",
  "Drosera slackii": "Ptycnostigma",
  "Drosera spatulata": "Drosera",
  "Drosera spilos": "Bryastrum",
  "Drosera squamosa": "Erythrorhiza",
  "Drosera stenopetala": "Psychophila",
  "Drosera stolonifera": "Stolonifera",
  "Drosera stricticaulis": "Ergaleium",
  "Drosera subhirtella": "Macrantha",
  "Drosera sulphurea": "Ergaleium",
  "Drosera tomentosa": "Brasiliae",
  "Drosera tracyi": "Drosera",
  "Drosera trinervia": "Ptycnostigma",
  "Drosera tubaestylis": "Erythrorhiza",
  "Drosera uniflora": "Psychophila",
  "Drosera venusta": "Ptycnostigma",
  "Drosera whittakeri": "Erythrorhiza",
  "Drosera xerophila": "Ptycnostigma",
  "Drosera zeyheri": "Ptycnostigma",
  "Drosera zonaria": "Erythrorhiza",
});

/** Whole-number percentage for display; never "0%" or "100%". */
export function formatPct(p) {
  if (p < 0.005) return '<1%';
  if (p >= 0.995) return '>99%';
  return `${Math.round(p * 100)}%`;
}

/** Summed probability per section, sections in sorted order. */
export function sectionProbs(probs, labels) {
  const out = new Map([...new Set(labels.map((n) => SECTIONS[n]))].sort().map((s) => [s, 0]));
  labels.forEach((n, i) => out.set(SECTIONS[n], out.get(SECTIONS[n]) + probs[i]));
  return out;
}

/**
 * Decision for one photo.
 * @param {ArrayLike<number>} probs  110 probabilities in label order (softmax(logits / 0.74)).
 * @param {string[]} labels          the 110 labels in model output order.
 * @param {{sections?: boolean}} opts  sections: false turns the section fallback off.
 */
export function decide(probs, labels, { sections = true } = {}) {
  if (probs.length !== labels.length) throw new Error(`expected ${labels.length} probabilities, got ${probs.length}`);
  const missing = labels.filter((n) => !(n in SECTIONS));
  if (missing.length) throw new Error(`no section for ${missing.join(', ')}; decision.js is for model ${MODEL_VERSION}`);
  const order = [...labels.keys()].sort((a, b) => probs[b] - probs[a] || a - b);
  const top = order.slice(0, TOP_K).map((i) => ({ label: labels[i], p: probs[i], pct: formatPct(probs[i]) }));
  const p = top[0].p;
  const out = { state: 'not-sure', label: null, p, section: null, sectionP: null, top };
  if (p >= P_ANSWER) return { ...out, state: 'answer', label: top[0].label };
  if (!sections) return out;
  let best = null;
  for (const [s, sp] of sectionProbs(probs, labels)) if (best === null || sp > best[1]) best = [s, sp];
  out.sectionP = best[1];
  if (best[1] >= P_SECTION) return { ...out, state: 'section', section: best[0] };
  return out;
}

// Wording for the result. Keep it honest: the model only knows 110 species and
// never says "this is not a sundew".
export const COPY = Object.freeze({
  answer: 'Best match among 110 sundew species',
  section: (s) => `Not sure of the species. Probably section ${s}`,
  notSure: 'Not sure',
  possibleMatches: 'Possible matches',
  otherMatches: 'Other possible matches',
  caveat: 'This tool only knows 110 sundew (Drosera) species. It cannot tell when a photo shows something else, '
    + 'and it may still suggest a sundew name, sometimes with high confidence.',
  accuracy: 'In testing on photos of the listed species, about 9 in 10 of its confident answers were right. '
    + 'Rare species are identified less reliably.',
  uploadHint: 'Upload a clear photo of a sundew plant.',
});
