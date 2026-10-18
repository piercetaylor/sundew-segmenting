// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// The identifier page (index.html): load the model, take a photo from any source,
// show the decision.
//
// Every photo, whether chosen, taken, dropped or pasted, goes through handleFile(blob, source).
// <body data-state> is loading | ready | working | done | error. #status keeps
// the technical text that the headless checks read (Ready / Done / Failed / This browser / Could not);
// #message is the plain-language line for people.

import { decide, COPY } from './decision.js';
import { createEngine } from './engine.js';
import { OUTLINE } from './features.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const inputs = [$('file'), $('camera')];
let engine = null;
let previewUrl = null;
let run = 0; // the latest photo; results of an earlier photo that finish later are dropped
let ready = false; // the model is loaded AND everything else the page loads has loaded (see init)

const setState = (s) => { document.body.dataset.state = s; };
const say = (text) => { $('message').textContent = text; };
const status = (text) => { $('status').textContent = text; };
const enable = (on) => {
  for (const i of inputs) i.disabled = !on;
  for (const b of document.querySelectorAll('.example-list button')) b.disabled = !on;
};

function progress(fraction) {
  const bar = $('progress');
  if (fraction === null) { bar.hidden = true; return; }
  bar.hidden = false;
  const pct = Math.round(Math.min(1, fraction) * 100);
  bar.firstElementChild.style.width = `${pct}%`;
  bar.setAttribute('aria-valuenow', String(pct));
}

// The model loads and runs in a worker (js/worker.js) so the page stays responsive; ?worker=0 runs it
// in the page, as browsers without module workers or a 2D OffscreenCanvas do anyway.
async function init() {
  $('hint').textContent = COPY.uploadHint;
  enable(false);
  setState('loading');
  status('Loading model...');
  progress(0);
  const onProgress = (got, total) => {
    status(`Downloading model ${(got / 1e6).toFixed(1)}${total ? ' / ' + (total / 1e6).toFixed(1) : ''} MB`);
    say(total ? `Downloading the identifier: ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB (only the first time)`
      : `Downloading the identifier: ${(got / 1e6).toFixed(0)} MB (only the first time)`);
    if (total) progress(got / total);
  };
  try {
    engine = await createEngine(onProgress, { worker: params.get('worker') !== '0' });
    // No request may follow a user's photo (site/test/privacy-check.mjs): finish loading the example
    // thumbnails and the italic heading font, used only for a named species, before accepting photos.
    await Promise.all([examplesLoaded, italicLoaded]);
    const i = engine.info;
    status(`Ready: sundew-species ${i.version}, onnxruntime-web ${i.ort}, ${i.backend}, ` +
      `${i.threads} thread(s), ${engine.inWorker ? 'in a worker' : 'in the page'}, loaded in ${(i.loadMs / 1000).toFixed(1)} s${i.fromCache ? ' (cached)' : ''}.`);
    progress(null);
    say('Ready. Choose a photo to identify.');
    ready = true;
    enable(true);
    setState('ready');
  } catch (e) {
    console.error(e);
    progress(null);
    status(`Could not load the model: ${e.message}`);
    say('The identifier could not load in this browser. Try reloading the page, or another browser.');
    setState('error');
  }
}

function el(parent, tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text) e.textContent = text;
  parent.appendChild(e);
  return e;
}

// species/<slug>.html, written by scripts/build_site_pages.py. A link only: nothing is fetched until it is followed.
const speciesPage = (label) => `species/${label.toLowerCase().replace(/ /g, '-')}.html`;

function matchList(parent, items) {
  const ul = el(parent, 'ul', 'matches');
  for (const t of items) {
    const li = el(ul, 'li', 'row');
    const a = el(li, 'a');
    a.href = speciesPage(t.label);
    el(a, 'i', '', t.label);
    el(li, 'small', '', t.pct);
    const track = el(li, 'div', 'track');
    track.setAttribute('aria-hidden', 'true');
    el(track, 'div', 'bar').style.width = `${Math.max(1, t.p * 100)}%`;
  }
}

// The decision rule (species at p >= 0.65 and energy >= 5.955, else a section at >= 0.7, else "not sure") lives in
// js/decision.js; it takes the raw logits and compares unrounded values.
function showResults(r) {
  const d = decide(r.logits, engine.labels);
  const box = $('results');
  box.replaceChildren();
  if (d.state === 'answer') {
    el(box, 'p', 'kicker', COPY.answer).id = 'result-title';
    const best = el(box, 'p');
    best.id = 'best';
    el(best, 'i', '', d.label);
    el(best, 'span', 'pct', d.top[0].pct); // the checks read the percentage as #best's last child
    const meter = el(box, 'div', 'meter');
    meter.setAttribute('aria-hidden', 'true');
    el(meter, 'span').style.width = `${d.top[0].p * 100}%`;
    const more = el(box, 'p', 'caveat');
    el(more, 'a', '', 'Photos and facts about this species').href = speciesPage(d.label);
    box.setAttribute('aria-labelledby', 'result-title');
  } else {
    el(box, 'p', 'note', d.state === 'section' ? COPY.section(d.section) : COPY.notSure).id = 'notsure';
    box.setAttribute('aria-labelledby', 'notsure');
  }
  el(box, 'h3', '', d.state === 'answer' ? COPY.otherMatches : COPY.possibleMatches);
  matchList(box, d.state === 'answer' ? d.top.slice(1) : d.top);
  el(box, 'p', 'caveat', COPY.caveat);
  el(box, 'p', 'caveat', COPY.accuracy);
  const actions = el(box, 'div', 'result-actions');
  const again = el(actions, 'button', 'btn secondary', 'Try another photo');
  again.type = 'button';
  again.addEventListener('click', () => { $('picker').scrollIntoView({ behavior: 'smooth', block: 'center' }); $('file').focus(); });
  const about = el(actions, 'a', 'btn secondary', 'How to read this');
  about.href = 'about.html#reading';
}

// example: the examples.json entry when the photo is one of the site's examples, for its credit line.
async function handleFile(f, source, example = null) {
  if (!f) return;
  if (!ready) { say('The identifier is still getting ready. Try again in a moment.'); return; }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(f);
  $('preview').src = previewUrl;
  $('preview').alt = example ? `Example photo of ${example.species}` : 'Your photo';
  showCredit(example);
  resetOutline();
  $('result').hidden = false;
  $('results').replaceChildren();
  status('Identifying...');
  say('Identifying…');
  setState('working');
  const mine = ++run;
  try {
    const r = await engine.classify(f);
    if (mine !== run) return;
    status(`Done in ${(r.timings.preprocess_ms + r.timings.inference_ms).toFixed(0)} ms ` +
      `(preprocess ${r.timings.preprocess_ms.toFixed(0)}, model ${r.timings.inference_ms.toFixed(0)}; ${source}).`);
    showResults(r);
    say('Done. The result is below.');
    setState('done');
    $('results').focus({ preventScroll: true });
    $('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (e) {
    console.error(e);
    if (mine !== run) return;
    // createImageBitmap rejects formats the browser cannot decode (e.g. HEIC outside Safari).
    const undecodable = e.name === 'InvalidStateError' || /decod/i.test(e.message) || /hei[cf]/i.test(f.type || f.name || '');
    const text = undecodable
      ? 'This browser could not read that photo. Please use a JPEG (on iPhone, HEIC photos may not open: export or share the photo as JPEG).'
      : `Failed: ${e.message}`;
    status(text);
    say(undecodable ? text : 'Something went wrong identifying that photo. Please try another one.');
    $('result').hidden = true;
    setState('error');
  }
}

// The plant outline (js/features.js). With OUTLINE false these do nothing and the controls stay hidden.
function resetOutline() {
  if (!OUTLINE) return;
  const c = $('outline');
  c.getContext('2d')?.clearRect(0, 0, c.width, c.height);
  c.hidden = !$('outline-toggle').checked;
  $('outline-area').textContent = '';
  $('outline-controls').hidden = false;
}

if (OUTLINE) {
  $('outline-toggle').addEventListener('change', (e) => { $('outline').hidden = !e.target.checked; });
}

// "Try an example": photos in img/examples/ (scripts/build_site_examples.py). A user's own photo never
// leaves the device; an example is fetched from this site only when its button is pressed.
function showCredit(ex) {
  const cap = $('photo-credit');
  cap.replaceChildren();
  cap.hidden = !ex;
  if (!ex) return;
  cap.append('Example: ');
  const sp = el(cap, 'a');
  sp.href = speciesPage(ex.species);
  el(sp, 'i', '', ex.species);
  cap.append(` as identified on iNaturalist. Photo ${ex.attribution}, `);
  el(cap, 'a', '', ex.licence).href = ex.licence_url;
  cap.append(', ');
  el(cap, 'a', '', 'observation').href = ex.source_page;
  cap.append('.');
}

// Resolves once the example list and every thumbnail have loaded (or failed); never rejects.
async function loadExamples() {
  let list;
  try {
    const res = await fetch('img/examples/examples.json');
    if (!res.ok) return;
    list = (await res.json()).examples;
  } catch (e) {
    console.warn('no examples:', e);
    return;
  }
  const thumbs = [];
  const ul = $('example-list');
  list.forEach((ex, i) => {
    const li = el(ul, 'li');
    const b = el(li, 'button', 'example');
    b.type = 'button';
    b.disabled = !ready;
    b.dataset.file = ex.file;
    const img = el(b, 'img');
    img.alt = `Example ${i + 1}`; // not lazy: every request must happen before a user's own photo is chosen
    thumbs.push(new Promise((done) => { img.onload = img.onerror = done; }));
    img.src = `img/examples/${ex.thumb}`;
    b.addEventListener('click', async () => {
      if (!ready) return;
      // Claim the run now: if the user picks their own photo while this downloads, theirs wins.
      const mine = ++run;
      say('Getting the example photo…');
      try {
        const res = await fetch(`img/examples/${ex.file}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        if (mine === run) await handleFile(blob, 'example', ex);
      } catch (e) {
        console.error(e);
        if (mine !== run) return;
        say('Could not load that example. Please try another.');
        setState('error');
      }
    });
  });
  $('examples').hidden = false;
  await Promise.all(thumbs);
}

for (const input of inputs) {
  input.addEventListener('change', () => {
    handleFile(input.files[0], input.id === 'camera' ? 'camera' : 'file');
    input.value = ''; // so choosing the same photo again still fires
  });
}

// Drag and drop onto the picker, and pasting an image anywhere on the page.
const picker = $('picker');
const firstImage = (items) => [...items].find((x) => x.kind === 'file' && x.type.startsWith('image/'))?.getAsFile();
picker.addEventListener('dragover', (e) => { e.preventDefault(); picker.classList.add('dragging'); });
picker.addEventListener('dragleave', (e) => { if (!picker.contains(e.relatedTarget)) picker.classList.remove('dragging'); });
picker.addEventListener('drop', (e) => {
  e.preventDefault();
  picker.classList.remove('dragging');
  const f = [...(e.dataTransfer?.files ?? [])].find((x) => x.type.startsWith('image/') || /\.(hei[cf]|jpe?g|png|webp)$/i.test(x.name));
  if (f) handleFile(f, 'drop');
  else say('That isn’t a photo. Drop an image file, such as a JPEG.');
});
// Stop a photo dropped just outside the picker from opening in the tab.
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());
document.addEventListener('paste', (e) => {
  const f = firstImage(e.clipboardData?.items ?? []);
  if (f) { e.preventDefault(); handleFile(f, 'paste'); }
});

const examplesLoaded = loadExamples();
const italicLoaded = document.fonts?.load('italic 600 1em "Source Serif 4"').catch(() => {}) ?? Promise.resolve();
init();
