// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Static checks of the generated species pages (scripts/build_site_pages.py), no browser needed:
//   node --test site/test/pages.test.mjs
// One page per model label; nothing on any page loads from another origin; species pages
// carry no scripts and no model; every photo has alt text and a credit under an accepted
// licence; every internal link and image resolves to a file in site/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOT = path.resolve(SITE, '..');
const labels = JSON.parse(fs.readFileSync(path.join(ROOT, 'release/species-v1.0.0/labels.json'), 'utf8'));
const species = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/site/species.json'), 'utf8')).species;
const slug = (n) => n.toLowerCase().replace(/ /g, '-');
const read = (rel) => fs.readFileSync(path.join(SITE, rel), 'utf8');
const pages = ['index.html', 'about.html', 'notice.html', 'species/index.html', ...labels.map((n) => `species/${slug(n)}.html`)];
const LICENCES = new Set(['cc0', 'cc-by', 'cc-by-nc']);

// [tag, attribute value] for every src/href in a page.
function refs(html) {
  const out = [];
  for (const m of html.matchAll(/<(\w+)\b[^>]*?\s(src|href)="([^"]*)"[^>]*>/g)) {
    out.push({ tag: m[1].toLowerCase(), attr: m[2], url: m[3], canonical: /\srel="canonical"/.test(m[0]) });
  }
  return out;
}

test('one species page per model label, and nothing else', () => {
  const files = fs.readdirSync(path.join(SITE, 'species')).filter((f) => f.endsWith('.html') && f !== 'index.html').sort();
  assert.deepEqual(files, labels.map((n) => `${slug(n)}.html`).sort());
  assert.deepEqual(species.map((s) => s.name), labels);
});

test('nothing is loaded from another origin', () => {
  for (const p of pages) {
    const html = read(p);
    for (const r of refs(html)) {
      if (r.tag === 'a' || r.canonical) continue; // links leave only when followed; canonical is never fetched
      assert.ok(!/^(https?:)?\/\//.test(r.url), `${p}: <${r.tag} ${r.attr}="${r.url}"> loads from another origin`);
    }
    assert.ok(!/@import|url\(\s*["']?(https?:)?\/\//.test(html), `${p}: CSS loads from another origin`);
  }
  const css = read('css/site.css');
  assert.ok(!/@import|url\(\s*["']?(https?:)?\/\//.test(css), 'site.css loads from another origin');
});

test('species pages load no scripts and no model; the gallery loads only filter.js', () => {
  for (const n of labels) {
    const html = read(`species/${slug(n)}.html`);
    assert.ok(!/<script\b/i.test(html), `${n}: has a script`);
    assert.ok(!/\.onnx|\/ort\//.test(html), `${n}: mentions the model or runtime`);
  }
  const scripts = [...read('species/index.html').matchAll(/<script\b[^>]*src="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(scripts, ['../js/filter.js']);
});

test('every photo has alt text, a credit, and an accepted licence', () => {
  for (const s of species) {
    const html = read(`species/${s.slug}.html`);
    assert.ok(s.photos.length > 0, `${s.name}: no photos`);
    for (const p of s.photos) {
      assert.ok(LICENCES.has(p.licence_code), `${s.name}: licence ${p.licence_code}`);
      assert.ok(p.attribution && p.source_page && p.licence_url, `${s.name}: incomplete credit for ${p.photo_id}`);
      const fig = html.match(new RegExp(`<figure[^>]*><img src="\\.\\./${p.src.replace(/[.]/g, '\\.')}"[^>]*>.*?</figure>`, 's'));
      assert.ok(fig, `${s.name}: photo ${p.photo_id} not shown`);
      assert.match(fig[0], /alt="[^"]+"/, `${s.name}: photo ${p.photo_id} has no alt text`);
      assert.ok(fig[0].includes(p.source_page) && fig[0].includes(p.licence_url), `${s.name}: photo ${p.photo_id} not credited`);
    }
  }
  for (const m of read('species/index.html').matchAll(/<img\b[^>]*>/g)) assert.match(m[0], /\salt="/, `gallery img without alt: ${m[0]}`);
});

test('every species page cites where the species was first described', () => {
  for (const s of species) {
    const html = read(`species/${s.slug}.html`);
    assert.match(html, /<h2 id="h-original">Original description<\/h2>/, s.name);
    assert.match(html, /<p class="citation">/, `${s.name}: no citation`);
    assert.match(html, /https:\/\/www\.ipni\.org\/n\//, `${s.name}: no IPNI link`);
  }
});

test('internal links and images resolve', () => {
  for (const p of pages) {
    for (const r of refs(read(p))) {
      if (r.canonical || /^(https?:|mailto:|#)/.test(r.url) || r.url.startsWith('//')) continue;
      const target = path.resolve(path.dirname(path.join(SITE, p)), r.url.split('#')[0]);
      const file = r.url.endsWith('/') || r.url === '' ? path.join(target, 'index.html') : target;
      // model/ and ort/ are copied in by the deploy workflow.
      if (/\/(model|ort)\//.test(file)) continue;
      assert.ok(fs.existsSync(file), `${p}: ${r.url} does not exist`);
    }
  }
});

test('example photos: files exist, CC0 or CC BY, credited, with an expected result', () => {
  const { examples } = JSON.parse(read('img/examples/examples.json'));
  assert.ok(examples.length >= 3, 'too few examples');
  const states = new Set(examples.map((x) => x.expected.state));
  for (const s of ['answer', 'section', 'not-sure']) assert.ok(states.has(s), `no ${s} example`);
  const credits = read('img/examples/ATTRIBUTION.md');
  for (const x of examples) {
    for (const f of [x.file, x.thumb]) assert.ok(fs.existsSync(path.join(SITE, 'img/examples', f)), `${f} missing`);
    assert.ok(['CC0 1.0', 'CC BY 4.0'].includes(x.licence), `${x.file}: licence ${x.licence}`);
    assert.ok(labels.includes(x.species), `${x.file}: unknown species ${x.species}`);
    assert.ok(credits.includes(x.source_page), `${x.file}: not in examples/ATTRIBUTION.md`);
    if (x.expected.state === 'answer') assert.ok(labels.includes(x.expected.label), `${x.file}: bad expected label`);
  }
});

test('notice.html, the old About page, redirects to about.html', () => {
  const html = read('notice.html');
  assert.match(html, /<meta http-equiv="refresh" content="0; url=about\.html">/);
  assert.match(html, /<a href="about\.html">/);
});

test('the plant outline stays off and hidden until it is switched on', () => {
  assert.match(read('js/features.js'), /^export const OUTLINE = false;$/m);
  const html = read('index.html');
  assert.match(html, /<canvas id="outline" hidden/);
  assert.match(html, /<div class="outline-controls" id="outline-controls" hidden>/);
});
