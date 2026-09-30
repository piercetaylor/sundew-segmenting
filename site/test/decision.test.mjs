// Run from the repo root: node --test site/test/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  decide, formatPct, sectionProbs, SECTIONS, MODEL_SHA256, TEMPERATURE, P_ANSWER, P_SECTION,
} from '../js/decision.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const labels = read('../../release/species-v1.0.0/labels.json');
const release = read('../../release/species-v1.0.0/release.json');
const sectionData = read('../../data/species-110-sections.json').species;
const vectors = read('./fixtures/decision-vectors.json');

const softmaxT = (logits, t) => {
  const z = logits.map((v) => v / t);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

// probs with top-1 = p, the rest spread so no section reaches P_SECTION.
function withTop(p, i = 0) {
  const probs = labels.map(() => (1 - p) / (labels.length - 1));
  probs[i] = p;
  return probs;
}

test('constants match release.json', () => {
  assert.equal(MODEL_SHA256, release.files['model-int8.onnx'].sha256);
  assert.equal(TEMPERATURE, release.temperature);
  assert.equal(vectors.temperature, TEMPERATURE);
  assert.equal(vectors.p_answer, P_ANSWER);
  assert.equal(vectors.p_section, P_SECTION);
});

test('sections cover exactly the model labels, as in data/species-110-sections.json', () => {
  assert.deepEqual(Object.keys(SECTIONS), labels);
  assert.deepEqual(vectors.labels, labels);
  for (const n of labels) assert.equal(SECTIONS[n], sectionData[n].section, n);
});

test(`agrees with the Python reference on ${vectors.cases.length} validation photos`, () => {
  const states = new Set();
  for (const c of vectors.cases) {
    const d = decide(softmaxT(c.logits, TEMPERATURE), labels);
    const e = c.expected;
    const where = `val_index ${c.val_index}`;
    assert.equal(d.state, e.state, where);
    assert.equal(d.label, e.label, where);
    assert.equal(d.section, e.section, where);
    assert.deepEqual(d.top.map((t) => t.label), e.top, where);
    assert.ok(Math.abs(d.p - e.p) < 1e-9, where);
    if (e.section_p === null) assert.equal(d.sectionP, null, where);
    else assert.ok(Math.abs(d.sectionP - e.section_p) < 1e-9, where);
    states.add(d.state);
  }
  assert.deepEqual([...states].sort(), ['answer', 'not-sure', 'section']);
});

test('threshold uses the unrounded probability', () => {
  assert.equal(decide(withTop(P_ANSWER), labels).state, 'answer');
  const d = decide(withTop(0.646), labels);
  assert.equal(d.state, 'not-sure');
  assert.equal(d.top[0].pct, '65%');
});

test('section fallback, and turning it off', () => {
  // Split 0.8 between two species of one section: no species reaches 0.65.
  const [a, b] = labels.filter((n) => SECTIONS[n] === 'Ptycnostigma').map((n) => labels.indexOf(n));
  const probs = labels.map(() => 0.2 / (labels.length - 2));
  probs[a] = 0.45;
  probs[b] = 0.35;
  const d = decide(probs, labels);
  assert.equal(d.state, 'section');
  assert.equal(d.section, 'Ptycnostigma');
  assert.ok(d.sectionP >= P_SECTION);
  assert.equal(d.top.length, 5);
  assert.equal(decide(probs, labels, { sections: false }).state, 'not-sure');
  const total = [...sectionProbs(probs, labels).values()].reduce((x, y) => x + y, 0);
  assert.ok(Math.abs(total - 1) < 1e-9);
});

test('rejects probabilities of the wrong length or unknown labels', () => {
  assert.throws(() => decide([1], labels), /expected 110/);
  assert.throws(() => decide([1], ['Drosera nonexistens']), /no section/);
});

test('whole-number percentages, never 0% or 100%', () => {
  assert.equal(formatPct(0), '<1%');
  assert.equal(formatPct(0.0049), '<1%');
  assert.equal(formatPct(0.005), '1%');
  assert.equal(formatPct(0.724), '72%');
  assert.equal(formatPct(0.9949), '99%');
  assert.equal(formatPct(0.995), '>99%');
});
