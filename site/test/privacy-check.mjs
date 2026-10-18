// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Privacy check for the site: does anything leave the browser with the photo?
//
// Loads the page in headless Chrome, picks a photo, waits for the result, and
// records every network request from page load to the end. It fails if any
// request goes to another origin, uses a method other than GET/HEAD, carries a
// body, or happens after the photo is chosen (the model, labels and runtime
// must all be loaded before that). Run it on the site as it will ship, before
// saying "your photo is processed in your browser".
//
// With --examples in place of the photo, it instead presses every "try an example" button. After
// each press the only request allowed is a same-origin GET of that example's own JPEG
// (img/examples/<file>), and the result must match what the Python reference expects for it
// (img/examples/examples.json: state, and the species or section named).
//
// Not part of `node --test`: it needs Chrome and a running server. .github/workflows/pages.yml
// runs it on the built site before every deploy, with a photo from synthetic-photo.mjs.
//   npm i puppeteer-core && npx @puppeteer/browsers install chrome-headless-shell@stable
//   python -m http.server 8765 --bind 127.0.0.1        # from the repo root
//   PUPPETEER_CORE=<path to puppeteer-core> node site/test/privacy-check.mjs \
//       <chrome-headless-shell> http://127.0.0.1:8765/site/ <photo.jpg>
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [exe, url, photo] = process.argv.slice(2);
if (!exe || !url || !photo) {
  console.error('usage: node site/test/privacy-check.mjs <chrome> <site url> <photo.jpg | --examples>');
  process.exit(2);
}
const pptr = process.env.PUPPETEER_CORE
  ? await import(pathToFileURL(resolve(process.env.PUPPETEER_CORE, 'lib/puppeteer/puppeteer-core.js')).href)
  : await import('puppeteer-core');
const puppeteer = pptr.default ?? pptr;

const origin = new URL(url).origin;
const browser = await puppeteer.launch({ executablePath: exe, args: ['--no-sandbox'] });
const page = await browser.newPage();
const log = [];
let phase = 'load';
page.on('request', (r) => {
  const post = r.postData();
  log.push({ phase, method: r.method(), type: r.resourceType(), url: r.url(), body: post ? post.length : 0 });
});

await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });
await page.waitForFunction(() => !document.getElementById('file').disabled, { timeout: 120000 });
const problems = [];
const shown = () => page.evaluate(() => ({
  status: document.getElementById('status').textContent,
  result: document.getElementById('results').innerText.slice(0, 300),
  names: document.querySelectorAll('#best i, #results .row i').length,
  best: document.querySelector('#best i')?.textContent ?? null,
  notsure: document.getElementById('notsure')?.textContent ?? null,
}));
let last;
if (photo === '--examples') {
  const examples = await page.evaluate(() => fetch('img/examples/examples.json').then((r) => r.json()).then((d) => d.examples));
  for (const ex of examples) {
    phase = `example:${ex.file}`;
    await page.evaluate(() => { document.body.dataset.state = 'pressed'; });
    await page.click(`button.example[data-file="${ex.file}"]`);
    await page.waitForFunction(() => ['done', 'error'].includes(document.body.dataset.state), { timeout: 120000 });
    last = await shown();
    const want = ex.expected;
    const got = last.best ? `answer ${last.best}` : last.notsure === 'Not sure' ? 'not-sure' : `section ${last.notsure}`;
    const ok = want.state === 'answer' ? last.best === want.label
      : want.state === 'section' ? (last.notsure ?? '').endsWith(`section ${want.section}`)
        : last.notsure === 'Not sure';
    console.log(`${ok ? 'ok  ' : 'BAD '} ${ex.file} (${ex.species}): expected ${want.state} ${want.label ?? want.section ?? ''}, shown ${got}`);
    if (!ok) problems.push(`${ex.file}: expected ${want.state} ${want.label ?? want.section ?? ''}, shown ${got}`);
  }
  if (!examples.length) problems.push('no examples on the page');
} else {
  phase = 'after-photo';
  await (await page.$('#file')).uploadFile(photo);
  await page.waitForFunction(() => document.getElementById('results').children.length > 0, { timeout: 120000 });
}
await new Promise((r) => setTimeout(r, 2000)); // catch anything sent just after the result
last = await shown();
const { status, result, names } = last;
const workers = page.workers().length;
await browser.close();

const sameOrigin = (u) => u.startsWith('blob:') || u.startsWith('data:') || new URL(u).origin === origin;
// The page must also have worked: a result with five names (CI runs this on the built site).
if (!/^Done/.test(status) || names !== 5) problems.push(`no result: status ${JSON.stringify(status)}, ${names} names shown`);
for (const r of log) {
  if (!sameOrigin(r.url)) problems.push(`other origin: ${r.url}`);
  if (!['GET', 'HEAD'].includes(r.method)) problems.push(`${r.method}: ${r.url}`);
  if (r.body) problems.push(`request body (${r.body} bytes): ${r.url}`);
  if (r.phase === 'load' || r.url.startsWith('blob:') || r.url.startsWith('data:')) continue;
  const allowed = r.phase.startsWith('example:') && new URL(r.url).pathname.endsWith(`/img/examples/${r.phase.slice(8)}`);
  if (!allowed) problems.push(`network request after a photo was chosen (${r.phase}): ${r.url}`);
}
// The model runs in a dedicated worker (js/worker.js). Puppeteer reports a dedicated worker's requests
// as the page's; the worker is what fetches the model, so no .onnx request means they were missed.
if (workers && !log.some((r) => /\.onnx(\?|$)/.test(r.url))) {
  problems.push(`${workers} worker(s) running but no model request was captured: worker requests are not being logged`);
}

console.log(`${log.length} requests (${log.filter((r) => r.phase === 'load').length} before the photo):`);
for (const r of log) console.log(`  [${r.phase}] ${r.method} ${r.type} ${r.url.startsWith('data:') ? r.url.slice(0, 40) + '...' : r.url}`);
console.log(`status: ${status}`);
console.log(`result shown: ${JSON.stringify(result)}`);
if (problems.length) {
  console.log(`FAIL:\n  ${problems.join('\n  ')}`);
  process.exit(1);
}
console.log(photo === '--examples'
  ? 'PASS: every request is a same-origin GET without a body; after each example was pressed, only that example was fetched, and every example shows the expected result.'
  : 'PASS: every request is a same-origin GET without a body, and none happened after the photo was chosen.');
