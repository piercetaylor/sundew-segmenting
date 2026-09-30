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
// Not part of `node --test`: it needs Chrome and a running server.
//   npm i puppeteer-core && npx @puppeteer/browsers install chrome-headless-shell@stable
//   python -m http.server 8765 --bind 127.0.0.1        # from the repo root
//   PUPPETEER_CORE=<path to puppeteer-core> node site/test/privacy-check.mjs \
//       <chrome-headless-shell> http://127.0.0.1:8765/site/ <photo.jpg>
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [exe, url, photo] = process.argv.slice(2);
if (!exe || !url || !photo) {
  console.error('usage: node site/test/privacy-check.mjs <chrome> <site url> <photo.jpg>');
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
phase = 'after-photo';
const input = await page.$('#file');
await input.uploadFile(photo);
await page.waitForFunction(() => document.getElementById('results').children.length > 0, { timeout: 120000 });
await new Promise((r) => setTimeout(r, 2000)); // catch anything sent just after the result
const result = await page.$eval('#results', (e) => e.innerText.slice(0, 300));
const workers = page.workers().length;
await browser.close();

const sameOrigin = (u) => u.startsWith('blob:') || u.startsWith('data:') || new URL(u).origin === origin;
const problems = [];
for (const r of log) {
  if (!sameOrigin(r.url)) problems.push(`other origin: ${r.url}`);
  if (!['GET', 'HEAD'].includes(r.method)) problems.push(`${r.method}: ${r.url}`);
  if (r.body) problems.push(`request body (${r.body} bytes): ${r.url}`);
  if (r.phase === 'after-photo' && !r.url.startsWith('blob:') && !r.url.startsWith('data:')) {
    problems.push(`network request after the photo was chosen: ${r.url}`);
  }
}
if (workers) problems.push(`${workers} worker(s) running: their requests are not captured here`);

console.log(`${log.length} requests (${log.filter((r) => r.phase === 'load').length} before the photo):`);
for (const r of log) console.log(`  [${r.phase}] ${r.method} ${r.type} ${r.url.startsWith('data:') ? r.url.slice(0, 40) + '...' : r.url}`);
console.log(`result shown: ${JSON.stringify(result)}`);
if (problems.length) {
  console.log(`FAIL:\n  ${problems.join('\n  ')}`);
  process.exit(1);
}
console.log('PASS: every request is a same-origin GET without a body, and none happened after the photo was chosen.');
