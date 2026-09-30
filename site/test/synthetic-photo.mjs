// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// Write a synthetic test photo (PNG, 800 x 600, a deterministic green/red pattern)
// for checks of the built site in CI, where the fixture photos are not available.
// It is not a sundew; the checks only need the page to decode it and show a result.
//   node site/test/synthetic-photo.mjs out.png
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const out = process.argv[2];
if (!out) {
  console.error('usage: node site/test/synthetic-photo.mjs <out.png>');
  process.exit(2);
}
const W = 800, H = 600;

const CRC = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}

// Each row: filter byte 0, then RGB. Green background, reddish discs on a grid, a soft gradient.
const raw = Buffer.alloc(H * (1 + W * 3));
for (let y = 0; y < H; y++) {
  const o = y * (1 + W * 3);
  for (let x = 0; x < W; x++) {
    const dx = (x % 100) - 50, dy = (y % 100) - 50;
    const disc = dx * dx + dy * dy < 900;
    const p = o + 1 + x * 3;
    raw[p] = disc ? 190 : 40 + (x * 60) / W;
    raw[p + 1] = disc ? 60 : 110 + (y * 80) / H;
    raw[p + 2] = disc ? 70 : 40;
  }
}
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(W, 0);
ihdr.writeUInt32BE(H, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 2; // RGB
writeFileSync(out, Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
]));
console.log(`wrote ${out} (${W} x ${H})`);
