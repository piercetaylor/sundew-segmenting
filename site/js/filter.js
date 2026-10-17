// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Pierce Taylor
//
// The species gallery's search box (species/index.html). Without JavaScript the box stays
// hidden and the full list shows.

const box = document.querySelector('.filter');
const input = document.getElementById('filter');
const count = document.getElementById('filter-count');
const cards = [...document.querySelectorAll('.sp-card')];
const sections = [...document.querySelectorAll('.sp-section')];

box.hidden = false;
input.addEventListener('input', () => {
  const q = input.value.trim().toLowerCase().replace(/^d(rosera)?\.?\s+/, '');
  let shown = 0;
  for (const c of cards) {
    const hit = !q || c.dataset.name.includes(q);
    c.hidden = !hit;
    if (hit) shown++;
  }
  for (const s of sections) s.hidden = !s.querySelector('.sp-card:not([hidden])');
  count.textContent = q ? `${shown} of ${cards.length} species` : '';
});
