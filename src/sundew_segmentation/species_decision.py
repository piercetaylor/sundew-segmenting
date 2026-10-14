"""What the site says for one photo: a species, a section, or "not sure".

Reference implementation of the site's rule for the species model v1.0.0.
site/js/decision.js is the browser copy; the test vectors in
site/test/fixtures/ are generated from this module, so the two must agree.

From the raw logits, with T = 0.74:

- p = top-1 of softmax(logits / T), the percentage the site shows;
- energy = T * logsumexp(logits / T) (negative energy: how strongly the photo
  matches any of the 110 species at all).

Rule (since 2026-09-30):

- p >= 0.65 AND energy >= 5.955: name the species;
- otherwise, if the summed probability of the best section is >= 0.7 (and the
  section fallback is on): name the section;
- otherwise: "not sure". The top 5 are shown in every case.

The energy condition was added after the open-set test
(docs/reports/species-open-set.md): with p alone about 1 in 5 photos of other
plants got a confident sundew name; with both, about 1 in 10
(docs/reports/species-open-set-fallback.md). Validation numbers for both rules
are in docs/reports/species-abstain.md. Thresholds compare unrounded values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_VERSION = "1.0.0"
TEMPERATURE = 0.74
P_ANSWER = 0.65
E_ANSWER = 5.955
P_SECTION = 0.7
TOP_K = 5


def softmax_t(logits: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """softmax(logits / T) over the last axis, in float64."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def neg_energy(logits: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """T * logsumexp(logits / T) over the last axis, in float64."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    m = z.max(axis=-1, keepdims=True)
    return temperature * (m[..., 0] + np.log(np.exp(z - m).sum(axis=-1)))


def load_sections(path: Path, labels: list[str]) -> dict[str, str]:
    """Species name -> section, for every label (data/species-110-sections.json)."""
    species = json.loads(Path(path).read_text())["species"]
    missing = [n for n in labels if n not in species]
    if missing:
        raise KeyError(f"no section for {missing}")
    return {n: species[n]["section"] for n in labels}


def section_probs(probs: np.ndarray, labels: list[str], sections: dict[str, str]) -> dict[str, float]:
    """Sum of species probabilities per section, sections in sorted order."""
    out = {s: 0.0 for s in sorted(set(sections[n] for n in labels))}
    for p, n in zip(probs, labels):
        out[sections[n]] += float(p)
    return out


def decide(logits, labels: list[str], sections: dict[str, str] | None = None,
           energy_threshold: float | None = E_ANSWER) -> dict:
    """Decision for one photo from its 110 raw logits (in label order).

    Pass ``sections=None`` to turn the section fallback off, and
    ``energy_threshold=None`` for the earlier p-only rule (used to reproduce
    reports written before 2026-09-30).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels),):
        raise ValueError(f"expected {len(labels)} logits, got shape {logits.shape}")
    probs = softmax_t(logits)
    energy = float(neg_energy(logits))
    order = sorted(range(len(labels)), key=lambda i: (-probs[i], i))
    top = [{"label": labels[i], "p": float(probs[i])} for i in order[:TOP_K]]
    p1 = top[0]["p"]
    out = {"state": "not-sure", "label": None, "p": p1, "energy": energy, "section": None, "section_p": None,
           "top": top}
    if p1 >= P_ANSWER and (energy_threshold is None or energy >= energy_threshold):
        out.update(state="answer", label=top[0]["label"])
    elif sections is not None:
        sp = section_probs(probs, labels, sections)
        best = max(sp, key=lambda s: (sp[s], -sorted(sp).index(s)))
        out["section_p"] = sp[best]
        if sp[best] >= P_SECTION:
            out.update(state="section", section=best)
    return out
