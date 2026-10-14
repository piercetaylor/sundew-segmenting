import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from sundew_segmentation.species_decision import (
    E_ANSWER, P_ANSWER, P_SECTION, TEMPERATURE, decide, load_sections, neg_energy, section_probs, softmax_t,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "release/species-v1.0.0/labels.json").read_text())
SECTIONS = load_sections(ROOT / "data/species-110-sections.json", LABELS)


def _script():
    spec = importlib.util.spec_from_file_location("analyze_species_abstain", ROOT / "scripts/analyze_species_abstain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def with_top(p, i=0):
    probs = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    probs[i] = p
    return probs


def logits_for(probs, energy):
    """Logits with softmax(logits / T) == probs and T * logsumexp(logits / T) == energy."""
    return TEMPERATURE * np.log(probs) + energy


class SpeciesDecisionTests(unittest.TestCase):
    def test_constants_match_release(self):
        release = json.loads((ROOT / "release/species-v1.0.0/release.json").read_text())
        self.assertEqual(TEMPERATURE, release["temperature"])

    def test_softmax_t_matches_definition(self):
        z = np.array([[2.0, 1.0, 0.0]])
        e = np.exp(z / TEMPERATURE)
        np.testing.assert_allclose(softmax_t(z), e / e.sum())

    def test_logits_for_round_trips(self):
        l = logits_for(with_top(0.7), 6.2)
        np.testing.assert_allclose(softmax_t(l), with_top(0.7))
        self.assertAlmostEqual(float(neg_energy(l)), 6.2)

    def test_species_needs_both_p_and_energy(self):
        name = lambda p, e, **kw: decide(logits_for(with_top(p), e), LABELS, None, **kw)["state"]
        self.assertEqual(name(0.9, E_ANSWER + 0.01), "answer")
        self.assertEqual(name(0.9, E_ANSWER - 0.01), "not-sure")
        self.assertEqual(name(0.66, 8.0), "answer")
        self.assertEqual(name(0.646, 8.0), "not-sure")  # unrounded: shows as 65%
        self.assertEqual(name(0.9, E_ANSWER - 0.01, energy_threshold=None), "answer")  # earlier p-only rule

    def test_section_fallback(self):
        members = [LABELS.index(n) for n in LABELS if SECTIONS[n] == "Ptycnostigma"][:2]
        probs = np.full(len(LABELS), 0.2 / (len(LABELS) - 2))
        probs[members] = [0.45, 0.35]
        d = decide(logits_for(probs, 8.0), LABELS, SECTIONS)
        self.assertEqual((d["state"], d["section"]), ("section", "Ptycnostigma"))
        self.assertGreaterEqual(d["section_p"], P_SECTION)
        self.assertEqual(len(d["top"]), 5)
        self.assertEqual(decide(logits_for(probs, 8.0), LABELS, None)["state"], "not-sure")
        self.assertAlmostEqual(sum(section_probs(probs, LABELS, SECTIONS).values()), 1.0)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            decide([1.0], LABELS, SECTIONS)

    def test_every_label_has_a_section(self):
        self.assertEqual(list(SECTIONS), LABELS)


class AbstainAnalysisTests(unittest.TestCase):
    def test_rule_stats_on_a_hand_worked_case(self):
        mod = _script()
        answer = np.array([True, True, False, False])
        correct = np.array([True, False, True, False])
        in5 = np.array([True, True, True, False])
        y = np.array([0, 0, 1, 1])
        r = mod.rule_stats(answer, correct, in5, in5, y)
        self.assertEqual(r["coverage"], 0.5)
        self.assertEqual(r["acc_answered"], 0.5)
        self.assertEqual(r["wrong_answers_per_100_photos"], 25.0)
        self.assertEqual(r["abstained_true_in_top5"], 0.5)
        self.assertEqual(r["coverage_species_balanced"], 0.5)

    def test_balanced_averages_per_species(self):
        mod = _script()
        self.assertAlmostEqual(mod.balanced(np.array([True, True, True, False]), np.array([0, 0, 0, 1])), 0.5)


if __name__ == "__main__":
    unittest.main()
