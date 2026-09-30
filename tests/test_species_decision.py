import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from sundew_segmentation.species_decision import (
    P_ANSWER, P_SECTION, TEMPERATURE, decide, load_sections, section_probs, softmax_t,
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


class SpeciesDecisionTests(unittest.TestCase):
    def test_constants_match_release(self):
        release = json.loads((ROOT / "release/species-v1.0.0/release.json").read_text())
        self.assertEqual(TEMPERATURE, release["temperature"])

    def test_softmax_t_matches_definition(self):
        z = np.array([[2.0, 1.0, 0.0]])
        e = np.exp(z / TEMPERATURE)
        np.testing.assert_allclose(softmax_t(z), e / e.sum())

    def test_answer_threshold_is_inclusive_and_unrounded(self):
        self.assertEqual(decide(with_top(P_ANSWER), LABELS, SECTIONS)["state"], "answer")
        self.assertEqual(decide(with_top(0.646), LABELS, SECTIONS)["state"], "not-sure")

    def test_section_fallback(self):
        members = [LABELS.index(n) for n in LABELS if SECTIONS[n] == "Ptycnostigma"][:2]
        probs = np.full(len(LABELS), 0.2 / (len(LABELS) - 2))
        probs[members] = [0.45, 0.35]
        d = decide(probs, LABELS, SECTIONS)
        self.assertEqual((d["state"], d["section"]), ("section", "Ptycnostigma"))
        self.assertGreaterEqual(d["section_p"], P_SECTION)
        self.assertEqual(len(d["top"]), 5)
        self.assertEqual(decide(probs, LABELS, None)["state"], "not-sure")
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
