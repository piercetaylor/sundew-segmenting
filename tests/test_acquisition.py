from __future__ import annotations

import unittest

from sundew_segmentation.acquisition import (
    ALLOWED_LICENSES,
    LICENSE_URLS,
    Candidate,
    extract_candidates,
    original_photo_url,
    select_diverse,
)
from sundew_segmentation.curation import select_core_set, split_for_observer


class AcquisitionTests(unittest.TestCase):
    def test_original_photo_url_preserves_query(self) -> None:
        actual = original_photo_url("https://static.inat.example/photos/1/square.jpg?x=1")
        self.assertEqual(actual, "https://static.inat.example/photos/1/original.jpg?x=1")

    def test_extract_candidates_enforces_photo_license_and_omits_location(self) -> None:
        observation = {
            "id": 10,
            "quality_grade": "research",
            "captive": False,
            "location": "sensitive coordinate",
            "taxon": {"id": 20, "name": "Drosera capensis"},
            "user": {"login": "observer"},
            "photos": [
                {
                    "id": 30,
                    "license_code": "cc-by-nc",
                    "url": "https://example.test/30/square.jpg",
                    "original_dimensions": {"width": 2000, "height": 1500},
                },
                {
                    "id": 31,
                    "license_code": "cc-by",
                    "url": "https://example.test/31/square.jpg",
                    "attribution": "Photographer, CC BY",
                    "original_dimensions": {"width": 2000, "height": 1500},
                },
            ],
        }
        candidates = extract_candidates([observation])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].photo_id, 31)
        self.assertNotIn("location", candidates[0].__dict__)

    def test_noncommercial_needs_an_explicit_widening(self) -> None:
        """The species corpus accepts CC BY-NC; the segmentation default must not.

        docs/licence-policy.md turns on the two policies staying distinct, and
        the failure mode is silent: widening only the API query would acquire
        photographs the narrower dataset is not allowed to hold.
        """
        observation = {
            "id": 11,
            "quality_grade": "research",
            "captive": False,
            "taxon": {"id": 21, "name": "Drosera aberrans"},
            "user": {"login": "observer"},
            "photos": [
                {
                    "id": 40,
                    "license_code": "cc-by-nc",
                    "url": "https://example.test/40/square.jpg",
                    "attribution": "Photographer, CC BY-NC",
                    "original_dimensions": {"width": 2000, "height": 1500},
                }
            ],
        }
        self.assertEqual(extract_candidates([observation]), [])

        widened = extract_candidates([observation], allowed_licenses=LICENSE_URLS)
        self.assertEqual(len(widened), 1)
        self.assertEqual(widened[0].license_code, "cc-by-nc")
        self.assertEqual(widened[0].license_url, LICENSE_URLS["cc-by-nc"])

    def test_sharealike_and_noderivatives_are_never_permitted(self) -> None:
        """No caller can opt into SA or ND, because LICENSE_URLS does not carry them."""
        for code in ("cc-by-sa", "cc-by-nc-sa", "cc-by-nd", "cc-by-nc-nd"):
            self.assertNotIn(code, LICENSE_URLS)
        self.assertEqual(set(ALLOWED_LICENSES), {"cc0", "cc-by"})

    def test_diversity_caps_species_and_observer(self) -> None:
        def candidate(photo_id: int, taxon: str, observer: str) -> Candidate:
            return Candidate(
                observation_id=photo_id,
                photo_id=photo_id,
                image_url="https://example.test/image.jpg",
                source_page="https://example.test/observation",
                license_code="cc0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                attribution="",
                creator=observer,
                taxon_id=None,
                taxon_name=taxon,
                common_name=None,
                observer_login=observer,
                observed_on=None,
                source_width=1000,
                source_height=1000,
            )

        pool = [candidate(i, "species-a", "observer-a") for i in range(4)]
        pool += [candidate(10 + i, "species-b", f"observer-{i}") for i in range(4)]
        selected = select_diverse(pool, limit=8, max_per_species=2, max_per_observer=1, seed=1)
        self.assertLessEqual(sum(item.taxon_name == "species-a" for item in selected), 2)
        self.assertLessEqual(sum(item.observer_login == "observer-a" for item in selected), 1)

    def test_observer_split_is_stable_and_grouped(self) -> None:
        self.assertEqual(split_for_observer("same-user"), split_for_observer("same-user"))
        self.assertIn(split_for_observer("same-user"), {"train", "validation", "test"})

    def test_core_selection_excludes_rejected_and_respects_target(self) -> None:
        rows = [
            {"taxon_name": f"Drosera {index % 4}", "observer_login": f"user-{index}"}
            for index in range(20)
        ]
        selected, reserve = select_core_set(rows, {1, 3}, target_count=10)
        self.assertEqual(len(selected), 10)
        self.assertFalse({1, 3} & set(selected))
        self.assertEqual(set(selected) | set(reserve), set(range(20)) - {1, 3})


if __name__ == "__main__":
    unittest.main()
