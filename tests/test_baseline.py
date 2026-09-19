from pathlib import Path
import unittest
from PIL import Image
import numpy as np
from sundew_segmentation.baseline import binary_metrics, inverse_frequency_weights, paired_samples
from sundew_segmentation.growth_forms import growth_form_for_taxon

class BaselineTests(unittest.TestCase):
    def test_metrics_perfect_and_empty(self):
        x = np.array([[1, 0], [0, 1]])
        self.assertEqual(binary_metrics(x, x)["dice"], 1.0)
        self.assertEqual(binary_metrics(np.zeros((2, 2)), np.zeros((2, 2)))["iou"], 1.0)

    def test_pairing_and_missing_mask_boundary(self):
        root = Path(self.id().replace(".", "_") + "_tmp")
        try:
            (root / "images/train").mkdir(parents=True)
            (root / "masks/train").mkdir(parents=True)
            Image.new("RGB", (4, 3)).save(root / "images/train/a.jpg")
            Image.new("L", (4, 3)).save(root / "masks/train/a.png")
            self.assertEqual(len(paired_samples(root / "images", root / "masks", "train")), 1)
            Image.new("RGB", (4, 3)).save(root / "images/train/b.jpg")
            with self.assertRaises(FileNotFoundError):
                paired_samples(root / "images", root / "masks", "train")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_inverse_frequency_weights_balance_groups(self):
        weights = inverse_frequency_weights(["rosette", "rosette", "branching"])
        self.assertEqual(weights, [0.5, 0.5, 1.0])
        self.assertEqual(sum(weights[:2]), weights[2])

    def test_growth_form_priors_cover_distinct_architectures(self):
        self.assertEqual(growth_form_for_taxon("Drosera rotundifolia"), "rosette")
        self.assertEqual(growth_form_for_taxon("Drosera auriculata"), "erect_or_branching")
        self.assertEqual(growth_form_for_taxon("Drosera binata"), "linear_or_forked")
        self.assertEqual(growth_form_for_taxon("Drosera pygmaea"), "dense_mat")
        self.assertEqual(growth_form_for_taxon("Drosera unknown"), "unknown")

if __name__ == "__main__":
    unittest.main()
