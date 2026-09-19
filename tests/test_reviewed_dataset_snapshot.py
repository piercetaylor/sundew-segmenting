import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.freeze_reviewed_dataset import SUPPORT_FILES, build_snapshot


class ReviewedDatasetSnapshotTests(unittest.TestCase):
    def test_snapshot_contains_only_accepted_pair_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            image = root / "data/curated/images/train/inat_1.jpg"
            mask = root / "data/curated/masks/train/inat_1.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4), "green").save(image)
            Image.new("L", (4, 4), 255).save(mask)

            metadata = {
                "curated_path": "data/curated/images/train/inat_1.jpg",
                "split": "train",
                "growth_form": "rosette",
                "license_code": "cc0",
            }
            metadata_path = root / "data/curated/metadata.jsonl"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

            payloads = {
                "metadata/annotation-audit.json": {
                    "tasks": 1,
                    "annotated_tasks": 1,
                    "quality_by_task": {"complete": 1},
                    "complete_without_mask": [],
                },
                "metadata/mask-audit.json": {
                    "masks": 1,
                    "errors": [],
                    "rows": [{"mask": "data/curated/masks/train/inat_1.png", "foreground_fraction": 1.0}],
                },
                "metadata/reviewed-mask-export.json": {
                    "exported": 1,
                    "skipped_complete": [],
                    "masks": [{
                        "task_id": 1,
                        "annotation_id": 2,
                        "split": "train",
                        "mask": "data/curated/masks/train/inat_1.png",
                    }],
                },
            }
            for target, source in SUPPORT_FILES.items():
                source_path = root / source
                source_path.parent.mkdir(parents=True, exist_ok=True)
                if target in payloads:
                    source_path.write_text(json.dumps(payloads[target]), encoding="utf-8")
                else:
                    source_path.write_text("snapshot documentation\n", encoding="utf-8")

            output = build_snapshot(root, Path(directory) / "snapshot", "test-version")
            manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))
            record = json.loads((output / "metadata/training-records.jsonl").read_text(encoding="utf-8"))

            self.assertEqual(manifest["pairs"], 1)
            self.assertTrue(manifest["test_split_locked"])
            self.assertEqual(record["review_quality"], "complete")
            self.assertTrue((output / record["snapshot_image"]).is_file())
            self.assertTrue((output / record["snapshot_mask"]).is_file())


if __name__ == "__main__":
    unittest.main()
