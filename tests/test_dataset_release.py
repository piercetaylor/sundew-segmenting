import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_dataset_release import RELEASE_FILES, build_release


class DatasetReleaseTests(unittest.TestCase):
    def test_metadata_release_contains_inventory_and_no_images(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "repo"
            for relative in RELEASE_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative.as_posix(), encoding="utf-8")

            output = build_release(root, temporary / "release")
            manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["release_name"], "sundew-segmenting-core")
            self.assertEqual(manifest["payload"], "metadata_only")
            self.assertFalse(manifest["images_included"])
            self.assertEqual(
                {item["path"] for item in manifest["files"]},
                {path.as_posix() for path in RELEASE_FILES},
            )
            self.assertFalse(list(output.rglob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
