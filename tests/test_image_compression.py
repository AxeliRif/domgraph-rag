import sys
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_compression import compress_image, compress_images_dir  # noqa: E402


class CompressImageTests(unittest.TestCase):
    def test_scale_halves_dimensions(self):
        img = Image.new("RGB", (200, 100), color=(10, 20, 30))

        out = compress_image(img, scale=0.5)

        self.assertEqual(out.size, (100, 50))

    def test_rejects_scale_out_of_range(self):
        img = Image.new("RGB", (10, 10))
        with self.assertRaises(ValueError):
            compress_image(img, scale=1.0)
        with self.assertRaises(ValueError):
            compress_image(img, scale=0.0)


class CompressImagesDirTests(unittest.TestCase):
    def test_mirrors_filenames_at_reduced_size(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "images"
            dest_dir = Path(tmp) / "images_compressed"
            source_dir.mkdir()

            for name, size in [("a__tile_0000.png", (200, 100)), ("a__tile_0001.png", (80, 80))]:
                Image.new("RGB", size, color=(50, 50, 50)).save(source_dir / name)

            count = compress_images_dir(source_dir, dest_dir, scale=0.5)

            self.assertEqual(count, 2)
            self.assertEqual(set(p.name for p in dest_dir.glob("*.png")), {"a__tile_0000.png", "a__tile_0001.png"})
            with Image.open(dest_dir / "a__tile_0000.png") as out:
                self.assertEqual(out.size, (100, 50))
            with Image.open(dest_dir / "a__tile_0001.png") as out:
                self.assertEqual(out.size, (40, 40))


if __name__ == "__main__":
    unittest.main()
