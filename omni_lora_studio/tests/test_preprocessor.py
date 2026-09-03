import unittest
import tempfile
from pathlib import Path
from PIL import Image
from omni_lora.dataset.preprocessor import DatasetPreprocessor

class TestPreprocessor(unittest.TestCase):
    def test_sanitize_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
            raw_path = tmppath / "raw.png"
            img.save(raw_path)

            out_path = tmppath / "clean.jpg"
            success = DatasetPreprocessor.sanitize_image(str(raw_path), str(out_path))

            self.assertTrue(success)
            self.assertTrue(out_path.exists())

            with Image.open(out_path) as res:
                self.assertEqual(res.mode, "RGB")

if __name__ == "__main__":
    unittest.main()
