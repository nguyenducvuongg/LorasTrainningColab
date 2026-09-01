import pytest
import os
import tempfile
import sys
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.dataset.normalizer import DatasetNormalizer
from lora_colab.dataset.cleaner import CaptionCleaner

def test_dataset_normalizer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(3):
            img = Image.new("RGB", (256, 256), color=(i * 50, 100, 150))
            img.save(os.path.join(tmp_dir, f"raw_{i}.jpg"))
            with open(os.path.join(tmp_dir, f"raw_{i}.txt"), "w") as f:
                f.write(f"sample caption {i}")

        res = DatasetNormalizer.normalize_folder(tmp_dir, prefix="face_test")
        assert res["processed_count"] == 3
        assert os.path.exists(os.path.join(tmp_dir, "face_test_0001.png"))
        assert os.path.exists(os.path.join(tmp_dir, "face_test_0001.txt"))
        assert os.path.exists(os.path.join(tmp_dir, "face_test_0003.png"))

def test_caption_cleaner():
    raw_tags = "1girl, solo, lowres, bad hands, beautiful eyes, blue hair"
    cleaned = CaptionCleaner.clean_text(raw_tags, trigger_word="sks character", is_danbooru_tags=True)
    assert cleaned.startswith("sks character, 1girl, solo, beautiful eyes, blue hair")
    assert "lowres" not in cleaned
    assert "bad hands" not in cleaned

    raw_natural = "a young woman standing in a garden with sunlight."
    cleaned_nat = CaptionCleaner.clean_text(raw_natural, trigger_word="cstyle", is_danbooru_tags=False)
    assert cleaned_nat.startswith("cstyle, a young woman standing in a garden")
