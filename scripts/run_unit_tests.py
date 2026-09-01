import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.core.hardware import HardwareProfiler, GPUProfile
from lora_colab.storage.gdrive_manager import GDriveWorkspaceManager
from lora_colab.storage.model_downloader import MODEL_REGISTRY
from lora_colab.storage.resume_manager import ResumeManager
from lora_colab.dataset.normalizer import DatasetNormalizer
from lora_colab.dataset.cleaner import CaptionCleaner
from lora_colab.dataset.captioning.gemini_api import SYSTEM_PROMPTS
import tempfile
from PIL import Image

class TestColabLoRAStudio(unittest.TestCase):

    def test_hardware_profiler(self):
        profile = HardwareProfiler.detect_and_profile("flux")
        self.assertIsInstance(profile, GPUProfile)
        self.assertIn(profile.tier, ["T4", "L4", "A100", "V100", "CPU", "GENERIC_CUDA"])
        self.assertGreaterEqual(profile.recommended_batch_size, 1)

    def test_model_registry(self):
        self.assertIn("flux-dev", MODEL_REGISTRY)
        self.assertIn("flux-kontext", MODEL_REGISTRY)
        self.assertIn("krea2-raw", MODEL_REGISTRY)
        self.assertIn("pony-v6", MODEL_REGISTRY)
        self.assertIn("sd15-base", MODEL_REGISTRY)

    def test_storage_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp_root:
            paths = GDriveWorkspaceManager.init_workspace(tmp_root)
            self.assertTrue(os.path.exists(os.path.join(tmp_root, "models", "flux")))
            self.assertTrue(os.path.exists(os.path.join(tmp_root, "outputs", "final_loras")))

            ckpt = os.path.join(tmp_root, "outputs", "checkpoints", "my_lora-step00000500.safetensors")
            with open(ckpt, "w") as f:
                f.write("test")

            res = ResumeManager.get_resume_status(os.path.join(tmp_root, "outputs", "checkpoints"))
            self.assertTrue(res["can_resume"])
            self.assertEqual(res["step"], 500)

    def test_normalizer_and_cleaner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(3):
                img = Image.new("RGB", (256, 256), color=(i * 50, 100, 150))
                img.save(os.path.join(tmp_dir, f"raw_{i}.png"))
                with open(os.path.join(tmp_dir, f"raw_{i}.txt"), "w") as f:
                    f.write(f"sample caption {i}")

            res = DatasetNormalizer.normalize_folder(tmp_dir, prefix="face_test")
            self.assertEqual(res["processed_count"], 3)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "face_test_0001.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "face_test_0001.txt")))

        raw_tags = "1girl, solo, lowres, bad hands, beautiful eyes, blue hair"
        cleaned = CaptionCleaner.clean_text(raw_tags, trigger_word="sks character", is_danbooru_tags=True)
        self.assertTrue(cleaned.startswith("sks character, 1girl, solo, beautiful eyes, blue hair"))
        self.assertNotIn("lowres", cleaned)

        self.assertIn("character", SYSTEM_PROMPTS)

if __name__ == "__main__":
    unittest.main()
