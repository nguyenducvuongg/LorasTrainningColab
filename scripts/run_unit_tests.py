import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.core.hardware import HardwareProfiler, GPUProfile
from lora_colab.core.environment import AutoEnvironmentManager
from lora_colab.storage.gdrive_manager import GDriveWorkspaceManager
from lora_colab.storage.model_downloader import MODEL_REGISTRY
from lora_colab.storage.resume_manager import ResumeManager
from lora_colab.dataset.normalizer import DatasetNormalizer
from lora_colab.dataset.cleaner import CaptionCleaner
from lora_colab.dataset.captioning.gemini_api import SYSTEM_PROMPTS
import tempfile
from PIL import Image

class TestColabLoRAStudio(unittest.TestCase):

    def test_environment_manager(self):
        info = AutoEnvironmentManager.get_runtime_info()
        self.assertIn("python_version", info)
        self.assertIn("torch_version", info)
        self.assertIn("cuda_version", info)

    def test_hardware_profiler(self):
        profile = HardwareProfiler.detect_and_profile("flux")
        self.assertIsInstance(profile, GPUProfile)
        self.assertIn(profile.tier, ["T4", "L4", "A100", "V100", "CPU", "GENERIC_CUDA"])
        self.assertGreaterEqual(profile.recommended_batch_size, 1)

    def test_model_registry(self):
        self.assertIn("flux-dev", MODEL_REGISTRY)
        self.assertIn("flux-kontext", MODEL_REGISTRY)
        self.assertIn("krea2-raw", MODEL_REGISTRY)
        self.assertIn("z-image-kolors", MODEL_REGISTRY)
        self.assertIn("qwen-image", MODEL_REGISTRY)
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

    def test_engine_factory(self):
        from lora_colab.engines.factory import EngineFactory
        from lora_colab.engines.aitoolkit_trainer import AIToolkitTrainer
        from lora_colab.engines.kohya_trainer import KohyaTrainer
        from lora_colab.engines.musubi_trainer import MusubiTrainer

        # flux-dev default is KohyaTrainer (flux_train_network.py for offline safetensors)
        self.assertEqual(EngineFactory.resolve_engine_type("flux-dev"), KohyaTrainer)
        # explicit choice for ai-toolkit still resolves to AIToolkitTrainer
        self.assertEqual(EngineFactory.resolve_engine_type("flux-dev", explicit_choice="ai-toolkit"), AIToolkitTrainer)
        self.assertEqual(EngineFactory.resolve_engine_type("krea2-raw"), AIToolkitTrainer)
        self.assertEqual(EngineFactory.resolve_engine_type("sdxl-base"), KohyaTrainer)
        self.assertEqual(EngineFactory.resolve_engine_type("wan2.1"), MusubiTrainer)
        self.assertEqual(EngineFactory.resolve_engine_type("qwen-image"), MusubiTrainer)

        # Test create_trainer with explicit_choice & engine_choice kwargs
        from lora_colab.core.config import LoRAConfig, DatasetConfig, NetworkConfig, TrainingConfig
        dummy_cfg = LoRAConfig(
            dataset=DatasetConfig(dataset_dir="/tmp"),
            network=NetworkConfig(),
            training=TrainingConfig(base_model_path="dummy.safetensors", model_family="sdxl-base")
        )
        trainer1 = EngineFactory.create_trainer(dummy_cfg, explicit_choice="kohya")
        self.assertIsInstance(trainer1, KohyaTrainer)
        trainer2 = EngineFactory.create_trainer(dummy_cfg, engine_choice="kohya")
        self.assertIsInstance(trainer2, KohyaTrainer)

    def test_dashboard_log_parsing(self):
        from lora_colab.monitoring.dashboard import LiveTrainingDashboard
        dash = LiveTrainingDashboard(total_steps=1650)
        
        # Test exact user log line from Colab:
        user_log_line = "mai_lora: 49%|█████████ | 810/1650 [28:41<27:46, 1.98s/it, lr: 2.6e-04 loss: 2.810e-02]"
        dash.parse_log_line(user_log_line)
        
        self.assertEqual(dash.current_step, 810)
        self.assertEqual(dash.total_steps, 1650)
        self.assertAlmostEqual(dash.current_loss, 0.0281, places=4)
        self.assertAlmostEqual(dash.current_lr, 0.00026, places=6)
        self.assertEqual(dash.speed_str, "1.98s/it")
        self.assertEqual(dash.eta_str, "27:46")

    def test_resume_manager_recursive_discovery(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Simulate nested directory structure from AI-Toolkit: checkpoints/my_lora/my_lora_00000800.safetensors
            sub_dir = os.path.join(tmp_dir, "my_lora")
            os.makedirs(sub_dir, exist_ok=True)
            ckpt_path = os.path.join(sub_dir, "my_lora_00000800.safetensors")
            with open(ckpt_path, "wb") as f:
                f.write(b"checkpoint_data")

            res = ResumeManager.find_latest_checkpoint(tmp_dir)
            self.assertIsNotNone(res)
            self.assertEqual(res[0], ckpt_path)
            self.assertEqual(res[1], 800)

    def test_kohya_flux_command_generation(self):
        from lora_colab.engines.kohya_trainer import KohyaTrainer
        from lora_colab.core.config import LoRAConfig
        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = os.path.join(tmp_dir, "models")
            flux_dir = os.path.join(models_dir, "flux")
            enc_dir = os.path.join(models_dir, "text_encoders")
            vae_dir = os.path.join(models_dir, "vae")
            os.makedirs(flux_dir, exist_ok=True)
            os.makedirs(enc_dir, exist_ok=True)
            os.makedirs(vae_dir, exist_ok=True)

            model_file = os.path.join(flux_dir, "flux1-dev.safetensors")
            clip_file = os.path.join(enc_dir, "clip_l.safetensors")
            t5_file = os.path.join(enc_dir, "t5xxl_fp8_e4m3fn.safetensors")
            vae_file = os.path.join(vae_dir, "ae.safetensors")

            for f in [model_file, clip_file, t5_file, vae_file]:
                with open(f, "wb") as fp:
                    fp.write(b"x" * 2000)

            from lora_colab.core.config import DatasetConfig, NetworkConfig, TrainingConfig
            cfg = LoRAConfig(
                dataset=DatasetConfig(dataset_dir=tmp_dir),
                network=NetworkConfig(),
                training=TrainingConfig(base_model_path=model_file, model_family="flux-dev")
            )
            
            trainer = KohyaTrainer(cfg)
            cmd = trainer.build_command_or_config()
            
            cmd_str = " ".join(cmd)
            self.assertIn("flux_train_network.py", cmd_str)
            self.assertIn("--network_module=networks.lora_flux", cmd_str)
            self.assertIn(f"--clip_l={clip_file}", cmd_str)
            self.assertIn(f"--t5xxl={t5_file}", cmd_str)
            self.assertIn(f"--ae={vae_file}", cmd_str)

if __name__ == "__main__":
    unittest.main()
