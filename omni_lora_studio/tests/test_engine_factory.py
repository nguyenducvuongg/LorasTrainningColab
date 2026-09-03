import unittest
from omni_lora.engines.factory import EngineFactory
from omni_lora.engines.kohya_flux import KohyaFluxTrainer
from omni_lora.engines.kohya_sdxl import KohyaSDXLTrainer
from omni_lora.engines.kohya_sd15 import KohyaSD15Trainer
from omni_lora.engines.aitoolkit_trainer import AIToolkitTrainer
from omni_lora.engines.musubi_trainer import MusubiTrainer

class TestEngineFactory(unittest.TestCase):
    def test_engine_resolution(self):
        self.assertEqual(EngineFactory.resolve_engine("flux-dev"), KohyaFluxTrainer)
        self.assertEqual(EngineFactory.resolve_engine("flux-schnell"), KohyaFluxTrainer)
        self.assertEqual(EngineFactory.resolve_engine("sdxl"), KohyaSDXLTrainer)
        self.assertEqual(EngineFactory.resolve_engine("pony-v6"), KohyaSDXLTrainer)
        self.assertEqual(EngineFactory.resolve_engine("sd15"), KohyaSD15Trainer)
        self.assertEqual(EngineFactory.resolve_engine("krea2"), AIToolkitTrainer)
        self.assertEqual(EngineFactory.resolve_engine("z-image"), MusubiTrainer)
        self.assertEqual(EngineFactory.resolve_engine("wan2.1"), MusubiTrainer)

if __name__ == "__main__":
    unittest.main()
