import unittest
import tempfile
from pathlib import Path
from omni_lora.core.config import OmniConfig, DatasetConfig, TrainingConfig, ModelFamily, TrainingObjective

class TestConfig(unittest.TestCase):
    def test_omni_config_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            dataset_cfg = DatasetConfig(
                dataset_path=str(tmppath / "dataset"),
                trigger_word="sks",
                class_word="person"
            )
            training_cfg = TrainingConfig(
                model_family=ModelFamily.FLUX_DEV,
                base_model_path="black-forest-labs/FLUX.1-dev",
                objective=TrainingObjective.FACE_IDENTITY_100,
                output_dir=str(tmppath / "output")
            )
            config = OmniConfig(dataset=dataset_cfg, training=training_cfg)

            self.assertEqual(config.dataset.trigger_word, "sks")
            self.assertTrue(config.training.use_dora)
            self.assertEqual(config.training.network_dim, 32)

            # Test YAML serialization & deserialization
            yaml_file = tmppath / "test_config.yaml"
            config.to_yaml(yaml_file)
            self.assertTrue(yaml_file.exists())

            loaded = OmniConfig.from_yaml(yaml_file)
            self.assertEqual(loaded.dataset.trigger_word, "sks")
            self.assertEqual(loaded.training.model_family, ModelFamily.FLUX_DEV)

if __name__ == "__main__":
    unittest.main()
