from typing import List
from pathlib import Path
import yaml
from .base import BaseTrainer
from ..core.config import OmniConfig

class AIToolkitTrainer(BaseTrainer):
    """Huấn luyện chuyên sâu cho FLUX.1 và Krea2-Raw bằng AI-Toolkit (Ostris)."""

    def generate_config_file(self) -> str:
        t = self.config.training
        d = self.config.dataset
        out_yaml = Path(t.output_dir) / "aitoolkit_config.yaml"
        out_yaml.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "job": "extension",
            "config": {
                "name": t.output_name,
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": t.output_dir,
                    "device": "cuda:0",
                    "network": {
                        "type": "dora" if t.use_dora else "lora",
                        "linear": t.network_dim,
                        "linear_alpha": t.network_alpha
                    },
                    "save": {
                        "dtype": "bfloat16",
                        "save_every": t.save_every_n_epochs,
                        "max_step_saves_to_keep": 4
                    },
                    "datasets": [{
                        "folder_path": d.dataset_path,
                        "caption_ext": "txt",
                        "default_caption": f"{d.trigger_word} {d.class_word}",
                        "resolution": [d.resolution]
                    }],
                    "train": {
                        "batch_size": t.batch_size or 1,
                        "steps": t.epochs * 100,
                        "gradient_accumulation_steps": t.gradient_accumulation_steps or 2,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "prodigy" if "prodigy" in t.optimizer_type.lower() else "adamw8bit",
                        "lr": t.learning_rate
                    },
                    "model": {
                        "name_or_path": t.base_model_path,
                        "is_flux": True,
                        "quantize": True
                    }
                }]
            }
        }

        with open(out_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        return str(out_yaml)

    def build_command(self) -> List[str]:
        cfg_path = self.generate_config_file()
        return ["python3", "ai-toolkit/run.py", cfg_path]
