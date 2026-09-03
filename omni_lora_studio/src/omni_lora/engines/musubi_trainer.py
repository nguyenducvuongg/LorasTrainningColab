from typing import List
from .base import BaseTrainer
from ..core.config import OmniConfig

class MusubiTrainer(BaseTrainer):
    """Huấn luyện Next-Gen DiT & Video LoRA (Z-Image/Kolors, Wan 2.1, Qwen2-VL) bằng Musubi-Tuner."""

    def build_command(self) -> List[str]:
        t = self.config.training
        d = self.config.dataset

        cmd = [
            "accelerate", "launch",
            "musubi-tuner/train.py",
            f"--dataset_path={d.dataset_path}",
            f"--output_dir={t.output_dir}",
            f"--output_name={t.output_name}",
            f"--network_dim={t.network_dim}",
            f"--network_alpha={t.network_alpha}",
            f"--epochs={t.epochs}",
            f"--batch_size={t.batch_size or 1}",
            f"--learning_rate={t.learning_rate}",
            "--mixed_precision=bf16"
        ]
        return cmd
