from typing import List
from .base import BaseTrainer
from ..core.config import OmniConfig
from ..core.hardware import HardwareProfiler

class KohyaSDXLTrainer(BaseTrainer):
    """Huấn luyện SDXL, Pony V6, Illustrious-XL và SD 3.5 bằng sdxl_train_network.py."""

    def build_command(self) -> List[str]:
        t = self.config.training
        d = self.config.dataset
        profile = HardwareProfiler.analyze("sdxl")

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            "sd-scripts/sdxl_train_network.py",
            f"--pretrained_model_name_or_path={t.base_model_path}",
            f"--train_data_dir={d.dataset_path}",
            f"--output_dir={t.output_dir}",
            f"--output_name={t.output_name}",
            f"--network_dim={t.network_dim}",
            f"--network_alpha={t.network_alpha}",
            f"--network_module=networks.lora",
            f"--learning_rate={t.learning_rate}",
            f"--max_train_epochs={t.epochs}",
            f"--train_batch_size={t.batch_size or profile.recommended_batch_size}",
            f"--gradient_accumulation_steps={t.gradient_accumulation_steps or profile.recommended_grad_accum}",
            f"--mixed_precision={profile.recommended_precision}",
            f"--save_every_n_epochs={t.save_every_n_epochs}",
            f"--seed={t.seed}",
            "--enable_bucket",
            f"--max_bucket_reso={d.resolution}",
            "--save_model_as=safetensors",
            "--min_snr_gamma=5.0", # Cân bằng noise gradient tránh cháy hình
        ]

        if t.use_dora:
            cmd.append("--dora_wd")

        if t.cache_latents_to_disk:
            cmd.append("--cache_latents_to_disk")

        if profile.enable_flash_attention:
            cmd.append("--xformers")
        else:
            cmd.append("--sdpa")

        if t.optimizer_type.lower() == "prodigy":
            cmd.extend(["--optimizer_type=Prodigy", "--optimizer_args", "d_coef=1.0", "weight_decay=0.01"])
        else:
            cmd.extend([f"--optimizer_type={t.optimizer_type}"])

        return cmd
