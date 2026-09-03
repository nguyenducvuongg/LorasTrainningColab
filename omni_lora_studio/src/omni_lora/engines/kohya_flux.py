from typing import List
from pathlib import Path
from .base import BaseTrainer
from ..core.config import OmniConfig
from ..core.hardware import HardwareProfiler

MODEL_MAPPING = {
    "flux.1-dev": "black-forest-labs/FLUX.1-dev",
    "flux-dev": "black-forest-labs/FLUX.1-dev",
    "flux.1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux-schnell": "black-forest-labs/FLUX.1-schnell",
}

class KohyaFluxTrainer(BaseTrainer):
    """Huấn luyện FLUX.1 (dev/schnell) bằng Kohya flux_train_network.py kết hợp Flow-Matching & FP8."""

    def build_command(self) -> List[str]:
        t = self.config.training
        d = self.config.dataset
        profile = HardwareProfiler.analyze("flux-dev")

        # Chuẩn hóa đường dẫn mô hình nền
        raw_path = str(t.base_model_path).strip()
        model_path = MODEL_MAPPING.get(raw_path.lower(), raw_path)

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            "sd-scripts/flux_train_network.py",
            f"--pretrained_model_name_or_path={model_path}",
            f"--train_data_dir={d.dataset_path}",
            f"--output_dir={t.output_dir}",
            f"--output_name={t.output_name}",
            f"--network_dim={t.network_dim}",
            f"--network_alpha={t.network_alpha}",
            f"--network_module=networks.lora_flux",
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
        ]

        if t.use_dora:
            cmd.append("--dora_wd")

        if t.cache_latents_to_disk:
            cmd.append("--cache_latents_to_disk")

        if profile.enable_fp8:
            cmd.append("--fp8_base")

        if profile.enable_cpu_offload:
            cmd.append("--cpu_offload_checkpointing")

        if profile.enable_flash_attention:
            cmd.append("--flash_attn")
        else:
            cmd.append("--sdpa")

        if t.optimizer_type.lower() == "prodigy":
            cmd.extend(["--optimizer_type=Prodigy", "--optimizer_args", "d_coef=1.0", "weight_decay=0.01"])
        else:
            cmd.extend([f"--optimizer_type={t.optimizer_type}"])

        return cmd
