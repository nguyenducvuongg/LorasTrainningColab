import os
import subprocess
from typing import List, Optional
from pathlib import Path
from .base import BaseTrainer
from ..core.config import OmniConfig
from ..core.hardware import HardwareProfiler
from ..core.logger import console

class KohyaFluxTrainer(BaseTrainer):
    """Huấn luyện FLUX.1 (dev/schnell) bằng Kohya flux_train_network.py kết hợp Flow-Matching & FP8."""

    AUX_URLS = {
        "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "t5xxl_fp8_e4m3fn.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "ae.safetensors": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
    }

    @classmethod
    def find_or_download_aux(cls, subfolder: str, filename: str) -> str:
        candidates = [
            f"/content/drive/MyDrive/Colab_LoRA_Studio/models/{subfolder}/{filename}",
            f"/content/drive/MyDrive/OmniLoRA_Studio/models/{subfolder}/{filename}",
            f"/content/models/{subfolder}/{filename}",
            f"./models/{subfolder}/{filename}",
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                console.print(f"[green]✓ Đã tìm thấy file phụ trợ có sẵn: {p}[/green]")
                return p

        url = cls.AUX_URLS.get(filename)
        if not url:
            return ""

        target = f"/content/models/{subfolder}/{filename}"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        console.print(f"[cyan]📥 Đang tự động nạp file phụ trợ FLUX: [yellow]{filename}[/yellow]...[/cyan]")
        subprocess.run(["wget", "-q", "-c", url, "-O", target], check=False)
        return target

    @classmethod
    def resolve_base_model(cls, base_model_path: str) -> str:
        # Nếu đã là đường dẫn file .safetensors tồn tại
        if os.path.exists(base_model_path) and os.path.getsize(base_model_path) > 1024 * 1024:
            return base_model_path

        # Tìm trên Google Drive hoặc local
        candidates = [
            "/content/drive/MyDrive/Colab_LoRA_Studio/models/flux/flux1-dev.safetensors",
            "/content/drive/MyDrive/Colab_LoRA_Studio/models/flux/flux1-dev-fp8.safetensors",
            "/content/drive/MyDrive/Colab_LoRA_Studio/models/flux/flux1-schnell.safetensors",
            "/content/drive/MyDrive/OmniLoRA_Studio/models/flux/flux1-dev.safetensors",
            "/content/models/flux/flux1-dev-fp8.safetensors",
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
                console.print(f"[green]✓ Đã phát hiện mô hình FLUX nền tại: {p}[/green]")
                return p

        # Nếu chưa có, tải bản FLUX.1-dev FP8 siêu gọn (11.8GB) ungated về /content/models/flux/
        target = "/content/models/flux/flux1-dev-fp8.safetensors"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        console.print("[bold yellow]⚡ Chưa tìm thấy file FLUX trên Drive. Đang tự động nạp bản FLUX.1-dev FP8 (ungated mirror)...[/bold yellow]")
        flux_url = "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
        subprocess.run(["wget", "-q", "-c", flux_url, "-O", target], check=False)
        return target

    def build_command(self) -> List[str]:
        t = self.config.training
        d = self.config.dataset
        profile = HardwareProfiler.analyze("flux-dev")

        # Chuẩn bị đầy đủ 4 file thành phần bắt buộc của FLUX
        model_file = self.resolve_base_model(str(t.base_model_path))
        clip_l = self.find_or_download_aux("text_encoders", "clip_l.safetensors")
        t5xxl = self.find_or_download_aux("text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
        ae = self.find_or_download_aux("vae", "ae.safetensors")

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "4",
            "sd-scripts/flux_train_network.py",
            f"--pretrained_model_name_or_path={model_file}",
            f"--clip_l={clip_l}",
            f"--t5xxl={t5xxl}",
            f"--ae={ae}",
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
