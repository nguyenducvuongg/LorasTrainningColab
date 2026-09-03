import os
import sys
import subprocess
from typing import Dict, Any, List, Optional
from .base import BaseTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console
from ..core.environment import AutoEnvironmentManager
from ..monitoring.dashboard import LiveTrainingDashboard

logger = setup_logger(__name__)

class KohyaTrainer(BaseTrainer):
    """
    Kohya_ss (sd-scripts) Trainer Engine.
    Industry standard for SDXL 1.0, Pony Diffusion V6, Illustrious-XL, SD 1.5, and SD 3.5.
    """

    BACKEND_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"
    DEFAULT_BACKEND_DIR = "/content/backends/sd-scripts"

    @classmethod
    def _ensure_backend_ready(cls):
        """Đảm bảo kho mã nguồn Kohya sd-scripts đã sẵn sàng."""
        if os.path.exists("/content") and not os.path.exists(cls.DEFAULT_BACKEND_DIR):
            console.print("[bold cyan]📥 Tải và cấu hình backend chính thức [bold]Kohya sd-scripts[/bold]...[/bold cyan]")
            try:
                os.makedirs(os.path.dirname(cls.DEFAULT_BACKEND_DIR), exist_ok=True)
                subprocess.check_call([
                    "git", "clone", "--depth", "1", "--recurse-submodules",
                    cls.BACKEND_REPO_URL, cls.DEFAULT_BACKEND_DIR
                ])
                console.print("[bold green]✓ Kohya sd-scripts đã sẵn sàng![/bold green]")
            except Exception as e:
                logger.warning(f"Could not clone sd-scripts: {e}")

        # Quét và cài đặt trọn gói các phụ thuộc cần thiết cho Kohya sd-scripts trong 1 lần duy nhất
        kohya_packages = [
            "toml>=0.10.2",
            "voluptuous>=0.13.0",
            "imagesize>=1.4.1",
            "albumentations>=1.4.0",
            "open-clip-torch>=2.24.0",
            "dadaptation>=3.1",
            "prodigyopt>=1.0",
            "lycoris-lora>=2.2.0",
            "tensorboard>=2.14.0",
            "huggingface-hub>=0.23.0",
        ]
        missing = [
            pkg for pkg in kohya_packages
            if not AutoEnvironmentManager.is_package_installed(pkg.split(">=")[0].strip())
        ]
        if missing:
            console.print(f"[bold yellow]📦 Tự động cài đặt siêu tốc phụ thuộc cho Kohya sd-scripts ({len(missing)} gói):[/bold yellow] [dim]{', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}[/dim]")
            AutoEnvironmentManager.install_packages(missing, silent=True)
            console.print("[bold green]✓ Toàn bộ gói phụ trợ Kohya sd-scripts đã nạp sẵn sàng 100%![/bold green]")

    def _resolve_script_path(self) -> str:
        self._ensure_backend_ready()
        script_name = self._determine_script_name()
        possible_paths = [
            os.path.join(self.DEFAULT_BACKEND_DIR, script_name),
            os.path.join("/content/sd-scripts", script_name),
            os.path.join(os.getcwd(), "backends", "sd-scripts", script_name),
            os.path.join("sd-scripts", script_name),
            os.path.join(os.getcwd(), "sd-scripts", script_name),
            script_name
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return os.path.join(self.DEFAULT_BACKEND_DIR, script_name)

    @staticmethod
    def _find_aux_file(base_model_path: str, subfolder: str, filename: str) -> Optional[str]:
        """Tự động tìm kiếm file phụ trợ (text encoder, VAE) trên Google Drive hoặc local."""
        candidates = []
        try:
            models_dir = os.path.dirname(os.path.dirname(os.path.abspath(base_model_path)))
            candidates.append(os.path.join(models_dir, subfolder, filename))
        except Exception:
            pass
        candidates.extend([
            os.path.join("/content/drive/MyDrive/Colab_LoRA_Studio/models", subfolder, filename),
            os.path.join("/content/models", subfolder, filename),
            os.path.join(os.getcwd(), "models", subfolder, filename),
        ])
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                return p
        return None

    def build_command_or_config(self, resume_from: Optional[str] = None) -> List[str]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network
        fam = t_cfg.model_family.lower()

        resolved_script = self._resolve_script_path()
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process=2",
            "--mixed_precision=" + ("bf16" if t_cfg.mixed_precision == "bf16" else "fp16"),
            resolved_script,
            f"--pretrained_model_name_or_path={t_cfg.base_model_path}",
            f"--train_data_dir={d_cfg.dataset_dir}",
            f"--output_dir={t_cfg.checkpoint_dir}",
            f"--output_name={t_cfg.output_name}",
            f"--save_model_as=safetensors",
            f"--resolution={d_cfg.resolution},{d_cfg.resolution}",
            f"--train_batch_size={t_cfg.batch_size}",
            f"--max_train_epochs={t_cfg.epochs}",
            f"--learning_rate={t_cfg.learning_rate}",
            f"--network_module={n_cfg.network_module}",
            f"--network_dim={n_cfg.network_dim}",
            f"--network_alpha={n_cfg.network_alpha}",
            f"--lr_scheduler={t_cfg.lr_scheduler}",
            f"--lr_warmup_steps={t_cfg.lr_warmup_steps}",
            f"--save_every_n_epochs={t_cfg.save_every_n_epochs}",
            f"--mixed_precision={t_cfg.mixed_precision}",
            f"--save_precision={'bf16' if t_cfg.mixed_precision == 'bf16' else 'fp16'}",
            f"--caption_extension={d_cfg.caption_extension}",
            f"--logging_dir={t_cfg.logging_dir}",
        ]

        # Tự động nạp Text Encoders & VAE theo từng kiến trúc model
        if "flux" in fam or resolved_script.endswith("flux_train_network.py"):
            clip_l = self._find_aux_file(t_cfg.base_model_path, "text_encoders", "clip_l.safetensors")
            t5xxl = self._find_aux_file(t_cfg.base_model_path, "text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
            ae = self._find_aux_file(t_cfg.base_model_path, "vae", "ae.safetensors")
            if clip_l:
                cmd.append(f"--clip_l={clip_l}")
            if t5xxl:
                cmd.append(f"--t5xxl={t5xxl}")
            if ae:
                cmd.append(f"--ae={ae}")
            # Flux yêu cầu networks.lora_flux
            for idx, arg in enumerate(cmd):
                if arg.startswith("--network_module="):
                    cmd[idx] = "--network_module=networks.lora_flux"
                    break

        elif "sd3" in fam or resolved_script.endswith("sd3_train_network.py"):
            clip_l = self._find_aux_file(t_cfg.base_model_path, "text_encoders", "clip_l.safetensors")
            clip_g = self._find_aux_file(t_cfg.base_model_path, "text_encoders", "clip_g.safetensors")
            t5xxl = self._find_aux_file(t_cfg.base_model_path, "text_encoders", "t5xxl_fp8_e4m3fn.safetensors")
            if clip_l:
                cmd.append(f"--clip_l={clip_l}")
            if clip_g:
                cmd.append(f"--clip_g={clip_g}")
            if t5xxl:
                cmd.append(f"--t5xxl={t5xxl}")

        elif any(k in fam for k in ["sdxl", "pony", "illustrious"]):
            sdxl_vae = self._find_aux_file(t_cfg.base_model_path, "vae", "sdxl_vae.safetensors")
            if sdxl_vae:
                cmd.append(f"--vae={sdxl_vae}")

        # Bucket resolution: dùng field chuẩn min_bucket_res/max_bucket_res từ DatasetConfig
        if d_cfg.enable_bucketing:
            cmd.extend([
                "--enable_bucket",
                f"--min_bucket_reso={d_cfg.min_bucket_res}",
                f"--max_bucket_reso={d_cfg.max_bucket_res}",
            ])

        if t_cfg.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if t_cfg.gradient_accumulation_steps > 1:
            cmd.append(f"--gradient_accumulation_steps={t_cfg.gradient_accumulation_steps}")

        if t_cfg.cache_latents_to_disk:
            cmd.append("--cache_latents_to_disk")
        elif t_cfg.cache_latents:
            cmd.append("--cache_latents")

        # cache_text_encoder_outputs (an toàn: dùng getattr)
        if getattr(t_cfg, "cache_text_encoder_outputs", False):
            cmd.append("--cache_text_encoder_outputs")

        if d_cfg.shuffle_caption:
            cmd.append("--shuffle_caption")

        if t_cfg.sample_every_n_steps and t_cfg.sample_prompt:
            cmd.extend([
                f"--sample_every_n_steps={t_cfg.sample_every_n_steps}",
                f"--sample_prompts={t_cfg.sample_prompt}",
                "--sample_sampler=euler_a",
            ])

        # Noise offset
        if getattr(t_cfg, "noise_offset", 0.0) > 0:
            cmd.append(f"--noise_offset={t_cfg.noise_offset}")

        # Min SNR Gamma
        if getattr(t_cfg, "min_snr_gamma", None):
            cmd.append(f"--min_snr_gamma={t_cfg.min_snr_gamma}")

        # Continue from existing LoRA weights
        if getattr(t_cfg, "network_weights", None) and os.path.exists(t_cfg.network_weights):
            cmd.append(f"--network_weights={t_cfg.network_weights}")

        # Optimizer selection
        opt = t_cfg.optimizer_type.lower()
        if "prodigy" in opt:
            cmd.extend([
                "--optimizer_type=Prodigy",
                "--optimizer_args",
                "decouple=True",
                "weight_decay=0.01",
                "d_coef=1.0",
                "use_bias_correction=True",
                "safeguard_warmup=True",
            ])
            # Prodigy requires LR = 1.0 and cosine scheduler
            cmd.append("--lr_scheduler=cosine")
        elif "dadaptation" in opt or "dadapt" in opt:
            cmd.extend([
                "--optimizer_type=DAdaptAdamPreprint",
                "--optimizer_args",
                "decouple=True",
                "weight_decay=0.01",
            ])
        elif "adamw8bit" in opt or "8bit" in opt:
            cmd.extend(["--optimizer_type=AdamW8bit"])
        elif "adafactor" in opt:
            cmd.extend(["--optimizer_type=Adafactor", "--scale_lr"])
        elif "lion" in opt:
            cmd.extend(["--optimizer_type=Lion"])
        else:
            cmd.extend(["--optimizer_type=AdamW"])

        # Auto-Resume
        if resume_from and os.path.exists(resume_from):
            cmd.append(f"--resume={resume_from}")

        # Text encoder training (chỉ khi không cache latents)
        if t_cfg.text_encoder_lr and not t_cfg.cache_latents_to_disk:
            cmd.append(f"--text_encoder_lr={t_cfg.text_encoder_lr}")

        return cmd

    def _determine_script_name(self) -> str:
        fam = self.config.training.model_family.lower()
        if any(k in fam for k in ["sdxl", "pony", "illustrious"]):
            return "sdxl_train_network.py"
        elif any(k in fam for k in ["sd35", "sd3.5", "sd3"]):
            return "sd3_train_network.py"
        elif "flux" in fam:
            return "flux_train_network.py"
        else:
            return "train_network.py"

    def train(self, resume_from: Optional[str] = None) -> bool:
        total_steps = self.config.training.max_train_steps or (self.config.training.epochs * 200)

        # 1. Khởi tạo Dashboard NGAY — hiển thị trạng thái setup trước khi clone/install
        dashboard = LiveTrainingDashboard(
            model_name=self.config.training.model_family,
            engine_name="Kohya sd-scripts",
            total_steps=total_steps,
            total_epochs=self.config.training.epochs,
            output_dir=self.config.training.checkpoint_dir
        )
        dashboard.set_status("⚙️ Đang chuẩn bị backend Kohya sd-scripts... (có thể mất 3-8 phút lần đầu)")
        dashboard.render()

        # 2. Clone & cài packages
        cmd = self.build_command_or_config(resume_from=resume_from)

        dashboard.set_status("✅ Backend sẵn sàng! Đang khởi chạy training...")
        dashboard.render()

        console.print(f"[bold green]🚀 Khởi chạy Kohya Trainer cho {self.config.training.model_family}...[/bold green]")
        console.print(f"  • Base Model: [cyan]{self.config.training.base_model_path}[/cyan]")
        console.print(f"  • Dataset: [cyan]{self.config.dataset.dataset_dir}[/cyan]")
        console.print(f"  • Lưu checkpoint trực tiếp tại: [yellow]{self.config.training.checkpoint_dir}[/yellow]")
        console.print(f"  • Optimizer: [green]{self.config.training.optimizer_type}[/green] | LR: [green]{self.config.training.learning_rate}[/green]")
        console.print(f"  • Command: [dim]{' '.join(cmd[:8])}...[/dim]")

        env = os.environ.copy()
        script_p = self._resolve_script_path()
        if script_p and os.path.exists(script_p):
            sd_dir = os.path.dirname(os.path.abspath(script_p))
            env["PYTHONPATH"] = f"{sd_dir}:{env.get('PYTHONPATH', '')}"

        # 3. Bắt đầu training thực sự
        success = AutoEnvironmentManager.execute_with_self_healing(
            cmd,
            env=env,
            on_log_line=dashboard.parse_log_line
        )

        dashboard.close(success=success)
        return success
