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
                subprocess.check_call(["git", "clone", "--depth", "1", cls.BACKEND_REPO_URL, cls.DEFAULT_BACKEND_DIR])
                console.print("[bold green]✓ Kohya sd-scripts đã sẵn sàng![/bold green]")
            except Exception as e:
                logger.warning(f"Could not clone sd-scripts: {e}")

        if os.path.exists(cls.DEFAULT_BACKEND_DIR):
            AutoEnvironmentManager.ensure_engine_dependencies(cls.DEFAULT_BACKEND_DIR)

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

    def build_command_or_config(self, resume_from: Optional[str] = None) -> List[str]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network

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
        ]

        if d_cfg.enable_bucketing:
            cmd.extend([
                "--enable_bucket",
                f"--min_bucket_reso={d_cfg.min_bucket_resolution}",
                f"--max_bucket_reso={d_cfg.max_bucket_resolution}"
            ])

        if t_cfg.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if t_cfg.cache_latents_to_disk:
            cmd.append("--cache_latents_to_disk")

        if t_cfg.cache_text_encoder_outputs:
            cmd.append("--cache_text_encoder_outputs")

        if d_cfg.shuffle_caption:
            cmd.append("--shuffle_caption")

        if t_cfg.sample_every_n_steps and t_cfg.sample_prompt:
            cmd.extend([
                f"--sample_every_n_steps={t_cfg.sample_every_n_steps}",
                f"--sample_prompts={t_cfg.sample_prompt}"
            ])

        # Optimizer selection
        opt = t_cfg.optimizer_type.lower()
        if "prodigy" in opt:
            cmd.extend([
                "--optimizer_type=Prodigy",
                "--optimizer_args", "decouple=True", "weight_decay=0.01", "d_coef=1.0", "use_bias_correction=True", "safeguard_warmup=True"
            ])
        elif "dadaptation" in opt or "dadapt" in opt:
            cmd.extend([
                "--optimizer_type=DAdaptAdamPreprint",
                "--optimizer_args", "decouple=True", "weight_decay=0.01"
            ])
        elif "adamw8bit" in opt or "8bit" in opt:
            cmd.extend(["--optimizer_type=AdamW8bit"])
        elif "lion" in opt:
            cmd.extend(["--optimizer_type=Lion"])
        else:
            cmd.extend(["--optimizer_type=AdamW"])

        # Auto-Resume
        if resume_from and os.path.exists(resume_from):
            cmd.append(f"--resume={resume_from}")

        # Text encoder training
        if t_cfg.text_encoder_lr and not t_cfg.cache_latents_to_disk:
            cmd.append(f"--text_encoder_lr={t_cfg.text_encoder_lr}")

        return cmd

    def _determine_script_name(self) -> str:
        fam = self.config.training.model_family.lower()
        if "sdxl" in fam or "pony" in fam or "illustrious" in fam:
            return "sdxl_train_network.py"
        elif "sd3" in fam:
            return "sd3_train_network.py"
        elif "flux" in fam:
            return "flux_train_network.py"
        else:
            return "train_network.py"

    def train(self, resume_from: Optional[str] = None) -> bool:
        cmd = self.build_command_or_config(resume_from=resume_from)
        
        console.print(f"[bold green]🚀 Khởi chạy Kohya Trainer cho {self.config.training.model_family}...[/bold green]")
        console.print(f"  • Base Model: [cyan]{self.config.training.base_model_path}[/cyan]")
        console.print(f"  • Lưu checkpoint trực tiếp tại: [yellow]{self.config.training.checkpoint_dir}[/yellow]")
        console.print(f"  • Command: [dim]{' '.join(cmd)}[/dim]")

        env = os.environ.copy()
        script_p = self._resolve_script_path()
        if script_p and os.path.exists(script_p):
            sd_dir = os.path.dirname(os.path.abspath(script_p))
            env["PYTHONPATH"] = f"{sd_dir}:{env.get('PYTHONPATH', '')}"

        total_steps = self.config.training.max_train_steps or (self.config.training.epochs * 200)
        dashboard = LiveTrainingDashboard(
            model_name=self.config.training.model_family,
            engine_name="Kohya sd-scripts",
            total_steps=total_steps,
            total_epochs=self.config.training.epochs,
            output_dir=self.config.training.checkpoint_dir
        )

        success = AutoEnvironmentManager.execute_with_self_healing(
            cmd,
            env=env,
            on_log_line=dashboard.parse_log_line
        )

        dashboard.close(success=success)
        return success
