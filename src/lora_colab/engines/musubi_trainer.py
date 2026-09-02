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

class MusubiTrainer(BaseTrainer):
    """
    Kohya Musubi-Tuner Trainer Engine.
    State-of-the-art trainer for Wan 2.1, Qwen2-VL, Z-Image/Kolors, and video architectures.
    """

    BACKEND_REPO_URL = "https://github.com/kohya-ss/musubi-tuner.git"
    DEFAULT_MUSUBI_DIR = "/content/backends/musubi-tuner"

    @classmethod
    def _ensure_backend_ready(cls):
        """Đảm bảo kho mã nguồn Musubi-Tuner đã sẵn sàng."""
        if os.path.exists("/content") and not os.path.exists(cls.DEFAULT_MUSUBI_DIR):
            console.print("[bold cyan]📥 Tải và cấu hình backend chính thức [bold]Musubi-Tuner[/bold]...[/bold cyan]")
            try:
                os.makedirs(os.path.dirname(cls.DEFAULT_MUSUBI_DIR), exist_ok=True)
                subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", cls.BACKEND_REPO_URL, cls.DEFAULT_MUSUBI_DIR])
                console.print("[bold green]✓ Musubi-Tuner đã sẵn sàng![/bold green]")
            except Exception as e:
                logger.warning(f"Could not clone musubi-tuner: {e}")

        if os.path.exists(cls.DEFAULT_MUSUBI_DIR):
            musubi_packages = [
                "toml>=0.10.2",
                "voluptuous>=0.13.0",
                "imagesize>=1.4.1",
                "albumentations>=1.4.0",
                "open-clip-torch>=2.24.0",
                "prodigyopt>=1.0",
                "lycoris-lora>=2.2.0"
            ]
            missing = [
                pkg for pkg in musubi_packages
                if not AutoEnvironmentManager.is_package_installed(pkg.split(">=")[0].strip())
            ]
            if missing:
                console.print(f"[bold yellow]📦 Tự động cài đặt trọn bộ phụ thuộc cho Musubi-Tuner ({len(missing)} gói):[/bold yellow] [dim]{', '.join(missing[:4])}...[/dim]")
                AutoEnvironmentManager.install_packages(missing, silent=True)
                console.print("[bold green]✓ Toàn bộ gói phụ trợ Musubi-Tuner đã sẵn sàng![/bold green]")

            AutoEnvironmentManager.ensure_engine_dependencies(cls.DEFAULT_MUSUBI_DIR)

    def _resolve_runner_path(self) -> str:
        self._ensure_backend_ready()
        possible_paths = [
            os.path.join(self.DEFAULT_MUSUBI_DIR, "train.py"),
            os.path.join(self.DEFAULT_MUSUBI_DIR, "wan_train_network.py"),
            os.path.join("/content/musubi-tuner/train.py"),
            os.path.join(os.getcwd(), "backends", "musubi-tuner", "train.py"),
            os.path.join(os.getcwd(), "musubi-tuner/train.py"),
            "train.py"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return os.path.join(self.DEFAULT_MUSUBI_DIR, "train.py")

    def build_command_or_config(self, resume_from: Optional[str] = None) -> List[str]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network

        runner_script = self._resolve_runner_path()
        cmd = [
            "accelerate", "launch",
            "--mixed_precision=" + ("bf16" if t_cfg.mixed_precision == "bf16" else "fp16"),
            runner_script,
            f"--dataset_config={d_cfg.dataset_dir}",
            f"--output_dir={t_cfg.checkpoint_dir}",
            f"--output_name={t_cfg.output_name}",
            f"--network_dim={n_cfg.network_dim}",
            f"--network_alpha={n_cfg.network_alpha}",
            f"--learning_rate={t_cfg.learning_rate}",
            f"--max_train_epochs={t_cfg.epochs}",
            f"--save_every_n_epochs=1",
            "--gradient_checkpointing",
        ]

        if resume_from and os.path.exists(resume_from):
            cmd.append(f"--resume={resume_from}")

        return cmd

    def train(self, resume_from: Optional[str] = None) -> bool:
        cmd = self.build_command_or_config(resume_from=resume_from)
        console.rule("[bold cyan]🚀 Musubi-Tuner (Kohya Next-Gen) Launch[/bold cyan]")
        console.print(f"  • Model: [cyan]{self.config.training.model_family}[/cyan]")
        console.print(f"  • Checkpoint Dir: [yellow]{self.config.training.checkpoint_dir}[/yellow]")
        console.print(f"  • Command: [dim]{' '.join(cmd)}[/dim]\n")

        env = os.environ.copy()
        if os.path.exists(self.DEFAULT_MUSUBI_DIR):
            env["PYTHONPATH"] = f"{self.DEFAULT_MUSUBI_DIR}:{env.get('PYTHONPATH', '')}"

        total_steps = self.config.training.max_train_steps or (self.config.training.epochs * 200)
        dashboard = LiveTrainingDashboard(
            model_name=self.config.training.model_family,
            engine_name="Musubi-Tuner",
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
