import os
import sys
import subprocess
from typing import Dict, Any, List, Optional
from .base import BaseTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class MusubiTrainer(BaseTrainer):
    """
    Kohya Musubi-Tuner Trainer Engine.
    State-of-the-art trainer for Wan 2.1, Qwen2-VL, Z-Image/Kolors, and video architectures.
    """

    DEFAULT_MUSUBI_DIR = "/content/backends/musubi-tuner"

    def _resolve_runner_path(self) -> str:
        possible_paths = [
            os.path.join(self.DEFAULT_MUSUBI_DIR, "train.py"),
            os.path.join(self.DEFAULT_MUSUBI_DIR, "wan_train_network.py"),
            os.path.join("/content/musubi-tuner/train.py"),
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
            f"--max_train_epochs={t_cfg.max_train_epochs}",
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

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()

            process.wait()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error running Musubi-Tuner: {e}")
            return False
