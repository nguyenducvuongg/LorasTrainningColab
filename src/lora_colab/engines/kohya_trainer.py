import os
import subprocess
import toml
from typing import Dict, Any, List, Optional
from .base import BaseTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class KohyaTrainer(BaseTrainer):
    """
    Kohya_ss (sd-scripts) Trainer Engine.
    Industry standard for SDXL 1.0, Pony Diffusion V6, Illustrious-XL, SD 1.5, and SD 3.5.
    """

    def _determine_script_name(self) -> str:
        fam = self.config.training.model_family.lower()
        if any(k in fam for k in ["sdxl", "pony", "illustrious", "animagine"]):
            return "sdxl_train_network.py"
        elif "sd3" in fam:
            return "sd3_train_network.py"
        elif "flux" in fam:
            return "flux_train_network.py"
        else:
            return "train_network.py"

    def build_command_or_config(self, resume_from: Optional[str] = None) -> List[str]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network

        script_name = self._determine_script_name()
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process=2",
            "--mixed_precision=" + ("bf16" if t_cfg.mixed_precision == "bf16" else "fp16"),
            f"sd-scripts/{script_name}",
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
                f"--min_bucket_reso={d_cfg.min_bucket_res}",
                f"--max_bucket_reso={d_cfg.max_bucket_res}",
            ])

        if t_cfg.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if t_cfg.cache_latents:
            cmd.append("--cache_latents")
            if t_cfg.cache_latents_to_disk:
                cmd.append("--cache_latents_to_disk")

        if t_cfg.cache_latents_to_disk and "sdxl" in script_name:
            cmd.append("--cache_text_encoder_outputs")
            cmd.append("--cache_text_encoder_outputs_to_disk")

        # Optimizer Configuration
        opt_type = t_cfg.optimizer_type.lower()
        if "prodigy" in opt_type:
            cmd.extend([
                "--optimizer_type=Prodigy",
                "--optimizer_args", "d_coef=1.0", "weight_decay=0.01", "decouple=True", "use_bias_correction=True"
            ])
        elif "8bit" in opt_type:
            cmd.extend(["--optimizer_type=AdamW8bit"])
        else:
            cmd.extend(["--optimizer_type=AdamW"])

        # Auto-Resume
        if resume_from and os.path.exists(resume_from):
            cmd.append(f"--resume={resume_from}")

        # Text encoder training
        if t_cfg.text_encoder_lr and not t_cfg.cache_latents_to_disk:
            cmd.append(f"--text_encoder_lr={t_cfg.text_encoder_lr}")

        return cmd

    def train(self, resume_from: Optional[str] = None) -> bool:
        cmd = self.build_command_or_config(resume_from=resume_from)
        
        console.print(f"[bold green]🚀 Launching Kohya Trainer for {self.config.training.model_family}...[/bold green]")
        console.print(f"  • Base Model: [cyan]{self.config.training.base_model_path}[/cyan]")
        console.print(f"  • Output Directory (Directly to Drive): [yellow]{self.config.training.checkpoint_dir}[/yellow]")
        console.print(f"  • Command: [dim]{' '.join(cmd)}[/dim]")

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error executing Kohya trainer: {e}")
            return False
