import os
import sys
import yaml
import subprocess
from typing import Dict, Any, Optional, List
from .base import BaseTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class AIToolkitTrainer(BaseTrainer):
    """
    AI-Toolkit (Ostris) Trainer Engine.
    Optimized for Flux.1-dev, Flux.1-schnell, Flux-Kontext, and Krea2-raw LoRA training.
    """

    def build_command_or_config(self) -> Dict[str, Any]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network

        is_flux = "flux" in t_cfg.model_family.lower()
        quantize_base = t_cfg.mixed_precision == "fp8" or "8bit" in t_cfg.optimizer_type.lower()

        ai_toolkit_yaml = {
            "job": "extension",
            "config": {
                "name": t_cfg.output_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": t_cfg.checkpoint_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "lora",
                            "linear": n_cfg.network_dim,
                            "linear_alpha": n_cfg.network_alpha,
                        },
                        "save": {
                            "dtype": "float16" if t_cfg.mixed_precision == "fp16" else "bfloat16",
                            "save_every": t_cfg.save_every_n_steps or 250,
                            "max_step_saves_to_keep": 4,
                        },
                        "datasets": [
                            {
                                "folder_path": d_cfg.dataset_dir,
                                "caption_ext": d_cfg.caption_extension.replace(".", ""),
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": d_cfg.shuffle_caption,
                                "cache_latents_to_disk": t_cfg.cache_latents_to_disk,
                                "resolution": [512, 768, 1024] if d_cfg.enable_bucketing else [d_cfg.resolution, d_cfg.resolution],
                            }
                        ],
                        "train": {
                            "batch_size": t_cfg.batch_size,
                            "steps": t_cfg.max_train_steps or (t_cfg.epochs * 200),
                            "gradient_accumulation_steps": t_cfg.gradient_accumulation_steps,
                            "train_unet": True,
                            "train_text_encoder": False,  # Keeps VRAM under control on T4/L4
                            "gradient_checkpointing": t_cfg.gradient_checkpointing,
                            "noise_scheduler": "flowmatch" if is_flux else "ddim",
                            "optimizer": "adamw8bit" if "8bit" in t_cfg.optimizer_type.lower() else "adamw",
                            "lr": t_cfg.learning_rate,
                            "ema_config": {
                                "use_ema": True,
                                "ema_decay": 0.99
                            },
                            "dtype": "bfloat16" if t_cfg.mixed_precision == "bf16" else "float16",
                        },
                        "model": {
                            "name_or_path": t_cfg.base_model_path,
                            "is_flux": is_flux,
                            "quantize": quantize_base,
                        },
                        "sample": {
                            "sampler": "flowmatch" if is_flux else "euler",
                            "sample_every": t_cfg.sample_every_n_steps,
                            "width": d_cfg.resolution,
                            "height": d_cfg.resolution,
                            "prompts": [
                                t_cfg.sample_prompt or "a portrait photo of sks person in high detail"
                            ],
                            "neg": "ugly, low quality, deformed, blurry",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 3.5 if is_flux else 7.0,
                            "sample_steps": 25 if is_flux else 28,
                        }
                    }
                ]
            }
        }

        return ai_toolkit_yaml

    def _resolve_run_cmd(self, config_path: str) -> List[str]:
        possible_paths = [
            "/content/backends/ai-toolkit/run.py",
            "/content/ai-toolkit/run.py",
            os.path.join(os.getcwd(), "ai-toolkit", "run.py"),
            "ai-toolkit/run.py"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return [sys.executable, p, config_path]
        return [sys.executable, "-m", "ai_toolkit.run", config_path]

    def train(self, resume_from: Optional[str] = None) -> bool:
        config_dict = self.build_command_or_config()
        temp_config_path = os.path.join(self.config.training.checkpoint_dir, "ai_toolkit_active_config.yaml")
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)

        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        console.print(f"[bold green]🚀 Launching AI-Toolkit Trainer for {self.config.training.model_family}...[/bold green]")
        console.print(f"  • Config: [cyan]{temp_config_path}[/cyan]")
        console.print(f"  • Output Checkpoints directly to: [yellow]{self.config.training.checkpoint_dir}[/yellow]")

        # Run AI-Toolkit process with auto-resolved runner
        cmd = self._resolve_run_cmd(temp_config_path)
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            return process.returncode == 0
        except Exception as e:
            logger.error(f"Error executing AI-Toolkit trainer: {e}")
            return False
