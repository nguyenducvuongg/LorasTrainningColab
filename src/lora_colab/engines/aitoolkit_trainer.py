import os
import sys
import yaml
import subprocess
import importlib.metadata
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console
from ..core.environment import AutoEnvironmentManager
from ..monitoring.dashboard import LiveTrainingDashboard

logger = setup_logger(__name__)

class AIToolkitTrainer(BaseTrainer):
    """
    AI-Toolkit (Ostris) Trainer Engine.
    Học hỏi và tối ưu hóa theo kiến trúc chuẩn SDVN & Ostris cho Flux.1 (Dev, Schnell, Kontext, Krea2-raw, Chroma) và SDXL.
    """

    BACKEND_REPO_URL = "https://github.com/ostris/ai-toolkit.git"
    DEFAULT_BACKEND_DIR = "/content/backends/ai-toolkit"

    @classmethod
    def _patch_ai_toolkit_bugs(cls):
        """
        Tự động vá lỗi thiếu hasattr trên CLIPTextEncoder và lỗi prompt_embeds.hidden_states NoneType
        trong mã nguồn AI-Toolkit (toolkit/stable_diffusion_model.py & toolkit/train_tools.py).
        """
        # 1. Vá lỗi stable_diffusion_model.py
        sd_model_file = os.path.join(cls.DEFAULT_BACKEND_DIR, "toolkit", "stable_diffusion_model.py")
        if os.path.exists(sd_model_file):
            try:
                with open(sd_model_file, "r", encoding="utf-8") as f:
                    content = f.read()

                target_bug = "te_has_grad = self.text_encoder.text_model.final_layer_norm.weight.requires_grad"
                if target_bug in content:
                    fixed_code = """try:
                    if hasattr(self.text_encoder, 'text_model') and hasattr(self.text_encoder.text_model, 'final_layer_norm'):
                        te_has_grad = self.text_encoder.text_model.final_layer_norm.weight.requires_grad
                    elif hasattr(self.text_encoder, 'final_layer_norm'):
                        te_has_grad = self.text_encoder.final_layer_norm.weight.requires_grad
                    else:
                        te_has_grad = any(p.requires_grad for p in self.text_encoder.parameters())
                except Exception:
                    te_has_grad = False"""
                    content = content.replace(target_bug, fixed_code)
                    with open(sd_model_file, "w", encoding="utf-8") as f:
                        f.write(content)
            except Exception as e:
                logger.warning(f"Could not patch stable_diffusion_model.py: {e}")

        # 2. Vá lỗi train_tools.py (TypeError: 'NoneType' object is not subscriptable khi hidden_states is None)
        train_tools_file = os.path.join(cls.DEFAULT_BACKEND_DIR, "toolkit", "train_tools.py")
        if os.path.exists(train_tools_file):
            try:
                with open(train_tools_file, "r", encoding="utf-8") as f:
                    content = f.read()

                old_code_1 = "prompt_embeds = prompt_embeds.hidden_states[-2]  # always penultimate layer"
                new_code_1 = """if hasattr(prompt_embeds, 'hidden_states') and prompt_embeds.hidden_states is not None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        elif isinstance(prompt_embeds, (tuple, list)) and len(prompt_embeds) > 2 and prompt_embeds[2] is not None:
            prompt_embeds = prompt_embeds[2][-2]
        elif hasattr(prompt_embeds, 'last_hidden_state') and prompt_embeds.last_hidden_state is not None:
            prompt_embeds = prompt_embeds.last_hidden_state
        else:
            prompt_embeds = prompt_embeds[0]"""

                old_code_2 = "prompt_embed = embeds.hidden_states[-2]  # always penultimate layer"
                new_code_2 = """if hasattr(embeds, 'hidden_states') and embeds.hidden_states is not None:
                prompt_embed = embeds.hidden_states[-2]
            elif isinstance(embeds, (tuple, list)) and len(embeds) > 2 and embeds[2] is not None:
                prompt_embed = embeds[2][-2]
            elif hasattr(embeds, 'last_hidden_state') and embeds.last_hidden_state is not None:
                prompt_embed = embeds.last_hidden_state
            else:
                prompt_embed = embeds[0]"""

                if old_code_1 in content:
                    content = content.replace(old_code_1, new_code_1)
                elif "prompt_embeds = prompt_embeds.hidden_states[-2]" in content:
                    content = content.replace("prompt_embeds = prompt_embeds.hidden_states[-2]", new_code_1)

                if old_code_2 in content:
                    content = content.replace(old_code_2, new_code_2)
                elif "prompt_embed = embeds.hidden_states[-2]" in content:
                    content = content.replace("prompt_embed = embeds.hidden_states[-2]", new_code_2)

                with open(train_tools_file, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Could not patch train_tools.py: {e}")

    @classmethod
    def _ensure_backend_ready(cls):
        """
        Đảm bảo kho mã nguồn Ostris AI-Toolkit đã được clone chuẩn xác (--recurse-submodules),
        vá lỗi nội bộ và nạp đầy đủ các phụ thuộc cần thiết chuẩn theo SDVN & Ostris.
        """
        # 1. Gỡ bỏ gói PyPI trùng tên nếu đã lỡ cài nhầm
        if AutoEnvironmentManager.is_package_installed("dataslots") or AutoEnvironmentManager.is_package_installed("ai_toolkit"):
            console.print("[yellow]🧹 Đang tự động gỡ bỏ thư viện PyPI 'ai-toolkit' trùng tên không liên quan...[/yellow]")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "ai-toolkit", "ai_toolkit", "dataslots", "wget"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                pass

        # 2. Clone kho mã nguồn chính thức của Ostris nếu chưa có (với --recurse-submodules)
        if os.path.exists("/content") and not os.path.exists(cls.DEFAULT_BACKEND_DIR):
            console.print("[bold cyan]📥 Tải và cấu hình backend chính thức [bold]Ostris/AI-Toolkit[/bold]...[/bold cyan]")
            try:
                os.makedirs(os.path.dirname(cls.DEFAULT_BACKEND_DIR), exist_ok=True)
                subprocess.check_call([
                    "git", "clone", "--depth", "1", "--recurse-submodules",
                    cls.BACKEND_REPO_URL, cls.DEFAULT_BACKEND_DIR
                ])
                console.print("[bold green]✓ AI-Toolkit (Ostris) đã sẵn sàng![/bold green]")
            except Exception as e:
                logger.warning(f"Could not clone ai-toolkit: {e}")

        # 3. Tự động áp dụng bản vá lỗi mã nguồn nội bộ
        cls._patch_ai_toolkit_bugs()

        # 4. Quét và cài đặt trọn gói các phụ thuộc thiết yếu cho AI-Toolkit
        ai_toolkit_packages = [
            "oyaml>=1.0",
            "albumentations>=1.4.0",
            "flatten_dict>=0.4.0",
            "k-diffusion>=0.1.0",
            "open-clip-torch>=2.24.0",
            "invisible-watermark>=0.2.0",
            "clean-fid>=0.1.35",
            "toml>=0.10.2",
            "bitsandbytes>=0.43.0",
            "optimum-quanto>=0.2.0",
            "tensorboard>=2.14.0",
            "python-dotenv>=1.0.0",
        ]
        missing = [
            pkg for pkg in ai_toolkit_packages
            if not AutoEnvironmentManager.is_package_installed(pkg.split(">=")[0].strip())
        ]
        if missing:
            console.print(f"[bold yellow]📦 Tự động cài đặt siêu tốc phụ thuộc cho AI-Toolkit ({len(missing)} gói):[/bold yellow] [dim]{', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}[/dim]")
            AutoEnvironmentManager.install_packages(missing, silent=True)
            console.print("[bold green]✓ Toàn bộ gói phụ trợ AI-Toolkit đã nạp sẵn sàng 100%![/bold green]")

    def _detect_model_arch(self) -> Tuple[bool, bool, bool]:
        """
        Nhận diện chính xác kiến trúc mô hình từ model_family (không dùng base_model_path để tránh false match).
        Returns: (is_flux, is_xl, is_sd15)
        """
        fam = self.config.training.model_family.lower()
        is_flux = any(k in fam for k in ["flux", "kontext", "schnell", "chroma"])
        # Krea là SDXL-based
        is_xl = any(k in fam for k in ["sdxl", "pony", "illustrious", "krea"])
        is_sd15 = any(k in fam for k in ["sd15", "sd1.5", "stable-diffusion-v1"])
        return is_flux, is_xl, is_sd15

    def _calc_total_steps(self) -> int:
        """Tính tổng số steps dựa trên epochs, dataset size và repeats."""
        t_cfg = self.config.training
        d_cfg = self.config.dataset
        if t_cfg.max_train_steps:
            return t_cfg.max_train_steps
        # Ước tính: nếu không biết dataset size, dùng giá trị mặc định an toàn
        # Dataset size thực tế sẽ được engine tự xử lý
        approx_images = 20  # ước tính tối thiểu
        try:
            if os.path.exists(d_cfg.dataset_dir):
                img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                approx_images = max(1, len([
                    f for f in os.listdir(d_cfg.dataset_dir)
                    if os.path.splitext(f)[-1].lower() in img_exts
                ]))
        except Exception:
            pass
        steps = (approx_images * d_cfg.repeats * t_cfg.epochs) // max(1, t_cfg.batch_size)
        return max(steps, 100)

    def build_command_or_config(self) -> Dict[str, Any]:
        cfg = self.config
        t_cfg = cfg.training
        d_cfg = cfg.dataset
        n_cfg = cfg.network

        is_flux, is_xl, is_sd15 = self._detect_model_arch()
        quantize_base = is_flux or is_xl or t_cfg.mixed_precision == "fp8" or "8bit" in t_cfg.optimizer_type.lower()

        total_steps = self._calc_total_steps()

        # Optimizer mapping: Prodigy → prodigy, AdamW8bit → adamw8bit, AdamW → adamw
        opt_lower = t_cfg.optimizer_type.lower()
        if "prodigy" in opt_lower:
            ai_optimizer = "prodigy"
        elif "adamw8bit" in opt_lower or "8bit" in opt_lower:
            ai_optimizer = "adamw8bit"
        elif "adafactor" in opt_lower:
            ai_optimizer = "adafactor"
        else:
            ai_optimizer = "adamw"

        process_config: Dict[str, Any] = {
            "type": "diffusion_trainer",
            "training_folder": t_cfg.checkpoint_dir,
            "device": "cuda:0",
            "performance_log_every": 10,
            "network": {
                "type": "lora",
                "linear": n_cfg.network_dim,
                "linear_alpha": n_cfg.network_alpha,
            },
            "save": {
                "dtype": "bfloat16" if t_cfg.mixed_precision == "bf16" else "float16",
                "save_every": t_cfg.save_every_n_steps or 250,
                "max_step_saves_to_keep": 4,
                "save_format": "safetensors",   # FIX: đổi từ "diffusers" → "safetensors"
                "push_to_hub": False,
            },
            "datasets": [
                {
                    "folder_path": d_cfg.dataset_dir,
                    "caption_ext": d_cfg.caption_extension.replace(".", ""),
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": d_cfg.shuffle_caption,
                    "cache_latents_to_disk": t_cfg.cache_latents_to_disk,
                    "resolution": [512, 768, 1024] if d_cfg.enable_bucketing else [d_cfg.resolution],
                    "num_repeats": d_cfg.repeats,
                }
            ],
            "train": {
                "batch_size": t_cfg.batch_size,
                "steps": total_steps,
                "gradient_accumulation": t_cfg.gradient_accumulation_steps,
                "gradient_accumulation_steps": t_cfg.gradient_accumulation_steps,
                "train_unet": True,
                "train_text_encoder": False,  # Keeps VRAM stable on T4/L4
                "gradient_checkpointing": t_cfg.gradient_checkpointing,
                "noise_scheduler": "flowmatch" if is_flux else "ddim",
                "optimizer": ai_optimizer,
                "timestep_type": "shift" if is_flux else "linear",
                "content_or_style": "balanced",
                "lr": t_cfg.learning_rate,
                "lr_scheduler": t_cfg.lr_scheduler or "constant",
                "ema_config": {
                    "use_ema": True,
                    "ema_decay": 0.99,
                },
                "dtype": "bfloat16" if t_cfg.mixed_precision == "bf16" else "float16",
                "loss_type": "mse",
            },
            "model": {
                "name_or_path": t_cfg.base_model_path,
                "arch": "flux" if is_flux else ("sdxl" if is_xl else "sd"),
                "is_flux": is_flux,
                "is_xl": is_xl,
                "quantize": quantize_base,
                "qtype": "qfloat8" if is_flux else None,
                "quantize_te": quantize_base if is_flux else False,
                "qtype_te": "qfloat8" if is_flux else None,
                "low_vram": True if (is_flux or is_xl) else False,
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
                "sample_steps": 20 if is_flux else 28,
            }
        }

        # Filter out None values for clean YAML serialization
        def clean_none(d):
            if isinstance(d, dict):
                return {k: clean_none(v) for k, v in d.items() if v is not None}
            return d

        ai_toolkit_yaml = {
            "job": "extension",
            "config": {
                "name": t_cfg.output_name,
                "process": [clean_none(process_config)]
            }
        }

        return ai_toolkit_yaml

    def _resolve_run_cmd(self, config_path: str) -> Tuple[List[str], Optional[str]]:
        possible_paths = [
            "/content/backends/ai-toolkit/run.py",
            os.path.join(os.getcwd(), "backends", "ai-toolkit", "run.py"),
            "/content/ai-toolkit/run.py",
            os.path.join(os.getcwd(), "ai-toolkit", "run.py"),
            "ai-toolkit/run.py"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return [sys.executable, p, config_path], os.path.dirname(os.path.abspath(p))

        # Nếu chưa tìm thấy, gọi đảm bảo backend và trả về file run.py chính xác
        self._ensure_backend_ready()
        target_run = os.path.join(self.DEFAULT_BACKEND_DIR, "run.py")
        return [sys.executable, target_run, config_path], self.DEFAULT_BACKEND_DIR

    def train(self, resume_from: Optional[str] = None) -> bool:
        is_flux, is_xl, is_sd15 = self._detect_model_arch()
        total_steps = self._calc_total_steps()
        arch_label = "Flux.1" if is_flux else ("SDXL/Krea" if is_xl else "SD 1.5")

        # 1. Khởi tạo Dashboard NGAY LẬP TỨC — hiện thị setup progress trước khi clone/install
        dashboard = LiveTrainingDashboard(
            model_name=self.config.training.model_family,
            engine_name=f"AI-Toolkit (Ostris) [{arch_label}]",
            total_steps=total_steps,
            total_epochs=self.config.training.epochs,
            output_dir=self.config.training.checkpoint_dir
        )
        dashboard.set_status("⚙️ Đang chuẩn bị backend AI-Toolkit... (có thể mất 5-15 phút lần đầu)")
        dashboard.render()

        # 2. Clone backend + cài packages (user thấy dashboard trong khi chờ)
        self._ensure_backend_ready()
        dashboard.set_status("✅ Backend sẵn sàng! Đang khởi chạy training...")
        dashboard.render()

        # 3. Sinh config YAML
        config_dict = self.build_command_or_config()
        temp_config_path = os.path.join(self.config.training.checkpoint_dir, "ai_toolkit_active_config.yaml")
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        console.print(f"[bold green]🚀 Khởi chạy AI-Toolkit Trainer ({arch_label}) cho {self.config.training.model_family}...[/bold green]")
        console.print(f"  • Cấu hình: [cyan]{temp_config_path}[/cyan]")
        console.print(f"  • Lưu checkpoint trực tiếp tại: [yellow]{self.config.training.checkpoint_dir}[/yellow]")
        console.print(f"  • Optimizer: [green]{self.config.training.optimizer_type}[/green] | LR: [green]{self.config.training.learning_rate}[/green]")

        cmd, ai_dir = self._resolve_run_cmd(temp_config_path)
        env = os.environ.copy()
        if ai_dir and os.path.exists(ai_dir):
            env["PYTHONPATH"] = f"{ai_dir}:{env.get('PYTHONPATH', '')}"

        # 4. Bắt đầu training — dashboard cập nhật liên tục qua parse_log_line
        success = AutoEnvironmentManager.execute_with_self_healing(
            cmd,
            env=env,
            on_log_line=dashboard.parse_log_line
        )

        dashboard.close(success=success)
        return success
