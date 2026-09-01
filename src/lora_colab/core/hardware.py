import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

@dataclass
class GPUProfile:
    device_name: str
    vram_gb: float
    ram_gb: float
    tier: str  # "T4", "L4", "A100", "V100", "CPU", "UNKNOWN"
    cuda_capability: str
    precision: str  # "fp8", "fp16", "bf16"
    recommended_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optimizer: str
    cache_latents_to_disk: bool
    train_text_encoder: bool
    extra_optimizations: Dict[str, Any] = field(default_factory=dict)

class HardwareProfiler:
    """Auto-detects GPU and system resources, applying dynamic VRAM/RAM optimizations."""
    
    @staticmethod
    def get_system_ram_gb() -> float:
        """Get total system RAM in GB."""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            try:
                # Linux fallback
                with open('/proc/meminfo') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            return round(int(line.split()[1]) / (1024 ** 2), 2)
            except Exception:
                return 12.0  # Default estimate for Colab free tier

    @classmethod
    def detect_and_profile(cls, target_model: str = "flux") -> GPUProfile:
        """
        Detects hardware and returns an auto-tuned profile tailored for the target model.
        Target model can be: 'flux', 'flux-kontext', 'krea', 'sdxl', 'pony', 'sd35', 'sd15', 'qwen'
        """
        ram_gb = cls.get_system_ram_gb()
        
        if torch is None or not torch.cuda.is_available():
            logger.warning("[bold yellow]No CUDA GPU detected (or torch not installed)! Falling back to CPU mode.[/bold yellow]")
            return GPUProfile(
                device_name="CPU",
                vram_gb=0.0,
                ram_gb=ram_gb,
                tier="CPU",
                cuda_capability="N/A",
                precision="fp32",
                recommended_batch_size=1,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                optimizer="AdamW",
                cache_latents_to_disk=True,
                train_text_encoder=False,
                extra_optimizations={"cpu_mode": True}
            )

        device_name = torch.cuda.get_device_name(0)
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(total_vram_bytes / (1024 ** 3), 2)
        major, minor = torch.cuda.get_device_capability(0)
        cuda_capability = f"{major}.{minor}"
        
        name_lower = device_name.lower()
        target_lower = target_model.lower()
        is_flux_or_heavy = any(k in target_lower for k in ["flux", "krea", "sd3", "qwen"])

        # Determine Tier
        if "a100" in name_lower or "h100" in name_lower or vram_gb >= 38.0:
            tier = "A100"
            precision = "bf16"
            batch_size = 4 if is_flux_or_heavy else 8
            grad_accum = 1
            grad_checkpoint = False if not is_flux_or_heavy else True
            optimizer = "adamw"
            cache_latents = True
            train_te = True if not is_flux_or_heavy else False
            extra_opts = {
                "quantization": None,
                "attention_mechanism": "flash_attention_2" if major >= 8 else "sdpa",
                "fused_backward_pass": True,
                "offload_cpu": False
            }

        elif "l4" in name_lower or (vram_gb >= 20.0 and vram_gb < 38.0):
            tier = "L4"
            precision = "bf16"
            batch_size = 2 if is_flux_or_heavy else 4
            grad_accum = 2 if is_flux_or_heavy else 1
            grad_checkpoint = True
            optimizer = "prodigy"  # Adaptive LR
            cache_latents = True
            train_te = False if is_flux_or_heavy else True
            extra_opts = {
                "quantization": "fp8" if is_flux_or_heavy else None,
                "attention_mechanism": "sdpa",
                "fused_backward_pass": True,
                "offload_cpu": False
            }

        elif "v100" in name_lower:
            tier = "V100"
            precision = "fp16"
            batch_size = 1 if is_flux_or_heavy else 2
            grad_accum = 4 if is_flux_or_heavy else 2
            grad_checkpoint = True
            optimizer = "adamw8bit"
            cache_latents = True
            train_te = False
            extra_opts = {
                "quantization": "fp8" if is_flux_or_heavy else None,
                "attention_mechanism": "xformers",
                "fused_backward_pass": False,
                "offload_cpu": True if is_flux_or_heavy else False
            }

        elif "t4" in name_lower or vram_gb <= 16.5:
            tier = "T4"
            precision = "fp16"  # T4 has slow native BF16, use FP16/FP8
            batch_size = 1
            grad_accum = 4 if is_flux_or_heavy else 2
            grad_checkpoint = True
            optimizer = "adamw8bit"
            cache_latents = True
            train_te = False
            extra_opts = {
                "quantization": "fp8" if is_flux_or_heavy else None,
                "t5_quantization": "4bit" if is_flux_or_heavy else None,
                "attention_mechanism": "sdpa",
                "fused_backward_pass": True,
                "offload_cpu": True if is_flux_or_heavy else False,
                "low_vram_mode": True
            }

        else:
            tier = "GENERIC_CUDA"
            precision = "fp16" if major < 8 else "bf16"
            batch_size = 2 if vram_gb >= 20.0 else 1
            grad_accum = 2
            grad_checkpoint = True
            optimizer = "adamw8bit"
            cache_latents = True
            train_te = False
            extra_opts = {
                "quantization": "fp8" if (is_flux_or_heavy and vram_gb < 24.0) else None,
                "attention_mechanism": "sdpa",
                "fused_backward_pass": True,
                "offload_cpu": False
            }

        profile = GPUProfile(
            device_name=device_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            tier=tier,
            cuda_capability=cuda_capability,
            precision=precision,
            recommended_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=grad_checkpoint,
            optimizer=optimizer,
            cache_latents_to_disk=cache_latents,
            train_text_encoder=train_te,
            extra_optimizations=extra_opts
        )
        
        return profile

    @classmethod
    def display_profile(cls, profile: GPUProfile):
        """Displays formatted hardware information and auto-tuned settings."""
        console.rule("[bold cyan]Colab Hardware & VRAM Auto-Profiler[/bold cyan]")
        console.print(f"[bold green]GPU Detected:[/bold green] {profile.device_name} ({profile.vram_gb} GB VRAM)")
        console.print(f"[bold green]System RAM:[/bold green] {profile.ram_gb} GB | [bold green]Compute Capability:[/bold green] {profile.cuda_capability}")
        console.print(f"[bold magenta]Profile Tier:[/bold magenta] [bold yellow]{profile.tier}[/bold yellow]")
        console.print(f"  • [cyan]Mixed Precision:[/cyan] {profile.precision}")
        console.print(f"  • [cyan]Batch Size:[/cyan] {profile.recommended_batch_size} (Grad Accum: {profile.gradient_accumulation_steps})")
        console.print(f"  • [cyan]Optimizer:[/cyan] {profile.optimizer}")
        console.print(f"  • [cyan]Gradient Checkpointing:[/cyan] {profile.gradient_checkpointing}")
        console.print(f"  • [cyan]Latent Disk Cache:[/cyan] {profile.cache_latents_to_disk}")
        console.print(f"  • [cyan]Train Text Encoder:[/cyan] {profile.train_text_encoder}")
        for k, v in profile.extra_optimizations.items():
            console.print(f"  • [cyan]{k}:[/cyan] {v}")
        console.rule()
