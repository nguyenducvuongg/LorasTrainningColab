import os
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .logger import setup_logger, console

logger = setup_logger(__name__)

@dataclass
class GPUProfile:
    device_name: str
    vram_gb: float
    compute_capability: tuple
    recommended_precision: str
    recommended_batch_size: int
    recommended_grad_accum: int
    enable_fp8: bool
    enable_cpu_offload: bool
    enable_flash_attention: bool
    hardware_tier: str  # "T4_FREE" | "L4_PRO" | "A100_PRO" | "CONSUMER" | "CPU"

class HardwareProfiler:
    """Tự động phân tích GPU và cấu hình tham số VRAM tối ưu để tránh OOM 100%."""

    @classmethod
    def analyze(cls, model_family: str = "flux-dev") -> GPUProfile:
        try:
            import torch
            if not torch.cuda.is_available():
                return cls._fallback_cpu_profile()

            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 1)
            cc = torch.cuda.get_device_capability(0)
        except Exception as e:
            return cls._fallback_cpu_profile()

        name_lower = device_name.lower()
        is_flux = "flux" in model_family.lower()

        # Phân tầng phần cứng
        if "t4" in name_lower or vram_gb <= 16.0:
            tier = "T4_FREE"
            precision = "fp16"
            fp8 = is_flux
            batch_size = 1
            grad_accum = 2 if is_flux else 1
            cpu_offload = is_flux
            flash_attn = False
        elif "l4" in name_lower or (16.0 < vram_gb <= 24.0):
            tier = "L4_PRO"
            precision = "bf16" if cc >= (8, 0) else "fp16"
            fp8 = is_flux
            batch_size = 2 if not is_flux else 1
            grad_accum = 2
            cpu_offload = False
            flash_attn = True
        elif "a100" in name_lower or "h100" in name_lower or vram_gb > 24.0:
            tier = "A100_PRO"
            precision = "bf16"
            fp8 = False
            batch_size = 4 if not is_flux else 2
            grad_accum = 1
            cpu_offload = False
            flash_attn = True
        else:
            tier = "CONSUMER"
            precision = "fp16"
            fp8 = is_flux and vram_gb < 20.0
            batch_size = 1
            grad_accum = 2
            cpu_offload = vram_gb < 16.0
            flash_attn = cc >= (8, 0)

        profile = GPUProfile(
            device_name=device_name,
            vram_gb=vram_gb,
            compute_capability=cc,
            recommended_precision=precision,
            recommended_batch_size=batch_size,
            recommended_grad_accum=grad_accum,
            enable_fp8=fp8,
            enable_cpu_offload=cpu_offload,
            enable_flash_attention=flash_attn,
            hardware_tier=tier
        )

        return profile

    @classmethod
    def _fallback_cpu_profile(cls) -> GPUProfile:
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            ram_gb = 16.0

        return GPUProfile(
            device_name=f"CPU Mode ({ram_gb} GB RAM)",
            vram_gb=0.0,
            compute_capability=(0, 0),
            recommended_precision="fp32",
            recommended_batch_size=1,
            recommended_grad_accum=1,
            enable_fp8=False,
            enable_cpu_offload=True,
            enable_flash_attention=False,
            hardware_tier="CPU"
        )
