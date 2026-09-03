from typing import Optional, Dict, Type, List
from .base import BaseTrainer
from .aitoolkit_trainer import AIToolkitTrainer
from .kohya_trainer import KohyaTrainer
from .musubi_trainer import MusubiTrainer
from .diffusers_trainer import DiffusersTrainer
from ..core.config import LoRAConfig
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class EngineFactory:
    """
    Intelligent Trainer Engine Selector & Dispatcher.
    Tự động phân tích Model Nền Tảng (Base Model) và gán Engine huấn luyện tối ưu nhất.
    Dùng priority list để tránh false match (ví dụ: "sd" in "sdxl" = True).
    """

    # Priority-ordered list: specific → general. Kiểm tra theo thứ tự từ trên xuống.
    PRIORITY_ENGINE_MAP: List[tuple] = [
        # Flux family (.safetensors) → KohyaTrainer (flux_train_network.py)
        (["flux-kontext", "flux-schnell", "flux-dev", "flux", "chroma"], KohyaTrainer),
        # Krea → AI-Toolkit (SDXL-based nhưng ai-toolkit hỗ trợ)
        (["krea2-raw", "krea"], AIToolkitTrainer),
        # SDXL variants (match specific trước để tránh "sd" match nhầm)
        (["sdxl-base", "sdxl", "pony-v6", "pony", "illustrious-xl", "illustrious"], KohyaTrainer),
        # SD 3.x
        (["sd35-medium", "sd35", "sd3.5", "sd3"], KohyaTrainer),
        # SD 1.5 (match sau SDXL để tránh "sd" nhầm)
        (["sd15-base", "sd15", "sd1.5", "stable-diffusion-v1"], KohyaTrainer),
        # Wan, Qwen, Kolors → Musubi
        (["qwen-image", "qwen2-vl", "qwen"], MusubiTrainer),
        (["z-image-kolors", "kolors"], MusubiTrainer),
        (["wan2", "wan"], MusubiTrainer),
        (["video"], MusubiTrainer),
    ]

    @classmethod
    def resolve_engine_type(cls, model_family: str, explicit_choice: Optional[str] = None) -> Type[BaseTrainer]:
        """Xác định engine phù hợp nhất theo priority list."""
        if explicit_choice and "auto" not in explicit_choice.lower():
            clean_choice = explicit_choice.lower()
            if "ai-toolkit" in clean_choice or "aitoolkit" in clean_choice or "ostris" in clean_choice:
                return AIToolkitTrainer
            elif "kohya" in clean_choice or "sd-scripts" in clean_choice or "sdscripts" in clean_choice:
                return KohyaTrainer
            elif "musubi" in clean_choice:
                return MusubiTrainer
            elif "diffusers" in clean_choice:
                return DiffusersTrainer

        # Auto-detect dựa theo model_family — dùng priority list
        fam = model_family.lower().strip()
        for keywords, engine_cls in cls.PRIORITY_ENGINE_MAP:
            for kw in keywords:
                if kw in fam:
                    return engine_cls

        # Fallback: Flux/Krea → AIToolkit, còn lại → Kohya
        if "flux" in fam or "krea" in fam or "chroma" in fam:
            return AIToolkitTrainer
        return KohyaTrainer

    @classmethod
    def get_engine_description(cls, engine_cls: Type[BaseTrainer]) -> str:
        descriptions = {
            AIToolkitTrainer: "AI-Toolkit (Ostris) — Tối ưu hàng đầu cho Flux.1-dev / schnell / Kontext / Krea2-Raw / Chroma",
            KohyaTrainer: "Kohya_ss (sd-scripts) — Chuẩn công nghiệp cho SDXL / Pony / Illustrious / SD 1.5 / SD 3.5",
            MusubiTrainer: "Musubi-Tuner (Kohya Next-Gen) — Tối ưu cho Wan 2.1 / Qwen2-VL / Z-Image / Kolors",
            DiffusersTrainer: "Native Diffusers + PEFT PyTorch Loop",
        }
        return descriptions.get(engine_cls, str(engine_cls.__name__))

    @classmethod
    def create_trainer(
        cls, 
        config: LoRAConfig, 
        explicit_choice: Optional[str] = None, 
        engine_choice: Optional[str] = None
    ) -> BaseTrainer:
        """Khởi tạo Trainer với engine tối ưu được chọn tự động."""
        choice = explicit_choice or engine_choice
        engine_cls = cls.resolve_engine_type(config.training.model_family, explicit_choice=choice)
        desc = cls.get_engine_description(engine_cls)

        console.rule("[bold cyan]🎯 Bộ Điều Phối Engine Huấn Luyện Thông Minh[/bold cyan]")
        console.print(f"  • Model Nền: [bold yellow]{config.training.model_family}[/bold yellow]")
        console.print(f"  • Engine Tối Ưu Tự Động: [bold green]{desc}[/bold green]")
        console.rule()

        return engine_cls(config)
