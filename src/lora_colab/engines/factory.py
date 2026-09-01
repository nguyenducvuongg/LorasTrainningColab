from typing import Optional, Dict, Type
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
    """

    FAMILY_ENGINE_MAP: Dict[str, Type[BaseTrainer]] = {
        # Flux & Krea -> AI-Toolkit (Ostris)
        "flux": AIToolkitTrainer,
        "flux-dev": AIToolkitTrainer,
        "flux-schnell": AIToolkitTrainer,
        "flux-kontext": AIToolkitTrainer,
        "krea": AIToolkitTrainer,
        "krea2-raw": AIToolkitTrainer,
        "chroma": AIToolkitTrainer,

        # SDXL, Pony, Illustrious, SD 1.5, SD 3.5 -> Kohya_ss (sd-scripts)
        "sdxl": KohyaTrainer,
        "sdxl-base": KohyaTrainer,
        "pony": KohyaTrainer,
        "pony-v6": KohyaTrainer,
        "illustrious": KohyaTrainer,
        "illustrious-xl": KohyaTrainer,
        "sd15": KohyaTrainer,
        "sd15-base": KohyaTrainer,
        "sd35": KohyaTrainer,
        "sd35-medium": KohyaTrainer,

        # Wan, Qwen, Kolors/Z-Image -> Musubi-Tuner / Diffusers
        "qwen": MusubiTrainer,
        "qwen-image": MusubiTrainer,
        "z-image-kolors": MusubiTrainer,
        "wan": MusubiTrainer,
        "video": MusubiTrainer,
    }

    @classmethod
    def resolve_engine_type(cls, model_family: str, explicit_choice: Optional[str] = None) -> Type[BaseTrainer]:
        """Xác định engine phù hợp nhất."""
        if explicit_choice and explicit_choice.lower() != "auto":
            clean_choice = explicit_choice.lower()
            if "ai-toolkit" in clean_choice or "aitoolkit" in clean_choice:
                return AIToolkitTrainer
            elif "kohya" in clean_choice or "sd-scripts" in clean_choice or "sdscripts" in clean_choice:
                return KohyaTrainer
            elif "musubi" in clean_choice:
                return MusubiTrainer
            elif "diffusers" in clean_choice:
                return DiffusersTrainer

        # Auto-detect based on model_family string
        fam = model_family.lower().strip()
        for key, engine_cls in cls.FAMILY_ENGINE_MAP.items():
            if key in fam:
                return engine_cls

        # Fallback to AIToolkit for Flux/Krea, else Kohya
        if "flux" in fam or "krea" in fam:
            return AIToolkitTrainer
        return KohyaTrainer

    @classmethod
    def get_engine_description(cls, engine_cls: Type[BaseTrainer]) -> str:
        if engine_cls == AIToolkitTrainer:
            return "AI-Toolkit (Ostris) - Tối ưu hàng đầu cho Flux.1-dev / schnell / Kontext / Krea2"
        elif engine_cls == KohyaTrainer:
            return "Kohya_ss (sd-scripts) - Chuẩn công nghiệp cho SDXL / Pony / Illustrious / SD1.5 / SD3.5"
        elif engine_cls == MusubiTrainer:
            return "Musubi-Tuner (Kohya Next-Gen) - Tối ưu cho Wan 2.1 / Qwen2-VL / Z-Image"
        elif engine_cls == DiffusersTrainer:
            return "Native Diffusers + PEFT PyTorch Loop"
        return str(engine_cls.__name__)

    @classmethod
    def create_trainer(cls, config: LoRAConfig, engine_choice: Optional[str] = None) -> BaseTrainer:
        """Khởi tạo Trainer với engine tối ưu được chọn tự động."""
        engine_cls = cls.resolve_engine_type(config.training.model_family, explicit_choice=engine_choice)
        desc = cls.get_engine_description(engine_cls)

        console.rule("[bold cyan]🎯 Bộ Điều Phối Engine Huấn Luyện Thông Minh[/bold cyan]")
        console.print(f"  • Model Nền: [bold yellow]{config.training.model_family}[/bold yellow]")
        console.print(f"  • Engine Tối Ưu Tự Động: [bold green]{desc}[/bold green]")
        console.rule()

        return engine_cls(config)
