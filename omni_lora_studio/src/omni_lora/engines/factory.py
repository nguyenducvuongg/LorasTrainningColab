from typing import Type, Optional, Union
from .base import BaseTrainer
from .kohya_flux import KohyaFluxTrainer
from .kohya_sdxl import KohyaSDXLTrainer
from .kohya_sd15 import KohyaSD15Trainer
from .aitoolkit_trainer import AIToolkitTrainer
from .musubi_trainer import MusubiTrainer
from ..core.config import OmniConfig, ModelFamily
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class EngineFactory:
    """
    Bộ Điều Phối Engine Huấn Luyện Toàn Diện (Universal Engine Dispatcher).
    Hỗ trợ: FLUX, SDXL, Pony, Illustrious, SD 3.5, SD 1.5, Krea2, Z-Image (Kolors), Wan 2.1.
    """

    @classmethod
    def resolve_engine(cls, model_family: Union[str, ModelFamily]) -> Type[BaseTrainer]:
        fam = str(model_family).lower().strip()

        # Krea2 -> AI-Toolkit chuyên sâu
        if "krea" in fam:
            return AIToolkitTrainer
        # Z-Image / Kolors / Wan 2.1 Video / DiT Next-Gen -> Musubi-Tuner
        elif "z-image" in fam or "kolors" in fam or "wan" in fam:
            return MusubiTrainer
        # Flux (dev / schnell / kontext) -> KohyaFluxTrainer (hoặc AIToolkit)
        elif "flux" in fam:
            return KohyaFluxTrainer
        # SD 1.5
        elif "sd15" in fam or "sd1.5" in fam or "v1-5" in fam:
            return KohyaSD15Trainer
        # SDXL, Pony V6, Illustrious-XL, SD 3.5 -> KohyaSDXLTrainer
        elif any(k in fam for k in ["sdxl", "pony", "illustrious", "sd35", "sd3.5"]):
            return KohyaSDXLTrainer
        # Fallback mặc định cho SDXL
        return KohyaSDXLTrainer

    @classmethod
    def create_trainer(cls, config: OmniConfig) -> BaseTrainer:
        engine_cls = cls.resolve_engine(config.training.model_family)
        console.rule("[bold magenta]🎯 Universal Engine Factory[/bold magenta]")
        console.print(f"  • Model Được Chọn: [bold yellow]{config.training.model_family}[/bold yellow]")
        console.print(f"  • Engine Lõi Điều Phối: [bold green]{engine_cls.__name__}[/bold green]")
        console.rule()
        return engine_cls(config)
