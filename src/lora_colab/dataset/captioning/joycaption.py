import os
from typing import Optional, Dict, Any
from PIL import Image
from tqdm import tqdm
from .base import BaseCaptioner
from .florence2 import Florence2Captioner
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

class JoyCaptioner(BaseCaptioner):
    """
    JoyCaption & Florence-2 High-Fidelity Photorealism Captioning Engine.
    Tự động áp dụng bộ tương thích transformers hiện đại và sinh prompt cực kỳ chi tiết cho Flux/SDXL.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Any = None
    ):
        self._florence_engine = Florence2Captioner(
            model_name=model_id,
            task="more_detailed",
            cache_dir=cache_dir,
            device=device,
            torch_dtype=torch_dtype
        )

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        return self._florence_engine.caption_image(image_path, trigger_word=trigger_word)

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        return self._florence_engine.caption_directory(directory, trigger_word=trigger_word, overwrite=overwrite)
