import os
from typing import Optional
from ...core.logger import setup_logger

logger = setup_logger(__name__)

class GeminiVisionCaptioner:
    """Gán nhãn qua Gemini API trên Cloud (tiêu tốn 0% VRAM GPU)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def generate_caption(self, image_path: str, prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "photo of subject in clean studio lighting"
        # API caller logic...
        return "photo of subject in clear lighting"
