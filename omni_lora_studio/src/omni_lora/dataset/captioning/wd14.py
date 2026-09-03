from typing import List
from ...core.logger import setup_logger

logger = setup_logger(__name__)

class WD14Tagger:
    """Tagger chuyên biệt chuẩn Danbooru/Tag-based cho Anime, Pony V6 & Illustrious."""

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def tag_image(self, image_path: str) -> List[str]:
        # Fallback tags nếu chưa tải weights onnx
        return ["masterpiece", "best quality", "solo", "looking at viewer"]
