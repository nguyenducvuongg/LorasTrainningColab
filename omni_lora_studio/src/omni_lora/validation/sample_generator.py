from typing import List, Optional
from pathlib import Path
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class SampleGridGenerator:
    """Tạo lưới ảnh kiểm nghiệm trong quá trình huấn luyện để đánh giá độ giống và chống overfit."""

    @classmethod
    def generate_prompt_matrix(cls, trigger_word: str, class_word: str) -> List[str]:
        return [
            f"studio portrait photo of {trigger_word} {class_word}, clean neutral grey background, rim lighting, 8k",
            f"candid street photo of {trigger_word} {class_word} wearing a stylish jacket, walking in rainy Tokyo, neon bokeh",
            f"cinematic close-up of {trigger_word} {class_word}, looking sideways, intense gaze, warm sunset lighting",
            f"high-fashion editorial photography of {trigger_word} {class_word}, elegant white turtleneck, minimalist room"
        ]
