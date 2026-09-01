"""
AI Captioning Base System & Task Presets
Định nghĩa các kịch bản Prompt chuyên biệt cho từng dạng bài toán LoRA
(Nhân vật, Skin/Portrait, Phục hồi, Phong cách, Tổng quát).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List

TASK_PROMPT_PRESETS: Dict[str, str] = {
    "General": (
        "Describe the image in a concise, precise, and descriptive natural language paragraph. "
        "Focus on the main subject, background, lighting, artistic style, and key elements."
    ),
    "Skin_Portrait": (
        "Focus strictly on high-detail facial texture, skin micro-details, blemishes, pores, natural wrinkles, "
        "lighting on the face, eye reflections, and realistic human features. "
        "Do not over-describe unrelated distant background."
    ),
    "Character_Outfit": (
        "Describe the pose, clothing, outfit details, fabric texture, accessories, background environment, and camera angle. "
        "Do not describe the person's specific facial features or facial identity, so the model learns to associate the identity with the trigger word."
    ),
    "Art_Style": (
        "Describe the subject and composition clearly, but completely omit mentioning the art style, medium, or artist name. "
        "The model must learn the style purely through the trigger word."
    ),
    "Upscale_Restoration": (
        "Describe the fine photographic details, crisp edges, sharpness, depth of field, realistic materials, "
        "reflections, and high-fidelity rendering aspects of the image."
    ),
}

LENGTH_CONSTRAINTS: Dict[str, str] = {
    "Short": "Keep the caption under 25 words.",
    "Medium": "Keep the caption around 40-70 words.",
    "Long": "Provide a comprehensive, detailed description of 80-120 words.",
}

def build_task_prompt(
    task_mode: str = "General",
    caption_length: str = "Medium",
    trigger_word: Optional[str] = None,
) -> str:
    """Tạo câu lệnh Prompt hoàn chỉnh cho Vision LLM theo yêu cầu bài toán."""
    base_instr = TASK_PROMPT_PRESETS.get(task_mode, TASK_PROMPT_PRESETS["General"])
    len_instr = LENGTH_CONSTRAINTS.get(caption_length, LENGTH_CONSTRAINTS["Medium"])

    prompt = f"{base_instr} {len_instr}"
    if trigger_word and trigger_word.strip():
        prompt += f" Always start the description with the trigger word '{trigger_word.strip()}':"

    return prompt

class BaseCaptioner(ABC):
    """Abstract Base Class for all Vision-Language Captioning engines."""

    @abstractmethod
    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        """Generates caption for a single image."""
        pass

    @abstractmethod
    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        """Generates captions for all images in a directory."""
        pass
