import os
import base64
from typing import Optional, Dict
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from .base import BaseCaptioner
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

logger = setup_logger(__name__)

class DeepSeekVisionCaptioner(BaseCaptioner):
    """Generates captions using DeepSeek / OpenAI compatible Vision API endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "deepseek-chat",
        task_type: str = "character"
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required for DeepSeek/OpenAI Vision Captioner.")
        
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model_name = model_name
        self.task_type = task_type
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        try:
            base64_img = self._encode_image(image_path)
            prompt = (
                "You are an expert training dataset captioner. Provide a dense, accurate 1-2 sentence description of this image "
                "focusing on the subject, style, lighting, composition, and visual details. Be precise and avoid introductory phrases."
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            raw_text = response.choices[0].message.content.strip()
            return CaptionCleaner.clean_text(raw_text, trigger_word=trigger_word, is_danbooru_tags=False)
        except Exception as e:
            logger.error(f"DeepSeek/OpenAI API caption failed for {image_path}: {e}")
            return f"{trigger_word or ''}, high quality image"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[-1].lower() in valid_exts
        ]

        console.print(f"[bold cyan]✨ Captioning {len(images)} images in '{directory}' using API ({self.model_name})...[/bold cyan]")
        success = 0
        skipped = 0

        for img_p in tqdm(images, desc="API Captioning"):
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            if os.path.exists(txt_p) and not overwrite:
                skipped += 1
                continue

            caption = self.caption_image(img_p, trigger_word=trigger_word)
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(caption)
            success += 1

        return {"processed": success, "skipped": skipped}
