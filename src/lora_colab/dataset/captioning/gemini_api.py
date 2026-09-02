import os
from PIL import Image
from typing import Optional, Dict
from tqdm import tqdm
from .base import BaseCaptioner
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

logger = setup_logger(__name__)

SYSTEM_PROMPTS = {
    "character": (
        "You are an expert AI dataset captioner for Flux and Diffusion models. "
        "Describe this character in detailed natural language. Focus on: gender, age, facial features, "
        "expression, hair style and color, eye color, clothing details, pose, framing (close-up, half-body, full-body), "
        "lighting, and background context. Keep it concise, descriptive, and objective without filler words."
    ),
    "style": (
        "You are an expert AI dataset captioner for Diffusion art style models. "
        "Describe the visual art style of this image in rich detail: medium (oil painting, 3D render, watercolor, anime illustration, vintage photo), "
        "linework, color palette, shading technique, brushwork, lighting mood, and artistic composition. Avoid filler words."
    ),
    "skin_enhancement": (
        "You are an expert AI dataset captioner for high-end photorealism and skin enhancement LoRAs. "
        "Describe the skin texture, pores, natural imperfections, subsurface scattering, dynamic range, lighting reflection, "
        "and camera lens depth of field with meticulous technical detail."
    ),
    "general": (
        "Describe the contents, subjects, colors, lighting, materials, and composition of this image in clear, precise descriptive English."
    )
}

class GeminiVisionCaptioner(BaseCaptioner):
    """Generates ultra-high-quality image captions using Google Gemini Vision API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", task_type: str = "character"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiVisionCaptioner.")
        self.model_name = model_name
        self.task_type = task_type
        self.system_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["general"])
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize google-genai client: {e}")
            raise e

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        """Calls Gemini Vision API on an image and returns formatted caption."""
        try:
            with Image.open(image_path) as img:
                # Resize if excessively large to save API bandwidth
                max_dim = 1536
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                
                prompt = (
                    f"{self.system_prompt}\n\n"
                    "Write a 1-3 sentence dense descriptive caption for this image suitable for image generation training."
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[img, prompt]
                )
                
                raw_caption = response.text.strip()
                cleaned = CaptionCleaner.clean_text(
                    raw_caption,
                    trigger_word=trigger_word,
                    is_danbooru_tags=False
                )
                return cleaned
        except Exception as e:
            logger.error(f"Gemini API caption failed for {image_path}: {e}")
            return f"{trigger_word or ''}, high quality image"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """Captions all images in a directory using Gemini API."""
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[-1].lower() in valid_exts
        ]

        images_to_process = []
        skipped_count = 0
        for img_path in images:
            base_name = os.path.splitext(img_path)[0]
            txt_path = base_name + ".txt"
            if os.path.exists(txt_path) and not overwrite and skip_existing:
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        if f.read().strip():
                            skipped_count += 1
                            continue
                except Exception:
                    pass
            images_to_process.append(img_path)

        if not images_to_process:
            console.print(f"[bold green]⚡ Tất cả {len(images)} ảnh đã có sẵn file caption .txt tương ứng![/bold green] Bỏ qua Gemini API.")
            return {"processed": 0, "skipped": skipped_count}

        console.print(f"[bold cyan]✨ Captioning {len(images_to_process)}/{len(images)} images in '{directory}' using Gemini Vision API ({self.model_name})...[/bold cyan]")
        
        success_count = 0
        for img_path in tqdm(images_to_process, desc="Gemini Captioning"):
            caption = self.caption_image(img_path, trigger_word=trigger_word)
            base_name = os.path.splitext(img_path)[0]
            txt_path = base_name + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            success_count += 1

        console.print(f"[bold green]✓ Gemini Captioning completed![/bold green] Processed: {success_count}, Skipped: {skipped_count}")
        return {"processed": success_count, "skipped": skipped_count}
