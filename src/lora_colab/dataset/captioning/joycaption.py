import os
from typing import Optional, Dict, Any
from PIL import Image
from tqdm import tqdm
from .base import BaseCaptioner
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

class JoyCaptioner(BaseCaptioner):
    """Local Vision-Language Model Captioner (Florence-2 / JoyCaption Alpha)."""

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: Optional[str] = None,
        torch_dtype: Any = None
    ):
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if (torch and torch.cuda.is_available()) else (torch.float32 if torch else None)
        else:
            self.torch_dtype = torch_dtype
            
        self.model = None
        self.processor = None
        self._loaded = False

    def _lazy_load_model(self):
        if self._loaded:
            return
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            console.print(f"[bold cyan]📥 Loading local captioning model ({self.model_id})...[/bold cyan]")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._loaded = True
            logger.info("Local captioning model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load local captioning model: {e}")
            raise e

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        self._lazy_load_model()
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = "<MORE_DETAILED_CAPTION>"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height)
            )

            raw_caption = parsed_answer.get(prompt, "")
            return CaptionCleaner.clean_text(raw_caption, trigger_word=trigger_word, is_danbooru_tags=False)
        except Exception as e:
            logger.error(f"JoyCaption / Florence failed for {image_path}: {e}")
            return f"{trigger_word or ''}, high quality photography"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        self._lazy_load_model()
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[-1].lower() in valid_exts
        ]

        console.print(f"[bold cyan]✨ Captioning {len(images)} images with local VLM ({self.model_id})...[/bold cyan]")
        success = 0
        skipped = 0

        for img_p in tqdm(images, desc="VLM Captioning"):
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            if os.path.exists(txt_p) and not overwrite:
                skipped += 1
                continue

            caption = self.caption_image(img_p, trigger_word=trigger_word)
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(caption)
            success += 1

        return {"processed": success, "skipped": skipped}
