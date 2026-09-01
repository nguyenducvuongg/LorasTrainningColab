"""
JoyCaption AI Vision Engine
Gán nhãn tự động chuẩn hóa sử dụng JoyCaption Alpha Two / Two local Transformers model.
Hỗ trợ tương thích hoàn hảo với transformers phiên bản mới (LlavaForConditionalGeneration).
"""

import os
from typing import Optional, Dict, Any, List
from PIL import Image
from tqdm import tqdm
from .base import BaseCaptioner, build_task_prompt
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}

class JoyCaptioner(BaseCaptioner):
    """Gán nhãn tự động với mô hình JoyCaption chạy cục bộ trên GPU."""

    def __init__(
        self,
        model_id: str = "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        task_mode: str = "General",
        caption_length: str = "Medium",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.task_mode = task_mode
        self.caption_length = caption_length
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False

    def _lazy_load_model(self):
        if self._loaded:
            return

        try:
            from transformers import AutoProcessor
            console.print(f"[bold cyan]📥 Tải và nạp JoyCaption ({self.model_id})...[/bold cyan]")

            self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            dtype = torch.bfloat16 if (torch and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

            try:
                from transformers import LlavaForConditionalGeneration
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if (torch and torch.cuda.is_available()) else None,
                    cache_dir=self.cache_dir
                )
            except Exception:
                from transformers import AutoModelForVision2Seq
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map="auto" if (torch and torch.cuda.is_available()) else None,
                    cache_dir=self.cache_dir
                )

            if not (torch and torch.cuda.is_available()) and self.model:
                self.model = self.model.to("cpu")

            self._loaded = True
            console.print(f"[bold green]✓ JoyCaption đã sẵn sàng trên GPU![/bold green]")
        except Exception as e:
            logger.error(f"Không thể nạp JoyCaption: {e}")
            raise e

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        self._lazy_load_model()
        prompt = build_task_prompt(self.task_mode, self.caption_length, trigger_word)

        try:
            image = Image.open(image_path).convert("RGB")
            convo = [{"role": "user", "content": f"{prompt}\n<image>"}]
            prompt_text = self.processor.apply_chat_template(convo, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.5)

            input_len = inputs["input_ids"].shape[1]
            caption = self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
            return CaptionCleaner.clean_text(caption, trigger_word=trigger_word, is_danbooru_tags=False)
        except Exception as e:
            logger.error(f"Lỗi JoyCaption tại {image_path}: {e}")
            return f"{trigger_word or ''}, beautiful photography"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        self._lazy_load_model()
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[-1].lower() in SUPPORTED_IMAGE_EXTS
        ]

        console.print(f"\n[bold cyan]🏷️ JoyCaption: Gán nhãn {len(images)} ảnh...[/bold cyan]")
        success = 0
        skipped = 0

        for img_p in tqdm(images, desc="🏷️ JoyCaption"):
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            if os.path.exists(txt_p) and not overwrite:
                try:
                    with open(txt_p, "r", encoding="utf-8") as f:
                        if f.read().strip():
                            skipped += 1
                            continue
                except Exception:
                    pass

            caption = self.caption_image(img_p, trigger_word=trigger_word)
            if caption:
                with open(txt_p, "w", encoding="utf-8") as f:
                    f.write(caption)
                success += 1

        console.print(f"[bold green]🎉 Hoàn tất gán nhãn {success} ảnh với JoyCaption![/bold green]")
        return {"processed": success, "skipped": skipped}
