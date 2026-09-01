import os
import shutil
from typing import Optional, Dict, Any, List
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

class Florence2Captioner(BaseCaptioner):
    """
    Microsoft Florence-2 Vision-Language Captioning Engine:
    - Hỗ trợ Florence-2-base, Florence-2-large, và MiaoshouAI PromptGen v2.0.
    - Vá triệt để lỗi 'Florence2LanguageConfig has no attribute forced_bos_token_id' trên transformers mới.
    - Hỗ trợ lưu cache model trực tiếp vào Google Drive để tái sử dụng mãi mãi.
    - Đa dạng chế độ: <MORE_DETAILED_CAPTION>, <DETAILED_CAPTION>, <GENERATE_TAGS>, <CAPTION>.
    """

    MODEL_MIRRORS = {
        "florence-2-base": "microsoft/Florence-2-base",
        "florence-2-large": "microsoft/Florence-2-large",
        "florence-2-promptgen": "MiaoshouAI/Florence-2-base-PromptGen-v2.0",
        "florence-2-promptgen-large": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
    }

    TASK_PROMPTS = {
        "more_detailed": "<MORE_DETAILED_CAPTION>",
        "detailed": "<DETAILED_CAPTION>",
        "tags": "<GENERATE_TAGS>",
        "caption": "<CAPTION>",
        "promptgen": "<GENERATE_TAGS>",
    }

    def __init__(
        self,
        model_name: str = "florence-2-base",
        task: str = "more_detailed",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Any = None
    ):
        self.model_id = self.MODEL_MIRRORS.get(model_name.lower(), model_name)
        self.task_key = task.lower()
        self.task_prompt = self.TASK_PROMPTS.get(self.task_key, "<MORE_DETAILED_CAPTION>")
        self.cache_dir = cache_dir

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

    @staticmethod
    def _apply_transformers_compatibility_patch(config: Any):
        """
        Vá lỗi tương thích giữa Florence-2 và các phiên bản transformers mới (>= 4.45):
        Khắc phục triệt để 'AttributeError: Florence2LanguageConfig object has no attribute forced_bos_token_id'.
        """
        configs_to_patch = [config]
        for attr in ["text_config", "language_config", "vision_config"]:
            if hasattr(config, attr):
                sub = getattr(config, attr)
                if sub is not None:
                    configs_to_patch.append(sub)

        for cfg in configs_to_patch:
            cls = cfg.__class__
            # Đảm bảo các thuộc tính token_id luôn tồn tại trên class và instance
            for token_attr in ["forced_bos_token_id", "forced_eos_token_id", "_attn_implementation"]:
                if not hasattr(cls, token_attr):
                    setattr(cls, token_attr, None)
                if not hasattr(cfg, token_attr):
                    setattr(cfg, token_attr, None)

            # Bổ sung __getattr__ an toàn nếu chưa có
            orig_getattr = getattr(cls, "__getattr__", None)
            def safe_getattr(self, name):
                if name in ("forced_bos_token_id", "forced_eos_token_id", "_attn_implementation"):
                    return None
                if orig_getattr:
                    return orig_getattr(self, name)
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            cls.__getattr__ = safe_getattr

    def _lazy_load_model(self):
        if self._loaded:
            return

        try:
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            console.print(f"[bold cyan]📥 Tải và nạp local caption model ({self.model_id})...[/bold cyan]")

            # 1. Nạp và vá Config trước khi khởi tạo Model
            config = AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            self._apply_transformers_compatibility_patch(config)

            # 2. Nạp Model với Config đã được vá hoàn chỉnh
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).to(self.device)

            # 3. Nạp Processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            self._loaded = True
            console.print(f"[bold green]✓ Florence-2 model ({self.model_id}) đã nạp thành công vào {self.device.upper()}![/bold green]")
        except Exception as e:
            logger.error(f"Failed to load Florence-2 model: {e}")
            raise e

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        self._lazy_load_model()
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=self.task_prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=384,
                    do_sample=False,
                    num_beams=3
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(image.width, image.height)
            )

            raw_caption = parsed_answer.get(self.task_prompt, "")
            if isinstance(raw_caption, dict):
                raw_caption = raw_caption.get("caption", str(raw_caption))

            return CaptionCleaner.clean_text(str(raw_caption), trigger_word=trigger_word, is_danbooru_tags=False)
        except Exception as e:
            logger.error(f"Florence-2 caption error for {image_path}: {e}")
            return f"{trigger_word or ''}, high quality portrait photo"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        self._lazy_load_model()
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif"}
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[-1].lower() in valid_exts
        ]

        console.print(f"[bold cyan]✨ Tiến hành gán nhãn {len(images)} ảnh bằng Florence-2 ({self.task_prompt})...[/bold cyan]")
        success = 0
        skipped = 0

        for img_p in tqdm(images, desc="Florence-2 Captioning"):
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            if os.path.exists(txt_p) and not overwrite:
                skipped += 1
                continue

            caption = self.caption_image(img_p, trigger_word=trigger_word)
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(caption)
            success += 1

        console.print(f"[bold green]✓ Gán nhãn hoàn tất![/bold green] Đã xử lý: {success}, Bỏ qua: {skipped}")
        return {"processed": success, "skipped": skipped}
