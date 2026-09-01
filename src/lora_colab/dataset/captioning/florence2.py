import os
import sys
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

def _apply_global_transformers_patch():
    """
    Vá lỗi tương thích toàn cầu cho PretrainedConfig & HeterogeneousPretrainedConfig
    trước khi nạp mã nguồn từ xa của Florence-2.
    Loại bỏ triệt để 'AttributeError: Florence2LanguageConfig object has no attribute forced_bos_token_id'.
    """
    try:
        import transformers.configuration_utils as cfg_utils
        from transformers.configuration_utils import PretrainedConfig

        orig_getattr = getattr(PretrainedConfig, "__getattr__", None)
        def patched_getattr(self, key):
            if key in ("forced_bos_token_id", "forced_eos_token_id", "_attn_implementation", "force_bos_token_to_be_generated"):
                return None
            if orig_getattr is not None:
                return orig_getattr(self, key)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

        PretrainedConfig.__getattr__ = patched_getattr

        # Patch HeterogeneousPretrainedConfig trong transformers mới (>= 4.45)
        try:
            import transformers.integrations.heterogeneity.configuration_utils as hetero_mod
            if hasattr(hetero_mod, "HeterogeneousPretrainedConfig"):
                orig_hetero = getattr(hetero_mod.HeterogeneousPretrainedConfig, "__getattr__", None)
                def patched_hetero(self, key):
                    if key in ("forced_bos_token_id", "forced_eos_token_id", "_attn_implementation", "force_bos_token_to_be_generated"):
                        return None
                    if orig_hetero is not None:
                        return orig_hetero(self, key)
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
                hetero_mod.HeterogeneousPretrainedConfig.__getattr__ = patched_hetero
        except Exception:
            pass

        # Quét và vá trực tiếp các class Florence2LanguageConfig đã nạp trong sys.modules
        for mod_name, mod in list(sys.modules.items()):
            if "florence2" in mod_name.lower() or "configuration_florence" in mod_name.lower():
                if hasattr(mod, "Florence2LanguageConfig"):
                    setattr(mod.Florence2LanguageConfig, "forced_bos_token_id", None)
                    setattr(mod.Florence2LanguageConfig, "forced_eos_token_id", None)
                    setattr(mod.Florence2LanguageConfig, "_attn_implementation", None)
    except Exception as e:
        logger.debug(f"Transformers patch info: {e}")

# Áp dụng patch ngay khi module được import
_apply_global_transformers_patch()

class Florence2Captioner(BaseCaptioner):
    """
    Microsoft Florence-2 & JoyCaption Vision-Language Captioning Engine:
    - Tự động vá lỗi tương thích với mọi phiên bản transformers mới nhất (>= 4.45+).
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

    def _lazy_load_model(self):
        if self._loaded:
            return

        try:
            # 1. Kích hoạt lại global patch trước khi gọi AutoConfig
            _apply_global_transformers_patch()

            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
            console.print(f"[bold cyan]📥 Tải và nạp local caption model ({self.model_id})...[/bold cyan]")

            # 2. Nạp Config
            config = AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            # Đảm bảo các sub-config cũng được vá
            for attr in ["text_config", "language_config", "vision_config"]:
                if hasattr(config, attr):
                    sub = getattr(config, attr)
                    if sub is not None:
                        setattr(sub.__class__, "forced_bos_token_id", None)
                        setattr(sub.__class__, "forced_eos_token_id", None)
                        setattr(sub.__class__, "_attn_implementation", None)

            # 3. Nạp Model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).to(self.device)

            # 4. Nạp Processor
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
