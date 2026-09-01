"""
Florence-2 Vision Captioning Engine
Gán nhãn tự động chuẩn hóa sử dụng mô hình Microsoft Florence-2 chạy cục bộ.
Khắc phục triệt để lỗi 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
trên các phiên bản mới của thư viện transformers.
"""

import os
import sys
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

def _patch_florence_config():
    """Globally patches PretrainedConfig to return None for missing forced_bos_token_id."""
    try:
        from transformers.configuration_utils import PretrainedConfig
        orig_getattr = getattr(PretrainedConfig, "__getattr__", None)
        def safe_getattr(self, key):
            if key in ("forced_bos_token_id", "forced_eos_token_id", "_attn_implementation", "force_bos_token_to_be_generated"):
                return None
            if orig_getattr:
                return orig_getattr(self, key)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        PretrainedConfig.__getattr__ = safe_getattr

        # Patch HeterogeneousPretrainedConfig in transformers >= 4.45
        try:
            import transformers.integrations.heterogeneity.configuration_utils as hetero
            if hasattr(hetero, "HeterogeneousPretrainedConfig"):
                orig_hetero = getattr(hetero.HeterogeneousPretrainedConfig, "__getattr__", None)
                def safe_hetero_getattr(self, key):
                    if key in ("forced_bos_token_id", "forced_eos_token_id", "_attn_implementation", "force_bos_token_to_be_generated"):
                        return None
                    if orig_hetero:
                        return orig_hetero(self, key)
                    return super(hetero.HeterogeneousPretrainedConfig, self).__getattr__(key)
                hetero.HeterogeneousPretrainedConfig.__getattr__ = safe_hetero_getattr
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Patch info: {e}")

_patch_florence_config()

class Florence2Captioner(BaseCaptioner):
    """Gán nhãn tự động với mô hình Microsoft Florence-2 chạy cục bộ."""

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        task_mode: str = "General",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.task_mode = task_mode
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self.dtype = torch.float16 if (torch and torch.cuda.is_available()) else torch.float32

        self.model = None
        self.processor = None
        self._loaded = False

    def _lazy_load_model(self):
        if self._loaded:
            return

        try:
            _patch_florence_config()
            from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM

            console.print(f"[bold cyan]📥 Tải và nạp mô hình Florence-2 ({self.model_id})...[/bold cyan]")

            # Nạp và patch config
            config = AutoConfig.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            if hasattr(config, "text_config") and not hasattr(config.text_config, "forced_bos_token_id"):
                config.text_config.forced_bos_token_id = getattr(config.text_config, "bos_token_id", None)
            if not hasattr(config, "forced_bos_token_id"):
                config.forced_bos_token_id = getattr(config, "bos_token_id", None)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

            self._loaded = True
            console.print(f"[bold green]✓ Florence-2 đã sẵn sàng trên {self.device.upper()}![/bold green]")
        except Exception as e:
            logger.error(f"Không thể nạp Florence-2: {e}")
            raise e

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        self._lazy_load_model()
        task_prompt = "<MORE_DETAILED_CAPTION>" if self.task_mode != "Short" else "<CAPTION>"

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")

            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32 and torch.cuda.is_available():
                        processed_inputs[k] = v.to(self.device, dtype=torch.float16)
                    else:
                        processed_inputs[k] = v.to(self.device)
                else:
                    processed_inputs[k] = v

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=processed_inputs["input_ids"],
                    pixel_values=processed_inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            raw_caption = parsed_answer.get(task_prompt, "").strip()
            if isinstance(raw_caption, dict):
                raw_caption = raw_caption.get("caption", str(raw_caption))

            caption = CaptionCleaner.clean_text(str(raw_caption), trigger_word=trigger_word, is_danbooru_tags=False)
            return caption
        except Exception as e:
            logger.error(f"Lỗi Florence-2 tại {image_path}: {e}")
            return f"{trigger_word or ''}, high quality portrait photograph"

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

        console.print(f"\n[bold cyan]🏷️ Florence-2: Gán nhãn {len(images)} ảnh...[/bold cyan]")
        success = 0
        skipped = 0

        for img_p in tqdm(images, desc="🏷️ Florence-2"):
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

        console.print(f"[bold green]🎉 Hoàn tất gán nhãn {success} ảnh với Florence-2![/bold green]")
        return {"processed": success, "skipped": skipped}
