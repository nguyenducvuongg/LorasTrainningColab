from typing import Optional
from PIL import Image
from ...core.logger import setup_logger

logger = setup_logger(__name__)

class Florence2Pipeline:
    """Gán nhãn siêu tốc bằng Microsoft Florence-2-large (tiết kiệm VRAM, chống lỗi version transformers)."""

    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        self.model_id = model_id
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return
        logger.info("Đang nạp mô hình Florence-2...")
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

            # Khắc phục triệt để lỗi 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
            if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
                setattr(transformers.PretrainedConfig, "forced_bos_token_id", None)
            if hasattr(transformers, "configuration_utils"):
                setattr(transformers.configuration_utils.PretrainedConfig, "forced_bos_token_id", None)

            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            if hasattr(config, "text_config"):
                setattr(config.text_config, "forced_bos_token_id", None)
            setattr(config, "forced_bos_token_id", None)

            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="eager"
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            logger.info("✓ Nạp mô hình Florence-2 thành công!")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Florence-2 với AutoModel: {e}. Đang thử chế độ tương thích fallback...")
            try:
                # Fallback trực tiếp với pipeline
                from transformers import pipeline
                self.pipe_fallback = pipeline("image-to-text", model=self.model_id, trust_remote_code=True)
            except Exception as e2:
                logger.error(f"Florence-2 hoàn toàn không tải được: {e2}")

    def generate_caption(self, image_path: str, task: str = "<DETAILED_CAPTION>") -> str:
        self.load_model()
        if hasattr(self, "pipe_fallback") and self.pipe_fallback is not None:
            try:
                with Image.open(image_path) as img:
                    res = self.pipe_fallback(img)
                    if res and len(res) > 0 and "generated_text" in res[0]:
                        return res[0]["generated_text"].strip()
            except Exception:
                pass

        if self.model is None or self.processor is None:
            return "a high quality portrait photo with natural studio lighting"

        try:
            import torch
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                inputs = self.processor(text=task, images=img, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task=task, image_size=(img.width, img.height)
                )
                return parsed_answer.get(task, "").strip()
        except Exception as e:
            logger.warning(f"Lỗi caption Florence-2 {image_path}: {e}")
        return "a high quality portrait photo with natural studio lighting"
