from typing import Optional
from PIL import Image
from ...core.logger import setup_logger

logger = setup_logger(__name__)

class JoyCaptionPipeline:
    """Tự động gán nhãn chi tiết ảnh thực tế bằng JoyCaption Alpha Two."""

    def __init__(self, model_id: str = "fancyfeast/llama-joycaption-alpha-two-hf-llava"):
        self.model_id = model_id
        self.pipe = None

    def load_model(self):
        if self.pipe is not None:
            return
        logger.info("Đang nạp mô hình JoyCaption...")
        try:
            import torch
            from transformers import pipeline
            self.pipe = pipeline(
                "image-to-text",
                model=self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Không thể khởi tạo JoyCaption cục bộ: {e}")

    def generate_caption(self, image_path: str) -> str:
        self.load_model()
        if self.pipe is None:
            return "a high quality portrait photo with natural lighting"
        try:
            with Image.open(image_path) as img:
                res = self.pipe(img, max_new_tokens=150)
                if res and isinstance(res, list) and "generated_text" in res[0]:
                    return res[0]["generated_text"].strip()
        except Exception as e:
            logger.warning(f"Lỗi caption JoyCaption {image_path}: {e}")
        return "a high quality detailed photo"
