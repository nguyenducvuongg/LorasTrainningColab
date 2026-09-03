from typing import Optional
from PIL import Image
from ...core.logger import setup_logger

logger = setup_logger(__name__)

class Florence2Pipeline:
    """Gán nhãn siêu tốc bằng Microsoft Florence-2-large (tiết kiệm VRAM)."""

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
            from transformers import AutoProcessor, AutoModelForCausalLM
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except Exception as e:
            logger.warning(f"Không thể khởi tạo Florence-2: {e}")

    def generate_caption(self, image_path: str, task: str = "<DETAILED_CAPTION>") -> str:
        self.load_model()
        if self.model is None or self.processor is None:
            return "a sharp portrait photograph"
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
            logger.warning(f"Lỗi caption Florence-2: {e}")
        return "a sharp portrait photograph"
