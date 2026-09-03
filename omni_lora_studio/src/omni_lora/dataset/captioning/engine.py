from typing import Optional
from pathlib import Path
from .identity_isolator import IdentityIsolator
from .joycaption import JoyCaptionPipeline
from .florence2 import Florence2Pipeline
from .wd14 import WD14Tagger
from .gemini_api import GeminiVisionCaptioner
from ...core.logger import setup_logger, console

logger = setup_logger(__name__)

class CaptioningEngine:
    """Bộ điều phối gán nhãn đa phương thức tích hợp cô lập chủ thể."""

    def __init__(
        self, 
        backend: str = "florence2", 
        trigger_word: str = "sks", 
        class_word: str = "person",
        enable_isolation: bool = True
    ):
        self.backend = backend.lower()
        self.trigger_word = trigger_word
        self.class_word = class_word
        self.enable_isolation = enable_isolation

        if self.backend == "joycaption":
            self.worker = JoyCaptionPipeline()
        elif self.backend == "florence2":
            self.worker = Florence2Pipeline()
        elif self.backend == "wd14":
            self.worker = WD14Tagger()
        elif self.backend == "gemini":
            self.worker = GeminiVisionCaptioner()
        else:
            self.worker = Florence2Pipeline()

    def process_file(self, image_path: str, overwrite: bool = False) -> str:
        txt_path = Path(image_path).with_suffix(".txt")
        if txt_path.exists() and not overwrite:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_caption = f.read().strip()
        else:
            if hasattr(self.worker, "generate_caption"):
                raw_caption = self.worker.generate_caption(image_path)
            elif hasattr(self.worker, "tag_image"):
                tags = self.worker.tag_image(image_path)
                raw_caption = ", ".join(tags)
            else:
                raw_caption = "a photo"

        # Áp dụng bộ lọc cô lập chủ thể để đạt 100% likeness
        if self.enable_isolation:
            final_caption = IdentityIsolator.purify_caption(
                raw_caption=raw_caption,
                trigger_word=self.trigger_word,
                class_word=self.class_word
            )
        else:
            final_caption = f"{self.trigger_word} {self.class_word}, {raw_caption}"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_caption)

        return final_caption
