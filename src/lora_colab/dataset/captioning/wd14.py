import os
import csv
from typing import Optional, Dict, List, Set, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm
from .base import BaseCaptioner
from ..cleaner import CaptionCleaner
from ...core.logger import setup_logger, console

logger = setup_logger(__name__)

WD14_MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

class WD14Tagger(BaseCaptioner):
    """Local WD14 Tagger (SmilingWolf v3) for anime, 2D art, and Pony/Illustrious LoRA training."""

    def __init__(
        self,
        model_repo: str = WD14_MODEL_REPO,
        general_threshold: float = 0.35,
        character_threshold: float = 0.65,
        blacklist: Optional[Set[str]] = None
    ):
        self.model_repo = model_repo
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.blacklist = blacklist
        self.model = None
        self.tags: List[str] = []
        self.tag_categories: List[int] = []  # 0: general, 4: character, 9: rating
        self._loaded = False

    def _lazy_load_model(self):
        """Lazy loads ONNX model and tags CSV from Hugging Face."""
        if self._loaded:
            return
        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort

            console.print(f"[bold cyan]📥 Loading WD14 Tagger v3 from HuggingFace ({self.model_repo})...[/bold cyan]")
            model_path = hf_hub_download(repo_id=self.model_repo, filename="model.onnx")
            csv_path = hf_hub_download(repo_id=self.model_repo, filename="selected_tags.csv")

            # Load Tags CSV
            self.tags = []
            self.tag_categories = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    self.tags.append(row[1])
                    self.tag_categories.append(int(row[2]))

            # Init ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._loaded = True
            logger.info("WD14 Tagger successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load WD14 Tagger: {e}")
            raise e

    def _preprocess_image(self, image_path: str, target_size: int = 448) -> np.ndarray:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Pad to square
            w, h = img.size
            max_side = max(w, h)
            padded = Image.new("RGB", (max_side, max_side), (255, 255, 255))
            padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
            # Resize
            resized = padded.resize((target_size, target_size), Image.BICUBIC)
            img_arr = np.array(resized, dtype=np.float32)
            # RGB to BGR
            img_arr = img_arr[:, :, ::-1]
            # Add batch dimension: (1, 448, 448, 3)
            return np.expand_dims(img_arr, axis=0)

    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        self._lazy_load_model()
        try:
            input_tensor = self._preprocess_image(image_path)
            outputs = self.session.run(None, {self.input_name: input_tensor})[0][0]

            predicted_tags: List[str] = []
            for tag, category, prob in zip(self.tags, self.tag_categories, outputs):
                if category == 0 and prob >= self.general_threshold:
                    predicted_tags.append(tag)
                elif category == 4 and prob >= self.character_threshold:
                    predicted_tags.append(tag)

            cleaned = CaptionCleaner.clean_tag_list(
                predicted_tags,
                trigger_word=trigger_word,
                blacklist=self.blacklist
            )
            return cleaned
        except Exception as e:
            logger.error(f"WD14 tagger failed for {image_path}: {e}")
            return trigger_word or "1girl, solo"

    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[-1].lower() in valid_exts
        ]

        images_to_process = []
        skipped = 0
        for img_p in images:
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            if os.path.exists(txt_p) and not overwrite and skip_existing:
                try:
                    with open(txt_p, "r", encoding="utf-8") as f:
                        if f.read().strip():
                            skipped += 1
                            continue
                except Exception:
                    pass
            images_to_process.append(img_p)

        if not images_to_process:
            console.print(f"[bold green]⚡ Tất cả {len(images)} ảnh đã có sẵn file caption .txt tương ứng![/bold green] Bỏ qua WD14.")
            return {"processed": 0, "skipped": skipped}

        self._lazy_load_model()
        console.print(f"[bold cyan]🏷️ Tagging {len(images_to_process)}/{len(images)} images in '{directory}' using WD14 Tagger v3...[/bold cyan]")
        success = 0

        for img_p in tqdm(images_to_process, desc="WD14 Tagging"):
            caption = self.caption_image(img_p, trigger_word=trigger_word)
            txt_p = os.path.splitext(img_p)[0] + ".txt"
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(caption)
            success += 1

        return {"processed": success, "skipped": skipped}
