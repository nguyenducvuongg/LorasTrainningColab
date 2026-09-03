import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class FaceAwareCropGenerator:
    """
    Tự động nhận diện khuôn mặt và sinh tập dữ liệu Đa Tỷ Lệ (Multi-Scale Face Dataset):
    - Macro Close-up Face (1024x1024): Nắm trọn chi tiết vi mô mắt, mũi, lỗ chân lông, ánh sáng trên da.
    - Medium Shot (Upper Body): Nắm bắt tỷ lệ vai, cổ, trang phục phần trên.
    - Wide Shot (Full Frame): Nắm bắt bố cục toàn cảnh và dáng người.
    ĐÂY LÀ CHÌA KHÓA VÀNG ĐỂ ĐẠT ĐỘ GIỐNG ẢNH ĐẦU VÀO Ở MỨC 100%.
    """

    def __init__(self):
        self.face_cascade = None
        self._init_cascade()

    def _init_cascade(self):
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def detect_face_bbox(self, pil_image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Nhận diện bounding box khuôn mặt lớn nhất (x, y, w, h)."""
        w, h = pil_image.size
        if self.face_cascade is None:
            # Heuristic Rule-of-Thirds crop nếu không có OpenCV
            fw = int(w * 0.4)
            fh = int(h * 0.4)
            fx = int(w * 0.3)
            fy = int(h * 0.15)
            return (fx, fy, fw, fh)

        try:
            import cv2
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(64, 64)
            )
            if len(faces) == 0:
                # Fallback center-upper box
                return (int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.5))
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        except Exception:
            return (int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.5))

    def process_and_generate_crops(
        self, 
        image_path: str, 
        output_dir: str, 
        trigger_word: str = "sks",
        class_word: str = "person"
    ) -> List[Dict[str, str]]:
        """
        Trích xuất đa tỷ lệ từ 1 bức ảnh nguồn:
        Trả về danh sách dict chứa: path ảnh mới, type ('face_macro' | 'medium' | 'full'), prefix gợi ý.
        """
        results = []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                stem = Path(image_path).stem

                # 1. Ảnh toàn thể (Full frame)
                full_path = out_dir / f"{stem}_full.jpg"
                img.save(full_path, "JPEG", quality=98)
                results.append({
                    "path": str(full_path),
                    "type": "full",
                    "prefix": f"wide shot photo of {trigger_word} {class_word}"
                })

                face_bbox = self.detect_face_bbox(img)
                if face_bbox is None:
                    return results

                fx, fy, fw, fh = face_bbox
                cx, cy = fx + fw // 2, fy + fh // 2

                # 2. Macro Close-up Face (Crop cận mặt mở rộng 1.4x)
                pad_face = int(max(fw, fh) * 0.7)
                x1 = max(0, cx - pad_face)
                y1 = max(0, cy - pad_face)
                x2 = min(w, cx + pad_face)
                y2 = min(h, cy + pad_face)
                face_crop = img.crop((x1, y1, x2, y2))
                face_path = out_dir / f"{stem}_face.jpg"
                face_crop.save(face_path, "JPEG", quality=98)
                results.append({
                    "path": str(face_path),
                    "type": "face_macro",
                    "prefix": f"ultra close-up portrait of {trigger_word} {class_word}, detailed eyes, realistic skin texture"
                })

                # 3. Medium Shot (Crop nửa thân trên)
                pad_med_w = int(fw * 1.8)
                pad_med_top = int(fh * 0.9)
                pad_med_bottom = int(fh * 2.8)
                mx1 = max(0, cx - pad_med_w)
                my1 = max(0, cy - pad_med_top)
                mx2 = min(w, cx + pad_med_w)
                my2 = min(h, cy + pad_med_bottom)
                med_crop = img.crop((mx1, my1, mx2, my2))
                med_path = out_dir / f"{stem}_medium.jpg"
                med_crop.save(med_path, "JPEG", quality=98)
                results.append({
                    "path": str(med_path),
                    "type": "medium",
                    "prefix": f"medium shot photo of {trigger_word} {class_word}, head and shoulders portrait"
                })

        except Exception as e:
            logger.warning(f"Lỗi khi trích xuất khuôn mặt {image_path}: {e}")

        return results
