import math
from typing import List, Tuple, Dict
from PIL import Image
from pathlib import Path
from ..core.logger import setup_logger

logger = setup_logger(__name__)

class AspectRatioBucketer:
    """
    Quản lý chia Buckets độ phân giải tương ứng theo tỉ lệ khung hình (Aspect Ratio Bucketing).
    Đảm bảo 100% không làm méo mặt hay kéo dãn cơ thể trong quá trình huấn luyện.
    """

    STANDARD_BUCKETS_1024: List[Tuple[int, int]] = [
        (1024, 1024), # 1:1 Square
        (832, 1216),  # 2:3 Vertical Portrait (rất phổ biến cho người)
        (768, 1344),  # 9:16 Full Body
        (1216, 832),  # 3:2 Horizontal
        (1344, 768),  # 16:9 Landscape
        (896, 1152),  # 3:4 Portrait
        (1152, 896),  # 4:3 Landscape
        (704, 1408),  # Extreme Vertical
        (1408, 704),  # Extreme Horizontal
    ]

    STANDARD_BUCKETS_512: List[Tuple[int, int]] = [
        (512, 512),
        (448, 576),
        (384, 704),
        (576, 448),
        (704, 384),
        (416, 640),
        (640, 416)
    ]

    @classmethod
    def get_best_bucket(cls, width: int, height: int, target_res: int = 1024) -> Tuple[int, int]:
        """Tìm bucket có tỉ lệ gần nhất với kích thước ảnh gốc."""
        buckets = cls.STANDARD_BUCKETS_1024 if target_res >= 768 else cls.STANDARD_BUCKETS_512
        orig_ratio = width / height

        best_bucket = buckets[0]
        min_diff = float("inf")

        for bw, bh in buckets:
            b_ratio = bw / bh
            diff = abs(math.log(orig_ratio / b_ratio))
            if diff < min_diff:
                min_diff = diff
                best_bucket = (bw, bh)

        return best_bucket

    @classmethod
    def analyze_dataset_buckets(cls, image_paths: List[str], target_res: int = 1024) -> Dict[Tuple[int, int], int]:
        """Thống kê sự phân bổ bucket của cả tập dữ liệu."""
        distribution: Dict[Tuple[int, int], int] = {}
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    bw, bh = cls.get_best_bucket(img.width, img.height, target_res)
                    distribution[(bw, bh)] = distribution.get((bw, bh), 0) + 1
            except Exception:
                continue
        return distribution
