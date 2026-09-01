import math
from typing import List, Tuple, Dict
from PIL import Image

class AspectRatioBucketer:
    """Computes optimal resolution buckets to train diverse aspect ratios without distortion."""

    @staticmethod
    def generate_buckets(
        base_res: int = 1024,
        min_res: int = 512,
        max_res: int = 2048,
        step: int = 64
    ) -> List[Tuple[int, int]]:
        """Generates a list of valid (width, height) resolution pairs preserving pixel area."""
        target_pixels = base_res * base_res
        buckets = set()

        for w in range(min_res, max_res + 1, step):
            h = int(target_pixels / w)
            # Align to step
            h = (h // step) * step
            if min_res <= h <= max_res:
                buckets.add((w, h))

        # Sort by aspect ratio (w/h)
        sorted_buckets = sorted(list(buckets), key=lambda x: x[0] / x[1])
        return sorted_buckets

    @classmethod
    def find_best_bucket(
        cls,
        image_w: int,
        image_h: int,
        buckets: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Finds the closest bucket aspect ratio for a given image size."""
        img_aspect = image_w / image_h
        best_bucket = buckets[0]
        min_diff = float("inf")

        for b_w, b_h in buckets:
            bucket_aspect = b_w / b_h
            diff = abs(img_aspect - bucket_aspect)
            if diff < min_diff:
                min_diff = diff
                best_bucket = (b_w, b_h)

        return best_bucket
