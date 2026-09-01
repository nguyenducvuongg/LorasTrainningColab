import os
import shutil
from PIL import Image, ImageOps
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif", ".tiff"}

class DatasetNormalizer:
    """
    Normalizes dataset images and associated caption files:
    - Auto-renames files to standard format: {prefix}_{index:04d}.{ext}
    - Converts corrupted/non-RGB modes (RGBA, P, CMYK) to clean standard RGB
    - Synchronizes matching .txt caption files
    """

    @classmethod
    def sanitize_image(cls, image_path: str, output_path: str, format: str = "PNG") -> bool:
        """Loads, cleans EXIF, converts to RGB, and saves image."""
        try:
            with Image.open(image_path) as img:
                # Transpose according to EXIF orientation tag
                img = ImageOps.exif_transpose(img)
                # Convert to RGB (handles transparency by pasting on white background)
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                    
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                if format.upper() == "PNG":
                    img.save(output_path, "PNG", optimize=True)
                else:
                    img.save(output_path, "JPEG", quality=95, optimize=True)
            return True
        except Exception as e:
            logger.error(f"Error sanitizing {image_path}: {e}")
            return False

    @classmethod
    def normalize_folder(
        cls,
        input_dir: str,
        output_dir: Optional[str] = None,
        prefix: str = "img",
        target_format: str = "PNG",
        start_index: int = 1
    ) -> Dict[str, Any]:
        """
        Normalizes and renames all images and captions in input_dir.
        If output_dir is None, normalizes in-place (via safe temporary staging).
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        in_place = output_dir is None or os.path.abspath(input_dir) == os.path.abspath(output_dir)
        target_dir = input_dir if in_place else output_dir
        os.makedirs(target_dir, exist_ok=True)

        # Collect valid image files
        raw_files = sorted(os.listdir(input_dir))
        image_files = [
            f for f in raw_files 
            if os.path.splitext(f)[-1].lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        if not image_files:
            logger.warning(f"No valid images found in {input_dir}")
            return {"processed_count": 0, "failed_count": 0}

        console.print(f"[bold cyan]🔄 Normalizing & Renaming {len(image_files)} images in '{input_dir}' with prefix '{prefix}'...[/bold cyan]")

        processed = 0
        failed = 0
        temp_staging = os.path.join(target_dir, "_temp_normalizing_staging")
        os.makedirs(temp_staging, exist_ok=True)

        for i, filename in enumerate(tqdm(image_files, desc="Normalizing Dataset")):
            idx = start_index + i
            base_src, ext_src = os.path.splitext(filename)
            src_img_path = os.path.join(input_dir, filename)
            src_txt_path = os.path.join(input_dir, base_src + ".txt")

            out_ext = ".png" if target_format.upper() == "PNG" else ".jpg"
            new_img_name = f"{prefix}_{idx:04d}{out_ext}"
            new_txt_name = f"{prefix}_{idx:04d}.txt"

            dest_img_path = os.path.join(temp_staging, new_img_name)
            dest_txt_path = os.path.join(temp_staging, new_txt_name)

            # Process Image
            success = cls.sanitize_image(src_img_path, dest_img_path, format=target_format)
            if success:
                # Copy caption if present
                if os.path.exists(src_txt_path):
                    shutil.copy2(src_txt_path, dest_txt_path)
                processed += 1
            else:
                failed += 1

        # Move staged files to final destination
        if in_place:
            # Clean original images from input_dir
            for f in image_files:
                img_p = os.path.join(input_dir, f)
                txt_p = os.path.join(input_dir, os.path.splitext(f)[0] + ".txt")
                if os.path.exists(img_p):
                    os.remove(img_p)
                if os.path.exists(txt_p):
                    os.remove(txt_p)

        for item in os.listdir(temp_staging):
            src_item = os.path.join(temp_staging, item)
            dst_item = os.path.join(target_dir, item)
            if os.path.exists(dst_item):
                os.remove(dst_item)
            shutil.move(src_item, dst_item)

        shutil.rmtree(temp_staging, ignore_errors=True)

        console.print(f"[bold green]✓ Normalization complete![/bold green] Processed: [bold green]{processed}[/bold green], Failed: [bold red]{failed}[/bold red]")
        return {
            "processed_count": processed,
            "failed_count": failed,
            "target_dir": target_dir
        }
