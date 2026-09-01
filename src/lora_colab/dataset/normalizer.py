import os
import shutil
import tempfile
from PIL import Image, ImageOps
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif", ".tiff"}

class DatasetNormalizer:
    """
    Normalizes dataset images and associated caption files:
    - Auto-renames files to standard format {prefix}_{index:04d}.{ext} (optional, default: True)
    - Converts corrupted/non-RGB modes (RGBA, P, CMYK) to clean standard RGB
    - Synchronizes matching .txt caption files
    - Uses local /tmp staging to prevent Google Drive FUSE [Errno 107] Transport endpoint disconnects
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
    def resolve_dataset_dir(cls, base_dir: str) -> str:
        """
        If base_dir contains direct images, returns base_dir.
        If base_dir has no direct images but contains subdirectories with images (e.g. 02_character/Mai_girl),
        returns the subfolder containing the images.
        """
        if not os.path.exists(base_dir):
            return base_dir
            
        direct_images = [
            f for f in os.listdir(base_dir)
            if os.path.isfile(os.path.join(base_dir, f)) and os.path.splitext(f)[-1].lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if direct_images:
            return base_dir
            
        subdirs = [
            os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir))
            if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith((".", "_"))
        ]
        for sub in subdirs:
            sub_images = [
                f for f in os.listdir(sub)
                if os.path.isfile(os.path.join(sub, f)) and os.path.splitext(f)[-1].lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if sub_images:
                console.print(f"[cyan]📁 Tự động phát hiện ảnh trong thư mục con: [bold]{sub}[/bold] ({len(sub_images)} ảnh)[/cyan]")
                return sub
                
        return base_dir

    @classmethod
    def normalize_folder(
        cls,
        input_dir: str,
        output_dir: Optional[str] = None,
        prefix: str = "img",
        target_format: str = "PNG",
        start_index: int = 1,
        enable_renaming: bool = True
    ) -> Dict[str, Any]:
        """
        Normalizes and sanitizes all images and captions in input_dir using local /tmp staging.
        If enable_renaming=False, preserves original file names without renaming.
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Auto-resolve subfolder if input_dir contains a subfolder with images
        input_dir = cls.resolve_dataset_dir(input_dir)

        in_place = output_dir is None or os.path.abspath(input_dir) == os.path.abspath(output_dir)
        target_dir = input_dir if in_place else output_dir
        os.makedirs(target_dir, exist_ok=True)

        # Collect valid image files
        raw_files = sorted(os.listdir(input_dir))
        image_files = [
            f for f in raw_files 
            if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[-1].lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        if not image_files:
            logger.warning(f"No valid images found in {input_dir}")
            return {"processed_count": 0, "failed_count": 0, "target_dir": target_dir}

        mode_desc = f"với tiền tố '{prefix}'" if enable_renaming else "(Giữ nguyên tên file gốc)"
        console.print(f"[bold cyan]🔄 Chuẩn hóa {len(image_files)} ảnh trong '{input_dir}' {mode_desc}...[/bold cyan]")

        # Use local /tmp directory for staging to ensure 100% stable I/O
        with tempfile.TemporaryDirectory(prefix="lora_norm_") as temp_staging:
            processed = 0
            failed = 0

            for i, filename in enumerate(tqdm(image_files, desc="Normalizing Dataset")):
                idx = start_index + i
                base_src, ext_src = os.path.splitext(filename)
                src_img_path = os.path.join(input_dir, filename)
                src_txt_path = os.path.join(input_dir, base_src + ".txt")

                out_ext = ".png" if target_format.upper() == "PNG" else ext_src.lower()
                
                if enable_renaming:
                    new_img_name = f"{prefix}_{idx:04d}{out_ext}"
                    new_txt_name = f"{prefix}_{idx:04d}.txt"
                else:
                    new_img_name = f"{base_src}{out_ext}"
                    new_txt_name = f"{base_src}.txt"

                dest_img_path = os.path.join(temp_staging, new_img_name)
                dest_txt_path = os.path.join(temp_staging, new_txt_name)

                # Process Image via local staging
                success = cls.sanitize_image(src_img_path, dest_img_path, format=target_format)
                if success:
                    # Copy caption if present
                    if os.path.exists(src_txt_path):
                        shutil.copy2(src_txt_path, dest_txt_path)
                    processed += 1
                else:
                    failed += 1

            if processed > 0:
                # If in-place, safely clean original files
                if in_place:
                    for f in image_files:
                        img_p = os.path.join(input_dir, f)
                        txt_p = os.path.join(input_dir, os.path.splitext(f)[0] + ".txt")
                        try:
                            if os.path.exists(img_p):
                                os.remove(img_p)
                            if os.path.exists(txt_p) and enable_renaming:
                                os.remove(txt_p)
                        except Exception as rm_err:
                            logger.warning(f"Warning during file cleanup: {rm_err}")

                # Copy all cleanly normalized files from /tmp to final target in Google Drive
                for item in os.listdir(temp_staging):
                    src_item = os.path.join(temp_staging, item)
                    dst_item = os.path.join(target_dir, item)
                    shutil.copy2(src_item, dst_item)

        console.print(f"[bold green]✓ Chuẩn hóa hoàn tất an toàn![/bold green] Đã xử lý: [bold green]{processed}[/bold green], Lỗi: [bold red]{failed}[/bold red]")
        return {
            "processed_count": processed,
            "failed_count": failed,
            "target_dir": target_dir
        }
