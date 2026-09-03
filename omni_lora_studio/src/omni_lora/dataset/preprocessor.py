import os
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageOps
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jfif", ".tiff", ".avif"}

class DatasetPreprocessor:
    """Tự động kiểm tra, sửa lỗi ảnh, giải nén ZIP, chuẩn hóa EXIF và đưa ảnh về RGB tiêu chuẩn."""

    @classmethod
    def sanitize_image(cls, input_path: str, output_path: str) -> bool:
        """Tải, xoay theo EXIF, chuẩn hóa RGBA -> RGB với nền trắng và lưu ảnh sạch."""
        try:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img)

                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[-1])
                    img = bg
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                img.save(output_path, "JPEG", quality=98, optimize=True)
                return True
        except Exception as e:
            logger.warning(f"Bỏ qua ảnh lỗi {input_path}: {e}")
            return False

    @classmethod
    def extract_zip_if_needed(cls, target_dir: str) -> None:
        """Tự động tìm kiếm và giải nén toàn bộ file ZIP trong thư mục dataset."""
        dir_path = Path(target_dir)
        for zip_file in dir_path.glob("*.zip"):
            console.print(f"[cyan]📦 Phát hiện file nén [yellow]{zip_file.name}[/yellow], đang tự động giải nén...[/cyan]")
            try:
                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(dir_path)
                console.print(f"[green]✓ Đã giải nén thành công {zip_file.name}![/green]")
            except Exception as e:
                logger.error(f"Lỗi giải nén {zip_file}: {e}")

    @classmethod
    def prepare_clean_dataset(cls, source_dir: str, clean_dir: str) -> List[str]:
        """Duyệt toàn bộ thư mục, khử file rác và xuất ra danh sách ảnh sạch kèm file txt tương ứng."""
        cls.extract_zip_if_needed(source_dir)
        source_path = Path(source_dir)
        clean_path = Path(clean_dir)
        clean_path.mkdir(parents=True, exist_ok=True)

        valid_images = []
        for file in sorted(source_path.rglob("*")):
            if file.suffix.lower() in SUPPORTED_EXTENSIONS and not file.name.startswith("."):
                out_img_path = clean_path / f"{file.stem}.jpg"
                if cls.sanitize_image(str(file), str(out_img_path)):
                    valid_images.append(str(out_img_path))
                    
                    # Đồng bộ file caption txt nếu đã có sẵn
                    caption_file = file.with_suffix(".txt")
                    if caption_file.exists():
                        shutil.copy2(caption_file, clean_path / f"{file.stem}.txt")

        console.print(f"[bold green]✨ Đã chuẩn hóa thành công {len(valid_images)} ảnh đạt chuẩn RGB 100%![/bold green]")
        return valid_images
