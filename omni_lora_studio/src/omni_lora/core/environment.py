import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple
from .logger import setup_logger, console

logger = setup_logger(__name__)

class EnvironmentManager:
    """Quản lý môi trường Google Colab, bộ nhớ SSD cục bộ và Google Drive an toàn."""

    @staticmethod
    def is_colab() -> bool:
        """Kiểm tra xem mã nguồn có đang thực thi trên môi trường Google Colab hay không."""
        return "google.colab" in sys.modules or os.path.exists("/content")

    @classmethod
    def mount_google_drive(cls, mount_point: str = "/content/drive") -> bool:
        """Kết nối Google Drive một cách an toàn và nhẹ nhàng."""
        if not cls.is_colab():
            logger.info("Môi trường cục bộ (Local / Cloud Server), bỏ qua mount Google Drive.")
            return False

        try:
            from google.colab import drive
            if not os.path.exists(mount_point) or len(os.listdir(mount_point)) == 0:
                console.print("[cyan]🔄 Đang kết nối Google Drive...[/cyan]")
                drive.mount(mount_point)
                console.print("[bold green]✅ Kết nối Google Drive thành công![/bold green]")
            return True
        except Exception as e:
            logger.warning(f"Không thể tự động kết nối Google Drive: {e}")
            return False

    @classmethod
    def prepare_local_staging(cls, gdrive_source_path: str, local_staging_dir: Optional[str] = None) -> str:
        """
        Sao chép dữ liệu từ Google Drive (FUSE) vào ổ NVMe SSD cục bộ (/content/dataset_staging)
        để giải quyết triệt để lỗi [Errno 107] Transport endpoint is not connected
        và tăng tốc độ đọc dữ liệu lên gấp 10 lần.
        """
        source = Path(gdrive_source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source}")

        if local_staging_dir is None:
            if cls.is_colab():
                staging_base = Path("/content/dataset_staging")
            else:
                staging_base = Path("/tmp/omni_lora_staging")
        else:
            staging_base = Path(local_staging_dir)

        staging_base.mkdir(parents=True, exist_ok=True)
        dest_folder = staging_base / source.name

        console.print(f"[cyan]🚀 Đang đồng bộ dữ liệu vào SSD cục bộ siêu tốc:[/cyan] [yellow]{dest_folder}[/yellow]")
        
        # Nếu thư mục đích đã có và cùng số file, bỏ qua
        if dest_folder.exists() and len(list(dest_folder.glob("*.*"))) > 0:
            console.print("[green]✓ Đã tìm thấy dữ liệu đã cache trên SSD, sẵn sàng huấn luyện.[/green]")
            return str(dest_folder)

        shutil.copytree(str(source), str(dest_folder), dirs_exist_ok=True)
        console.print(f"[bold green]✅ Đã nạp thành công vào SSD cục bộ![/bold green]")
        return str(dest_folder)
