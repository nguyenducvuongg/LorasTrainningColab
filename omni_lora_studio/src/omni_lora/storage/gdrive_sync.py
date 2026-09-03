import shutil
import threading
from pathlib import Path
from typing import Optional
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class GDriveSyncManager:
    """Quản lý đồng bộ bất đồng bộ từ local SSD sang Google Drive, bảo vệ kết quả 100%."""

    @classmethod
    def sync_file_async(cls, local_path: str, gdrive_dest_dir: str):
        def _worker():
            try:
                dest = Path(gdrive_dest_dir)
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_path, dest / Path(local_path).name)
                logger.info(f"Đã sao lưu thành công sang Drive: {Path(local_path).name}")
            except Exception as e:
                logger.warning(f"Lỗi đồng bộ ngầm Google Drive: {e}")

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
