import os
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class ModelCacheManager:
    """Tải và lưu trữ cache cục bộ các mô hình nền tảng từ HuggingFace, Civitai hoặc URL trực tiếp."""

    @classmethod
    def download_file(cls, url: str, destination_path: str, token: Optional[str] = None) -> bool:
        dest = Path(destination_path)
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            console.print(f"[green]✓ Mô hình đã tồn tại sẵn trong cache: {dest.name}[/green]")
            return True

        dest.parent.mkdir(parents=True, exist_ok=True)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        console.print(f"[cyan]📥 Đang tải mô hình từ URL vào cache: {dest.name}...[/cyan]")
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(dest, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            console.print(f"[bold green]✅ Tải hoàn tất: {dest.name}[/bold green]")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình {url}: {e}")
            if dest.exists():
                dest.unlink()
            return False
