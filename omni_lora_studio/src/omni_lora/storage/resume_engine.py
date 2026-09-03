import re
from pathlib import Path
from typing import Optional, Tuple
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class ResumeEngine:
    """Tự động phát hiện checkpoint gần nhất và khôi phục quá trình huấn luyện khi Colab bị đứt kết nối."""

    @classmethod
    def find_latest_checkpoint(cls, output_dir: str) -> Optional[Tuple[str, int]]:
        out_path = Path(output_dir)
        if not out_path.exists():
            return None

        candidates = list(out_path.glob("*.safetensors")) + list(out_path.glob("*.pt"))
        if not candidates:
            return None

        # Sắp xếp theo mtime (thời gian sửa đổi mới nhất)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for p in candidates:
            match = re.search(r"[-_](?:epoch|step|e|s)[-_]?(\d+)", p.stem, re.IGNORECASE)
            if match:
                step_or_epoch = int(match.group(1))
                console.print(f"[bold green]🔄 Đã tìm thấy điểm dừng trước đó:[/bold green] [yellow]{p.name}[/yellow] (vòng {step_or_epoch})")
                return (str(p), step_or_epoch)

        return (str(candidates[0]), 0)
