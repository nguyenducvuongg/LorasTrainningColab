import shutil
from pathlib import Path
from typing import Dict, List, Optional
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class BestCheckpointSelector:
    """Theo dõi điểm Likeness qua từng Epoch và tự động lưu phiên bản LoRA tốt nhất."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.history: Dict[str, float] = {}
        self.best_checkpoint: Optional[str] = None
        self.best_score: float = -1.0

    def record_score(self, checkpoint_path: str, likeness_score: float) -> bool:
        """Ghi nhận điểm số, trả về True nếu đây là kỷ lục mới."""
        self.history[checkpoint_path] = likeness_score
        if likeness_score > self.best_score:
            self.best_score = likeness_score
            self.best_checkpoint = checkpoint_path

            # Tạo bản sao checkpoint tốt nhất
            best_target = self.output_dir / "BEST_100_LIKENESS_MODEL.safetensors"
            try:
                shutil.copy2(checkpoint_path, best_target)
                console.print(f"[bold green]🏆 Kỷ lục mới! Điểm tương đồng: {likeness_score}%[/bold green]")
                console.print(f"[green]✓ Đã cập nhật checkpoint tốt nhất tại:[/green] [yellow]{best_target.name}[/yellow]")
            except Exception as e:
                logger.warning(f"Lỗi khi lưu best checkpoint: {e}")
            return True
        return False
