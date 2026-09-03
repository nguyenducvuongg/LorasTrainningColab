import time
from rich.table import Table
from rich.panel import Panel
from ..core.logger import console

class LiveMetricsDashboard:
    """Hiển thị bảng trạng thái và chỉ số huấn luyện trực quan theo thời gian thực."""

    @classmethod
    def render_summary(
        cls, 
        model_name: str, 
        current_epoch: int, 
        total_epochs: int, 
        loss: float, 
        likeness_score: float, 
        vram_used_gb: float
    ):
        table = Table(title="📊 OmniLoRA Studio - Trạng Thái Huấn Luyện Trực Tiếp", border_style="cyan")
        table.add_column("Chỉ Số", style="bold yellow")
        table.add_column("Giá Trị Hiện Tại", style="bold green")

        table.add_row("Mô Hình Nền", model_name)
        table.add_row("Tiến Độ Epoch", f"{current_epoch} / {total_epochs}")
        table.add_row("Training Loss", f"{loss:.4f}")
        table.add_row("Điểm Likeness (100% Target)", f"[bold magenta]{likeness_score:.1f}%[/bold magenta]")
        table.add_row("VRAM GPU Đang Dùng", f"{vram_used_gb:.1f} GB")

        console.print(table)
