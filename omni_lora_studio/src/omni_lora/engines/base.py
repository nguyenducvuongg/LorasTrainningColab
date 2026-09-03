import abc
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..core.config import OmniConfig
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class BaseTrainer(abc.ABC):
    """Lớp trừu tượng cho tất cả các engine huấn luyện LoRA."""

    def __init__(self, config: OmniConfig):
        self.config = config

    @abc.abstractmethod
    def build_command(self) -> List[str]:
        """Xây dựng danh sách câu lệnh CLI thực thi huấn luyện."""
        pass

    def run_training(self) -> bool:
        """Thực thi tiến trình huấn luyện và truyền log thời gian thực."""
        cmd = self.build_command()
        cmd_str = " ".join(cmd)
        
        console.rule("[bold cyan]🚀 Khởi Động Tiến Trình Huấn Luyện LoRA[/bold cyan]")
        console.print(f"[dim]Lệnh: {cmd_str[:200]}...[/dim]\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                print(line, end="")

            process.wait()
            if process.returncode == 0:
                console.print("\n[bold green]🎉 Huấn luyện thành công hoàn tất 100%![/bold green]")
                return True
            else:
                console.print(f"\n[bold red]❌ Quá trình huấn luyện kết thúc với mã lỗi: {process.returncode}[/bold red]")
                return False
        except Exception as e:
            logger.error(f"Lỗi thực thi huấn luyện: {e}")
            return False
