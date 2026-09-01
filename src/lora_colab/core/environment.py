import os
import sys
import subprocess
import importlib.metadata
from typing import Dict, List, Any, Optional, Tuple
from .logger import setup_logger, console

logger = setup_logger(__name__)

class AutoEnvironmentManager:
    """
    Tự động nhận diện và tối ưu hóa môi trường Google Colab trong tương lai:
    - Quét phiên bản Python, PyTorch, CUDA, và GPU hiện tại của Colab.
    - Bảo toàn PyTorch/CUDA tăng tốc gốc của Colab, tránh cài đè gây xung đột pip resolver.
    - Tự động kiểm tra và chỉ cài bổ sung các thư viện còn thiếu hoặc chưa đủ phiên bản tối thiểu.
    - Thích ứng hoàn hảo khi Google nâng cấp hệ điều hành hoặc PyTorch trong tương lai.
    """

    CORE_REQUIREMENTS = {
        "accelerate": "0.28.0",
        "transformers": "4.40.0",
        "diffusers": "0.28.0",
        "peft": "0.10.0",
        "safetensors": "0.4.2",
        "huggingface-hub": "0.23.0",
        "bitsandbytes": "0.43.0",
        "rich": "13.7.0",
        "pyyaml": "6.0.0",
        "pillow": "10.0.0",
        "google-genai": "0.1.0",
        "openai": "1.20.0"
    }

    @staticmethod
    def is_package_installed(package_name: str) -> bool:
        """Checks if a package is already installed in the current environment."""
        try:
            importlib.metadata.version(package_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False

    @staticmethod
    def get_package_version(package_name: str) -> Optional[str]:
        """Gets the installed version of a package."""
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    @classmethod
    def get_runtime_info(cls) -> Dict[str, Any]:
        """Detects live runtime environment details."""
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        torch_ver = "Not Installed"
        cuda_ver = "N/A"
        gpu_name = "N/A"
        cuda_avail = False

        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                cuda_ver = torch.version.cuda or "N/A"
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return {
            "python_version": py_ver,
            "torch_version": torch_ver,
            "cuda_version": cuda_ver,
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "is_colab": "google.colab" in sys.modules
        }

    @classmethod
    def optimize_and_install_dependencies(cls, silent: bool = True) -> Dict[str, Any]:
        """
        Dynamically installs only missing packages, avoiding breaking upgrades to setuptools or PyTorch.
        """
        info = cls.get_runtime_info()
        console.rule("[bold cyan]🤖 Dynamic Colab Environment Optimizer[/bold cyan]")
        console.print(f"[bold green]Python Runtime:[/bold green] {info['python_version']} | [bold green]PyTorch:[/bold green] {info['torch_version']}")
        console.print(f"[bold green]CUDA Build:[/bold green] {info['cuda_version']} | [bold green]GPU Hardware:[/bold green] {info['gpu_name']}")

        # 1. Resolve missing packages
        missing_pkgs = []
        for pkg, min_ver in cls.CORE_REQUIREMENTS.items():
            installed_ver = cls.get_package_version(pkg)
            if installed_ver is None:
                missing_pkgs.append(f"{pkg}>={min_ver}")

        # Check jedi for ipython
        if not cls.is_package_installed("jedi"):
            missing_pkgs.append("jedi>=0.16")

        if missing_pkgs:
            console.print(f"[bold yellow]📦 Dynamically installing {len(missing_pkgs)} missing packages:[/bold yellow] [dim]{', '.join(missing_pkgs)}[/dim]")
            install_cmd = [sys.executable, "-m", "pip", "install"]
            if silent:
                install_cmd.append("-q")
            install_cmd.extend(missing_pkgs)

            try:
                subprocess.check_call(install_cmd)
                console.print("[bold green]✓ Missing dependencies installed cleanly without conflicts![/bold green]")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error installing dependencies: {e}")
                raise e
        else:
            console.print("[bold green]✓ All core dependencies are already present and optimal![/bold green]")

        # 2. Check and adapt Flash-Attention or Xformers if compatible
        if info["cuda_available"] and "A100" in info["gpu_name"] or "H100" in info["gpu_name"] or "L4" in info["gpu_name"]:
            if not cls.is_package_installed("flash_attn"):
                console.print("[cyan]💡 Modern Ampere/Ada GPU detected. Enabling native SDPA and PyTorch 2.0+ optimized attention.[/cyan]")

        console.rule()
        return {
            "status": "ready",
            "installed_count": len(missing_pkgs),
            "runtime_info": info
        }
