import os
import sys
import subprocess
import importlib.metadata
from typing import Dict, List, Any, Optional, Tuple
from .logger import setup_logger, console

logger = setup_logger(__name__)

class AutoEnvironmentManager:
    """
    Tự động nhận diện và tối ưu hóa môi trường Google Colab toàn diện:
    - Quét phiên bản Python, PyTorch, CUDA, và GPU hiện tại của Colab.
    - Bảo toàn PyTorch/CUDA tăng tốc gốc của Colab, tránh xung đột pip resolver.
    - Tự động kiểm tra và cài đặt đầy đủ tất cả gói cần thiết (Prodigy, Einops, SentencePiece, LyCORIS, Transformers...).
    - Tự động chuẩn bị các backend huấn luyện (Kohya sd-scripts, AI-Toolkit) khi cần.
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
        "prodigyopt": "1.0",
        "dadaptation": "3.1",
        "einops": "0.7.0",
        "sentencepiece": "0.2.0",
        "protobuf": "3.20.0",
        "lycoris-lora": "2.2.0",
        "rich": "13.7.0",
        "pyyaml": "6.0.0",
        "oyaml": "1.0",
        "albumentations": "1.4.0",
        "flatten_dict": "0.4.0",
        "invisible-watermark": "0.2.0",
        "k-diffusion": "0.1.0",
        "open-clip-torch": "2.24.0",
        "optimum-quanto": "0.2.0",
        "toml": "0.10.2",
        "pillow": "10.0.0",
        "google-genai": "0.1.0",
        "openai": "1.20.0"
    }

    @staticmethod
    def is_package_installed(package_name: str) -> bool:
        """Checks if a package is already installed in the current environment."""
        # Handle normalized naming (e.g. lycoris-lora vs lycoris_lora)
        norm_name = package_name.replace("-", "_")
        alt_name = package_name.replace("_", "-")
        for name in [package_name, norm_name, alt_name]:
            try:
                importlib.metadata.version(name)
                return True
            except importlib.metadata.PackageNotFoundError:
                continue
        return False

    @staticmethod
    def get_package_version(package_name: str) -> Optional[str]:
        """Gets the installed version of a package."""
        norm_name = package_name.replace("-", "_")
        alt_name = package_name.replace("_", "-")
        for name in [package_name, norm_name, alt_name]:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
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
            "is_colab": "google.colab" in sys.modules or os.path.exists("/content")
        }

    @classmethod
    def ensure_backend_repositories(cls, backends_dir: str = "/content/backends"):
        """Ensures Kohya sd-scripts and AI-Toolkit are cloned and available."""
        if not os.path.exists("/content"):
            return  # Not in Colab environment

        os.makedirs(backends_dir, exist_ok=True)

        # 1. Kohya sd-scripts
        kohya_path = os.path.join(backends_dir, "sd-scripts")
        if not os.path.exists(kohya_path):
            console.print("[cyan]📥 Chuẩn bị backend Kohya sd-scripts...[/cyan]")
            try:
                subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/kohya-ss/sd-scripts.git", kohya_path])
                console.print("[green]✓ Kohya sd-scripts đã sẵn sàng![/green]")
            except Exception as e:
                logger.warning(f"Could not clone sd-scripts: {e}")

        # 2. AI-Toolkit
        aitoolkit_path = os.path.join(backends_dir, "ai-toolkit")
        if not os.path.exists(aitoolkit_path):
            console.print("[cyan]📥 Chuẩn bị backend AI-Toolkit cho Flux.1...[/cyan]")
            try:
                subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/ostris/ai-toolkit.git", aitoolkit_path])
                console.print("[green]✓ AI-Toolkit đã sẵn sàng![/green]")
            except Exception as e:
                logger.warning(f"Could not clone ai-toolkit: {e}")

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
            if not cls.is_package_installed(pkg):
                missing_pkgs.append(f"{pkg}>={min_ver}")

        # Check jedi for ipython
        if not cls.is_package_installed("jedi"):
            missing_pkgs.append("jedi>=0.16")

        if missing_pkgs:
            console.print(f"[bold yellow]📦 Tự động cài bổ sung {len(missing_pkgs)} gói tối ưu:[/bold yellow] [dim]{', '.join(missing_pkgs)}[/dim]")
            install_cmd = [sys.executable, "-m", "pip", "install"]
            if silent:
                install_cmd.append("-q")
            install_cmd.extend(missing_pkgs)

            try:
                subprocess.check_call(install_cmd)
                console.print("[bold green]✓ Cài đặt dependencies hoàn tất mượt mà không xung đột![/bold green]")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error installing dependencies: {e}")
                raise e
        else:
            console.print("[bold green]✓ Toàn bộ gói lõi (Prodigy, Einops, Transformers, LoRA SDK) đã sẵn sàng![/bold green]")

        # 2. Clone backend engines if running in Colab
        if info["is_colab"]:
            cls.ensure_backend_repositories()

        console.rule()
        return {
            "status": "ready",
            "installed_count": len(missing_pkgs),
            "runtime_info": info
        }
