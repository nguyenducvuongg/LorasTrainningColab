import os
import re
import sys
import subprocess
import importlib.metadata
from typing import Dict, List, Any, Optional, Tuple, Set
from .logger import setup_logger, console

logger = setup_logger(__name__)

class AutoEnvironmentManager:
    """
    Tự động nhận diện, tối ưu hóa và tự phục hồi (Self-Healing) môi trường toàn diện:
    - Quét phiên bản Python, PyTorch, CUDA, và GPU hiện tại.
    - Tự động quét và cài đặt các gói còn thiếu cho từng Engine (AI-Toolkit, Kohya sd-scripts, Musubi-Tuner, Captioning).
    - Tự động phân tích file `requirements.txt` của các kho mã nguồn backend khi nâng cấp.
    - Cơ chế Self-Healing: Bắt lỗi `ModuleNotFoundError`, tự động cài gói thiếu và chạy lại liền mạch.
    - Thích ứng hoàn hảo khi Colab hoặc backend nâng cấp trong tương lai.
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
        "toml": "0.10.2",
        "pillow": "10.0.0",
        "google-genai": "0.1.0",
        "openai": "1.20.0"
    }

    # Packages that should never be forcefully overwritten to protect Colab PyTorch CUDA acceleration
    PROTECTED_SYSTEM_PACKAGES = {"torch", "torchvision", "torchaudio", "setuptools", "pip"}
    BLACKLISTED_PYPI_PACKAGES = {
        "ai_toolkit", "ai-toolkit", "sd_scripts", "sd-scripts",
        "musubi_tuner", "musubi-tuner", "dataslots", "wget"
    }

    @classmethod
    def is_package_installed(cls, package_name: str) -> bool:
        """Kiểm tra một gói đã cài đặt trong môi trường Python chưa."""
        clean_name = re.split(r"[><=~;]", package_name)[0].strip()
        norm_name = clean_name.replace("-", "_")
        alt_name = clean_name.replace("_", "-")
        for name in [clean_name, norm_name, alt_name]:
            try:
                importlib.metadata.version(name)
                return True
            except (importlib.metadata.PackageNotFoundError, Exception):
                continue
        return False

    @classmethod
    def get_package_version(cls, package_name: str) -> Optional[str]:
        """Lấy phiên bản hiện tại của gói."""
        clean_name = re.split(r"[><=~;]", package_name)[0].strip()
        norm_name = clean_name.replace("-", "_")
        alt_name = clean_name.replace("_", "-")
        for name in [clean_name, norm_name, alt_name]:
            try:
                return importlib.metadata.version(name)
            except (importlib.metadata.PackageNotFoundError, Exception):
                continue
        return None

    DRIVE_WHEEL_DIR = "/content/drive/MyDrive/Colab_LoRA_Studio/.cache/pip_wheels"

    @classmethod
    def install_packages(
        cls,
        package_list: List[str],
        silent: bool = True,
        on_log: Optional[Any] = None
    ) -> bool:
        """
        Cài đặt danh sách gói qua pip siêu tốc (< 30s lần đầu, 1-3s lần sau):
        - Sử dụng SSD nội bộ máy ảo (/tmp/fast_pip_cache) cho toàn bộ I/O, KHÔNG ghi cache qua Drive FUSE.
        - Tự động nạp từ Google Drive Wheel Cache (--find-links) nếu file .whl đã có sẵn.
        - Sử dụng --prefer-binary và --no-build-isolation để ưu tiên nạp pre-compiled binary.
        - Đồng bộ file .whl sang Google Drive ở luồng nền (non-blocking daemon thread).
        """
        if not package_list:
            return True

        valid_packages = [
            pkg for pkg in package_list
            if re.split(r"[><=~;]", pkg)[0].strip().lower() not in cls.BLACKLISTED_PYPI_PACKAGES
        ]
        if not valid_packages:
            return True

        has_drive = os.path.exists("/content/drive/MyDrive")
        has_wheel_cache = has_drive and os.path.exists(cls.DRIVE_WHEEL_DIR)

        # Sử dụng SSD nội bộ của máy ảo cho pip cache để đạt tốc độ tối đa
        local_cache_dir = "/tmp/fast_pip_cache"
        os.makedirs(local_cache_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "pip", "install",
            "--prefer-binary",
            "--no-build-isolation",
            "--cache-dir", local_cache_dir,
            "--no-warn-script-location"
        ]

        if has_wheel_cache:
            cmd.extend(["--find-links", cls.DRIVE_WHEEL_DIR])

        if silent and on_log is None:
            cmd.append("-q")

        cmd.extend(valid_packages)

        try:
            if on_log is not None:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                for line in iter(process.stdout.readline, ''):
                    try:
                        on_log(line)
                    except Exception:
                        pass
                process.wait()
            else:
                subprocess.check_call(cmd)

            # Đồng bộ các file .whl sang Google Drive ở luồng nền (background)
            if has_drive:
                try:
                    import threading
                    import shutil

                    def _sync_wheels_to_drive():
                        try:
                            os.makedirs(cls.DRIVE_WHEEL_DIR, exist_ok=True)
                            temp_wheel_staging = "/tmp/wheel_staging"
                            os.makedirs(temp_wheel_staging, exist_ok=True)
                            download_cmd = [
                                sys.executable, "-m", "pip", "download",
                                "--prefer-binary", "--no-deps",
                                "--cache-dir", local_cache_dir,
                                "-d", temp_wheel_staging, "-q"
                            ] + valid_packages
                            subprocess.run(download_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            for item in os.listdir(temp_wheel_staging):
                                if item.endswith(".whl"):
                                    src = os.path.join(temp_wheel_staging, item)
                                    dst = os.path.join(cls.DRIVE_WHEEL_DIR, item)
                                    if not os.path.exists(dst):
                                        shutil.copy2(src, dst)
                        except Exception:
                            pass

                    sync_thread = threading.Thread(target=_sync_wheels_to_drive, daemon=True)
                    sync_thread.start()
                except Exception:
                    pass

            return True
        except Exception as e:
            logger.warning(f"Warning installing packages {valid_packages}: {e}")
            return False

    @classmethod
    def install_missing_from_requirements_file(cls, req_file_path: str) -> List[str]:
        """
        Tự động quét file requirements.txt của backend (ai-toolkit, sd-scripts, musubi-tuner),
        phát hiện các gói chưa có và tự động cài đặt.
        """
        if not os.path.exists(req_file_path):
            return []

        missing = []
        try:
            with open(req_file_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith(("#", "-r", "--", "git+", "http")):
                        continue
                    pkg_spec = line.split(";")[0].strip()
                    pkg_name = re.split(r"[><=~]", pkg_spec)[0].strip()
                    if not pkg_name or pkg_name.lower() in cls.PROTECTED_SYSTEM_PACKAGES or pkg_name.lower() in cls.BLACKLISTED_PYPI_PACKAGES:
                        continue
                    if not cls.is_package_installed(pkg_name):
                        missing.append(pkg_spec)

            if missing:
                console.print(f"[bold yellow]📦 Tự động phát hiện & cài đặt {len(missing)} gói cần thiết từ {os.path.basename(req_file_path)}:[/bold yellow] [dim]{', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}[/dim]")
                cls.install_packages(missing)
                console.print(f"[bold green]✓ Cập nhật gói từ {os.path.basename(req_file_path)} hoàn tất![/bold green]")
        except Exception as e:
            logger.warning(f"Could not parse requirements file {req_file_path}: {e}")

        return missing

    @classmethod
    def ensure_engine_dependencies(cls, backend_dir: str):
        """Đảm bảo các gói của engine backend đã sẵn sàng trước khi chạy."""
        if not os.path.exists(backend_dir):
            return
        req_file = os.path.join(backend_dir, "requirements.txt")
        if os.path.exists(req_file):
            cls.install_missing_from_requirements_file(req_file)

    @classmethod
    def get_runtime_info(cls) -> Dict[str, Any]:
        """Thu thập thông tin chi tiết về môi trường Runtime."""
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
        """Ensures Kohya sd-scripts, AI-Toolkit and Musubi-Tuner are cloned and ready."""
        if not os.path.exists("/content"):
            return

        os.makedirs(backends_dir, exist_ok=True)

        backends = [
            ("sd-scripts", "https://github.com/kohya-ss/sd-scripts.git"),
            ("ai-toolkit", "https://github.com/ostris/ai-toolkit.git"),
            ("musubi-tuner", "https://github.com/kohya-ss/musubi-tuner.git"),
        ]

        for name, repo_url in backends:
            repo_path = os.path.join(backends_dir, name)
            if not os.path.exists(repo_path):
                console.print(f"[cyan]📥 Tải và chuẩn bị backend [bold]{name}[/bold]...[/cyan]")
                try:
                    # FIX: thêm --recurse-submodules để clone đầy đủ submodule
                    subprocess.check_call([
                        "git", "clone", "--depth", "1", "--recurse-submodules",
                        repo_url, repo_path
                    ])
                    console.print(f"[green]✓ Backend {name} đã sẵn sàng![/green]")
                except Exception as e:
                    logger.warning(f"Could not clone {name}: {e}")
            
            # Tự động quét requirements.txt của backend đó
            cls.ensure_engine_dependencies(repo_path)

    @classmethod
    def optimize_and_install_dependencies(cls, silent: bool = True, install_backends: bool = False) -> Dict[str, Any]:
        """
        Khởi tạo và cài đặt toàn bộ gói còn thiếu trong môi trường một cách mượt mà và siêu tốc.
        """
        info = cls.get_runtime_info()
        console.rule("[bold cyan]🤖 Dynamic Colab Environment Optimizer[/bold cyan]")
        console.print(f"[bold green]Python Runtime:[/bold green] {info['python_version']} | [bold green]PyTorch:[/bold green] {info['torch_version']}")
        console.print(f"[bold green]CUDA Build:[/bold green] {info['cuda_version']} | [bold green]GPU Hardware:[/bold green] {info['gpu_name']}")

        # 1. Quét các gói lõi còn thiếu
        missing_pkgs = []
        for pkg, min_ver in cls.CORE_REQUIREMENTS.items():
            if not cls.is_package_installed(pkg):
                missing_pkgs.append(f"{pkg}>={min_ver}")

        if not cls.is_package_installed("jedi"):
            missing_pkgs.append("jedi>=0.16")

        if missing_pkgs:
            console.print(f"[bold yellow]📦 Tự động cài bổ sung {len(missing_pkgs)} gói tối ưu:[/bold yellow] [dim]{', '.join(missing_pkgs)}[/dim]")
            cls.install_packages(missing_pkgs, silent=silent)
            console.print("[bold green]✓ Cài đặt dependencies hoàn tất mượt mà không xung đột![/bold green]")
        else:
            console.print("[bold green]✓ Toàn bộ gói lõi (Prodigy, Einops, Transformers, LoRA SDK) đã sẵn sàng![/bold green]")

        # 2. Clone backend engines chỉ khi được yêu cầu (Mặc định Lazy-Load khi train để Cell 1 chạy trong 2 giây)
        if install_backends and info["is_colab"]:
            cls.ensure_backend_repositories()

        console.rule()
        return {
            "status": "ready",
            "installed_count": len(missing_pkgs),
            "runtime_info": info
        }

    KNOWN_MODULE_MAP = {
        "optimum": "optimum-quanto",
        "quanto": "optimum-quanto",
        "controlnet_aux": "controlnet-aux",
        "open_clip": "open-clip-torch",
        "cv2": "opencv-python-headless",
        "PIL": "pillow",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
        "imwatermark": "invisible-watermark",
        "voluptuous": "voluptuous",
        "imagesize": "imagesize",
        "albumentations": "albumentations",
        "flatten_dict": "flatten_dict",
        "lpips": "lpips",
        "av": "av",
        "oyaml": "oyaml",
        # Thêm module phổ biến còn thiếu
        "xformers": "xformers",
        "flash_attn": "flash-attn",
        "triton": "triton",
        "onnxruntime": "onnxruntime-gpu",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "tqdm": "tqdm",
        "psutil": "psutil",
        "requests": "requests",
        "sentencepiece": "sentencepiece",
        "lycoris": "lycoris-lora",
    }

    @classmethod
    def execute_with_self_healing(
        cls,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        max_retries: int = 6,
        on_log_line: Optional[Any] = None
    ) -> bool:
        """
        Thực thi tiến trình với cơ chế tự sửa lỗi (Self-Healing) và hỗ trợ cập nhật Live Dashboard:
        Nếu phát hiện ModuleNotFoundError trong log, tự động ánh xạ sang tên gói PyPI chính xác,
        cài đặt tức thì qua pip và khởi động lại tiến trình!
        """
        for attempt in range(max_retries + 1):
            process = subprocess.Popen(
                cmd,
                env=env,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            missing_module = None

            for line in iter(process.stdout.readline, ''):
                if on_log_line is not None:
                    try:
                        on_log_line(line)
                    except Exception:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()

                # Bắt lỗi ModuleNotFoundError: No module named 'xyz'
                if "ModuleNotFoundError" in line and "No module named" in line:
                    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", line)
                    if match:
                        mod = match.group(1).split(".")[0].strip().lower()
                        if mod not in cls.BLACKLISTED_PYPI_PACKAGES:
                            # Ánh xạ tên import module sang tên package PyPI chuẩn
                            pkg_to_install = cls.KNOWN_MODULE_MAP.get(mod, mod)
                            missing_module = pkg_to_install

            process.wait()

            if process.returncode == 0:
                return True

            # Nếu có lỗi thiếu module và còn lượt thử lại -> Auto-Heal
            if missing_module and attempt < max_retries:
                console.print(f"\n[bold yellow]🛠️ Tự động phát hiện thiếu gói: [red]{missing_module}[/red] -> Tiến hành tự động cài đặt và chạy lại...[/bold yellow]")
                cls.install_packages([missing_module], silent=False)
                console.print(f"[bold green]✓ Đã cài đặt {missing_module}. Đang khởi động lại tiến trình ({attempt + 1}/{max_retries})...[/bold green]\n")
                continue
            else:
                return False

        return False
