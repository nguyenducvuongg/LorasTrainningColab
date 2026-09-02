import os
import sys
from typing import Dict, List, Any
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class GDriveWorkspaceManager:
    """
    Manages Google Drive workspace initialization, scans existing directories,
    prevents overwrites, and organizes models, datasets, and outputs safely.
    """
    
    DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/Colab_LoRA_Studio"
    LOCAL_FALLBACK_ROOT = os.path.expanduser("~/Colab_LoRA_Studio")

    STANDARD_DIRECTORIES = [
        # Models
        "models/flux",
        "models/flux_kontext",
        "models/sdxl",
        "models/krea",
        "models/sd35",
        "models/sd15",
        "models/qwen",
        "models/text_encoders",
        "models/vae",
        "models/captioners",
        # Datasets
        "datasets/01_face",
        "datasets/02_character",
        "datasets/03_style",
        "datasets/04_skin_enhancement",
        "datasets/05_control",
        "datasets/06_custom",
        "datasets/raw_uploads",
        # Outputs
        "outputs/checkpoints",
        "outputs/samples",
        "outputs/final_loras",
        "outputs/logs",
        # Configs
        "configs",
        # Persistent Cache (Lưu file .whl để nạp tức thì trong 1 giây)
        ".cache/pip_wheels",
        ".cache/pip_cache"
    ]

    @classmethod
    def is_colab(cls) -> bool:
        """Check if executing inside Google Colab."""
        return "google.colab" in sys.modules or os.path.exists("/content")

    @classmethod
    def ensure_drive_connected(cls, test_path: str = "/content/drive/MyDrive") -> bool:
        """Verifies Google Drive FUSE mount is active, auto-recovering if Errno 107 disconnected."""
        if not cls.is_colab():
            return True
        try:
            if os.path.exists(test_path):
                os.listdir(test_path)
                return True
        except Exception as e:
            logger.warning(f"Google Drive FUSE connection lost ({e}). Auto-remounting...")
            try:
                from google.colab import drive
                drive.flush_and_unmount()
                drive.mount('/content/drive', force_remount=True)
                logger.info("✓ Google Drive successfully reconnected!")
                return True
            except Exception as remount_err:
                logger.error(f"Failed to auto-remount Google Drive: {remount_err}")
                return False
        return True

    @classmethod
    def mount_google_drive(cls) -> bool:
        """Mounts Google Drive immediately if inside Colab."""
        if cls.is_colab():
            if os.path.exists("/content/drive/MyDrive"):
                console.print("[bold green]✓ Google Drive đã được kết nối sẵn tại /content/drive[/bold green]")
                return True
            try:
                from google.colab import drive
                console.print("[bold cyan]🚀 Đang yêu cầu kết nối Google Drive...[/bold cyan]")
                drive.mount('/content/drive', force_remount=False)
                console.print("[bold green]✓ Google Drive đã kết nối thành công tại /content/drive![/bold green]")
                return True
            except Exception as e:
                logger.error(f"Failed to mount Google Drive: {e}")
                return False
        else:
            logger.info(f"Running outside Google Colab. Using local workspace: {cls.LOCAL_FALLBACK_ROOT}")
            return True

    @classmethod
    def init_workspace(cls, custom_root: str = None) -> Dict[str, str]:
        """
        Initializes workspace directory structure without overwriting existing files.
        Scans and returns the resolved paths.
        """
        root_dir = custom_root or (cls.DEFAULT_DRIVE_ROOT if cls.is_colab() else cls.LOCAL_FALLBACK_ROOT)
        
        created_count = 0
        existing_count = 0
        path_map: Dict[str, str] = {"root": root_dir}

        for sub_dir in cls.STANDARD_DIRECTORIES:
            full_path = os.path.join(root_dir, sub_dir)
            path_map[sub_dir.replace("/", "_")] = full_path
            
            if os.path.exists(full_path):
                existing_count += 1
            else:
                try:
                    os.makedirs(full_path, exist_ok=True)
                    created_count += 1
                except Exception as e:
                    logger.warning(f"Could not create folder {full_path}: {e}")

        console.rule("[bold cyan]Google Drive Workspace Status[/bold cyan]")
        console.print(f"[bold green]Workspace Root:[/bold green] [yellow]{root_dir}[/yellow]")
        console.print(f"  • Existing Folders (Preserved): [bold green]{existing_count}[/bold green]")
        console.print(f"  • Newly Created Folders: [bold cyan]{created_count}[/bold cyan]")
        console.print(f"[bold green]✓ Tất cả models, datasets & LoRA checkpoints sẽ được lưu 100% trực tiếp trên Google Drive.[/bold green]")
        console.rule()

        return path_map

    @classmethod
    def scan_existing_models(cls, root_dir: str = None) -> List[Dict[str, Any]]:
        """Scans and lists existing base model weights in Google Drive."""
        root = root_dir or (cls.DEFAULT_DRIVE_ROOT if cls.is_colab() else cls.LOCAL_FALLBACK_ROOT)
        models_dir = os.path.join(root, "models")
        
        found_models = []
        if not os.path.exists(models_dir):
            return found_models

        for root_p, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith((".safetensors", ".ckpt", ".pt", ".bin", ".gguf")):
                    full_p = os.path.join(root_p, file)
                    try:
                        size_gb = round(os.path.getsize(full_p) / (1024 ** 3), 2)
                        rel_cat = os.path.relpath(root_p, models_dir)
                        found_models.append({
                            "category": rel_cat,
                            "filename": file,
                            "size_gb": size_gb,
                            "path": full_p
                        })
                    except Exception:
                        pass
        return found_models
