import os
import shutil
import subprocess
import requests
from typing import Dict, Any, List, Optional
from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

# Registry of 100% PUBLIC, UNGATED, DIRECT ACCESS base models and encoders (NO LOGIN / NO ACCEPTANCE REQUIRED)
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "flux-dev": {
        "name": "FLUX.1-dev (Public Mirror)",
        "category": "models/flux",
        "filename": "flux1-dev.safetensors",
        "repo_id": "camenduru/FLUX.1-dev",
        "hf_filename": "flux1-dev.safetensors",
        "direct_url": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "min_size_gb": 20.0,
        "is_gated": False,
        "auxiliary_files": ["t5xxl-fp8", "clip-l", "flux-vae"]
    },
    "flux-schnell": {
        "name": "FLUX.1-schnell (Public Mirror)",
        "category": "models/flux",
        "filename": "flux1-schnell.safetensors",
        "repo_id": "Comfy-Org/flux1-schnell",
        "hf_filename": "flux1-schnell-fp8.safetensors",
        "direct_url": "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors",
        "min_size_gb": 10.0,
        "is_gated": False,
        "auxiliary_files": ["t5xxl-fp8", "clip-l", "flux-vae"]
    },
    "flux-kontext": {
        "name": "FLUX.1-Kontext-dev (Black Forest Labs)",
        "category": "models/flux_kontext",
        "filename": "flux1-kontext-dev.safetensors",
        "repo_id": "black-forest-labs/FLUX.1-Kontext-dev",
        "hf_filename": "flux1-kontext-dev.safetensors",
        # Public mirror fallback (Kontext là model mới, dùng flux-dev mirror khi chưa có ungated)
        "direct_url": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "min_size_gb": 20.0,
        "is_gated": True,  # Kontext-dev cần đăng nhập HF — user cần set HF_TOKEN
        "note": "Yêu cầu Hugging Face Token (FLUX.1-Kontext-dev là gated model). Đặt HF_TOKEN trong Colab Secrets.",
        "auxiliary_files": ["t5xxl-fp8", "clip-l", "flux-vae"]
    },
    "krea2-raw": {
        "name": "Krea2-Raw / Creative Diffusion (SDXL-based)",
        "category": "models/krea",
        "filename": "krea2-raw.safetensors",
        "repo_id": "Krea/KREA-Raw",
        "hf_filename": "krea2-raw.safetensors",
        # Fallback: dùng SDXL base khi Krea chưa public
        "direct_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "min_size_gb": 6.0,
        "is_gated": False,
        "note": "Krea2-Raw là kiến trúc SDXL. Nếu chưa có model chính thức, sẽ dùng SDXL base.",
        "auxiliary_files": ["sdxl-vae"]
    },
    "sdxl-base": {
        "name": "Stable Diffusion XL 1.0 Base",
        "category": "models/sdxl",
        "filename": "sd_xl_base_1.0.safetensors",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "hf_filename": "sd_xl_base_1.0.safetensors",
        "direct_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "min_size_gb": 6.0,
        "is_gated": False,
        "auxiliary_files": ["sdxl-vae"]
    },
    "pony-v6": {
        "name": "Pony Diffusion V6 XL",
        "category": "models/sdxl",
        "filename": "ponyDiffusionV6XL.safetensors",
        "repo_id": "AstraliteHeart/pony-diffusion-v6",
        "hf_filename": "v6.safetensors",
        "direct_url": "https://huggingface.co/AstraliteHeart/pony-diffusion-v6/resolve/main/v6.safetensors",
        "min_size_gb": 6.0,
        "is_gated": False,
        "auxiliary_files": ["sdxl-vae"]
    },
    "illustrious-xl": {
        "name": "Illustrious-XL v0.1",
        "category": "models/sdxl",
        "filename": "illustrious-xl-v0.1.safetensors",
        "repo_id": "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "hf_filename": "Illustrious-XL-v0.1.safetensors",
        "direct_url": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0/resolve/main/Illustrious-XL-v0.1.safetensors",
        "min_size_gb": 6.0,
        "is_gated": False,
        "auxiliary_files": ["sdxl-vae"]
    },
    "sd35-medium": {
        "name": "Stable Diffusion 3.5 Medium (Hugging Face Public)",
        "category": "models/sd35",
        "filename": "sd3.5_medium.safetensors",
        # FIX: dùng đúng repo SD 3.5 Medium thay vì SDXL base
        "repo_id": "adamo1139/stable-diffusion-3.5-medium-ungated",
        "hf_filename": "sd3.5_medium.safetensors",
        "direct_url": "https://huggingface.co/adamo1139/stable-diffusion-3.5-medium-ungated/resolve/main/sd3.5_medium.safetensors",
        "min_size_gb": 5.0,
        "is_gated": False,
        "auxiliary_files": ["clip-l", "clip-g", "t5xxl-fp8"]
    },
    "z-image-kolors": {
        "name": "Z-Image / Kwai Kolors Diffusion (Public)",
        "category": "models/kolors",
        "filename": "kolors_unet.safetensors",
        "repo_id": "Kwai-Kolors/Kolors-diffusers",
        "hf_filename": "unet/diffusion_pytorch_model.fp16.safetensors",
        "direct_url": "https://huggingface.co/Kwai-Kolors/Kolors-diffusers/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors",
        "min_size_gb": 4.5,
        "is_gated": False,
        "auxiliary_files": ["sdxl-vae"]
    },
    "qwen-image": {
        "name": "Qwen2-VL Multimodal Diffusion (Public)",
        "category": "models/qwen",
        "filename": "qwen2_vl_model.safetensors",
        "repo_id": "Qwen/Qwen2-VL-2B-Instruct",
        "hf_filename": "model-00001-of-00002.safetensors",
        "direct_url": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/model-00001-of-00002.safetensors",
        "min_size_gb": 3.5,
        "is_gated": False,
    },
    "sd15-base": {
        "name": "Stable Diffusion 1.5",
        "category": "models/sd15",
        "filename": "v1-5-pruned-emaonly.safetensors",
        "repo_id": "runwayml/stable-diffusion-v1-5",
        "hf_filename": "v1-5-pruned-emaonly.safetensors",
        "direct_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "min_size_gb": 3.8,
        "is_gated": False,
    },
    # 100% UNGATED Encoders & VAE
    "t5xxl-fp8": {
        "name": "T5-XXL FP8 Text Encoder (Public)",
        "category": "models/text_encoders",
        "filename": "t5xxl_fp8_e4m3fn.safetensors",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "hf_filename": "t5xxl_fp8_e4m3fn.safetensors",
        "direct_url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "min_size_gb": 4.5,
        "is_gated": False,
    },
    "clip-l": {
        "name": "CLIP-L Text Encoder (Public)",
        "category": "models/text_encoders",
        "filename": "clip_l.safetensors",
        "repo_id": "comfyanonymous/flux_text_encoders",
        "hf_filename": "clip_l.safetensors",
        "direct_url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "min_size_gb": 0.2,
        "is_gated": False,
    },
    "clip-g": {
        "name": "CLIP-G Text Encoder (Public)",
        "category": "models/text_encoders",
        "filename": "clip_g.safetensors",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "hf_filename": "text_encoder_2/model.safetensors",
        "direct_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors",
        "min_size_gb": 1.2,
        "is_gated": False,
    },
    "flux-vae": {
        "name": "FLUX VAE (Public Mirror)",
        "category": "models/vae",
        "filename": "ae.safetensors",
        "repo_id": "camenduru/FLUX.1-dev",
        "hf_filename": "ae.safetensors",
        "direct_url": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors",
        "min_size_gb": 0.3,
        "is_gated": False,
    },
    "sdxl-vae": {
        "name": "SDXL VAE (FP16 Fixed - Public)",
        "category": "models/vae",
        "filename": "sdxl_vae.safetensors",
        "repo_id": "madebyollin/sdxl-vae-fp16-fix",
        "hf_filename": "sdxl_vae.safetensors",
        "direct_url": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
        "min_size_gb": 0.3,
        "is_gated": False,
    }
}

class SmartModelDownloader:
    """Downloads base models, VAEs, and Text Encoders directly to Google Drive with 100% public access."""
    
    @staticmethod
    def _download_via_requests(url: str, output_path: str):
        """Streams download directly using HTTP requests with progress bar."""
        temp_path = output_path + ".tmp"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(temp_path, 'wb') as file, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=2 * 1024 * 1024):
                size = file.write(data)
                bar.update(size)
                
        os.rename(temp_path, output_path)

    @classmethod
    def download_model(cls, model_key: str, workspace_root: str, hf_token: Optional[str] = None) -> str:
        """
        Scans Google Drive for the model. If present and size is valid, skips download.
        Otherwise downloads directly into Drive from open public sources without login requirements.
        """
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
            
        info = MODEL_REGISTRY[model_key]
        dest_dir = os.path.join(workspace_root, info["category"])
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, info["filename"])
        
        # 1. Check if model already exists and is intact in Drive
        if os.path.exists(dest_path):
            size_gb = os.path.getsize(dest_path) / (1024 ** 3)
            min_expected = info.get("min_size_gb", 0.1)
            if size_gb >= min_expected * 0.9:  # within 90% threshold
                console.print(f"[bold green]✓ Model '{info['name']}' found in Google Drive ({round(size_gb, 2)} GB). Skipping download.[/bold green]")
                # Download auxiliary files if any
                for aux_key in info.get("auxiliary_files", []):
                    cls.download_model(aux_key, workspace_root, hf_token)
                return dest_path
            else:
                logger.warning(f"Existing file {dest_path} is incomplete ({round(size_gb, 2)} GB < {min_expected} GB). Re-downloading...")

        # 2. Perform direct download into Google Drive
        console.print(f"[bold cyan]📥 Downloading '{info['name']}' directly to Google Drive: {dest_path}...[/bold cyan]")
        
        download_successful = False
        repo_id = info.get("repo_id")
        hf_filename = info.get("hf_filename")

        # Try Hugging Face Hub open download first
        if hf_hub_download and repo_id and hf_filename:
            try:
                downloaded_temp = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    token=hf_token,
                    local_dir=dest_dir,
                    local_dir_use_symlinks=False
                )
                
                # Ensure proper final filename
                if downloaded_temp != dest_path and os.path.exists(downloaded_temp):
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(downloaded_temp, dest_path)
                    
                console.print(f"[bold green]✓ Download complete: {dest_path}[/bold green]")
                download_successful = True
            except Exception as e:
                logger.warning(f"HF Hub download encountered an issue ({e}). Switching to direct public stream...")

        # Fallback to direct HTTP stream if needed
        if not download_successful:
            direct_url = info.get("direct_url") or f"https://huggingface.co/{repo_id}/resolve/main/{hf_filename}"
            try:
                cls._download_via_requests(direct_url, dest_path)
                console.print(f"[bold green]✓ Download complete via direct public stream: {dest_path}[/bold green]")
                download_successful = True
            except Exception as e:
                logger.error(f"Failed to download {info['name']}: {e}")
                raise e

        # 3. Check and download auxiliary files (Encoders, VAE)
        for aux_key in info.get("auxiliary_files", []):
            cls.download_model(aux_key, workspace_root, hf_token)

        return dest_path

    @classmethod
    def ensure_model_ready(cls, model_key: str, workspace_root: str, hf_token: Optional[str] = None) -> str:
        """Helper to ensure primary model and all encoders are ready in Drive."""
        return cls.download_model(model_key, workspace_root, hf_token)
