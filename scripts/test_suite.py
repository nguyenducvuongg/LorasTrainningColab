import sys
import os
import tempfile
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from lora_colab.core.hardware import HardwareProfiler
from lora_colab.core.config import ConfigManager
from lora_colab.core.logger import console
from lora_colab.storage.gdrive_manager import GDriveWorkspaceManager
from lora_colab.storage.model_downloader import MODEL_REGISTRY
from lora_colab.dataset.normalizer import DatasetNormalizer
from lora_colab.dataset.cleaner import CaptionCleaner

def run_tests():
    console.rule("[bold green]Running Colab LoRA Studio System Verification[/bold green]")

    # 1. Test Hardware Profiler
    console.print("\n[bold cyan]1. Testing Hardware Profiler...[/bold cyan]")
    profile = HardwareProfiler.detect_and_profile("flux")
    HardwareProfiler.display_profile(profile)
    assert profile.tier in ["T4", "L4", "A100", "V100", "CPU", "GENERIC_CUDA"]
    console.print("[bold green]✓ Hardware Profiler passed![/bold green]")

    # 2. Test Model Registry
    console.print("\n[bold cyan]2. Testing Model Registry...[/bold cyan]")
    console.print(f"Total Registered Models: [bold yellow]{len(MODEL_REGISTRY)}[/bold yellow]")
    for k, v in MODEL_REGISTRY.items():
        assert "name" in v and "category" in v and "filename" in v
        console.print(f"  • [green]{k}[/green] -> {v['name']} ({v['category']}/{v['filename']})")
    console.print("[bold green]✓ Model Registry validated![/bold green]")

    # 3. Test Config Manager
    console.print("\n[bold cyan]3. Testing Config Manager...[/bold cyan]")
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "models"))
    sample_config_path = os.path.join(configs_dir, "flux_dev.yaml")
    config = ConfigManager.load_config(sample_config_path)
    config = ConfigManager.apply_hardware_profile(config, profile)
    assert config.training.model_family == "flux"
    assert config.training.batch_size == profile.recommended_batch_size
    console.print("[bold green]✓ Config Manager & Auto-Tuner passed![/bold green]")

    # 4. Test Normalizer & Cleaner
    console.print("\n[bold cyan]4. Testing Dataset Normalizer & Caption Cleaner...[/bold cyan]")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy image
        img_path = os.path.join(tmp_dir, "raw_sample_img.png")
        img = Image.new("RGBA", (512, 512), (255, 0, 0, 128))
        img.save(img_path)

        # Create dummy caption
        txt_path = os.path.join(tmp_dir, "raw_sample_img.txt")
        with open(txt_path, "w") as f:
            f.write("1girl, solo, lowres, blurry, blue eyes, masterpiece")

        # Run normalization
        res = DatasetNormalizer.normalize_folder(tmp_dir, prefix="test_char")
        assert res["processed_count"] == 1
        
        # Test caption cleaner
        norm_txt = os.path.join(tmp_dir, "test_char_0001.txt")
        with open(norm_txt, "r") as f:
            raw_caption = f.read()
        cleaned = CaptionCleaner.clean_text(raw_caption, trigger_word="sks character", is_danbooru_tags=True)
        assert "lowres" not in cleaned and "blurry" not in cleaned
        assert cleaned.startswith("sks character")
        console.print(f"Cleaned Caption: [italic green]{cleaned}[/italic green]")
        console.print("[bold green]✓ Dataset Normalizer & Caption Cleaner passed![/bold green]")

    # 5. Test Workspace Manager
    console.print("\n[bold cyan]5. Testing GDrive Workspace Manager...[/bold cyan]")
    with tempfile.TemporaryDirectory() as tmp_workspace:
        path_map = GDriveWorkspaceManager.init_workspace(tmp_workspace)
        assert os.path.exists(os.path.join(tmp_workspace, "models", "flux"))
        assert os.path.exists(os.path.join(tmp_workspace, "outputs", "final_loras"))
        console.print("[bold green]✓ Workspace Manager passed![/bold green]")

    console.rule("[bold green]🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY![/bold green]")

if __name__ == "__main__":
    run_tests()
