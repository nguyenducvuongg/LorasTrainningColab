import os
import argparse
import sys
from .core.logger import setup_logger, console
from .core.hardware import HardwareProfiler
from .core.config import ConfigManager
from .storage.gdrive_manager import GDriveWorkspaceManager
from .storage.model_downloader import SmartModelDownloader
from .storage.resume_manager import ResumeManager
from .dataset.normalizer import DatasetNormalizer
from .dataset.captioning.gemini_api import GeminiVisionCaptioner
from .dataset.captioning.deepseek_api import DeepSeekVisionCaptioner
from .dataset.captioning.wd14 import WD14Tagger
from .dataset.captioning.joycaption import JoyCaptioner
from .engines.aitoolkit_trainer import AIToolkitTrainer
from .engines.kohya_trainer import KohyaTrainer
from .engines.diffusers_trainer import DiffusersTrainer

logger = setup_logger("lora_colab.cli")

def main():
    parser = argparse.ArgumentParser(description="Colab LoRA Studio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Init Drive
    sub_init = subparsers.add_parser("init-drive", help="Mount Google Drive and initialize directory structure")
    sub_init.add_argument("--root", type=str, default=None, help="Custom root directory")

    # 2. Hardware Profile
    sub_prof = subparsers.add_parser("profile", help="Detect GPU hardware and display auto-tuned settings")
    sub_prof.add_argument("--model", type=str, default="flux", help="Target model family (flux, sdxl, pony, sd15)")

    # 3. Download Model
    sub_dl = subparsers.add_parser("download-model", help="Download base model directly to Drive")
    sub_dl.add_argument("--model", type=str, required=True, help="Model key (flux-dev, sdxl-base, pony-v6, etc.)")
    sub_dl.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token")
    sub_dl.add_argument("--root", type=str, default=None, help="Workspace root")

    # 4. Normalize Dataset
    sub_norm = subparsers.add_parser("normalize-dataset", help="Rename and clean images and caption files")
    sub_norm.add_argument("--input-dir", type=str, required=True, help="Dataset directory")
    sub_norm.add_argument("--prefix", type=str, default="char", help="Prefix for filenames (e.g. char_0001.png)")
    sub_norm.add_argument("--format", type=str, default="PNG", help="Image format (PNG or JPEG)")

    # 5. Caption Dataset
    sub_cap = subparsers.add_parser("caption", help="Auto-caption dataset images")
    sub_cap.add_argument("--dir", type=str, required=True, help="Directory of images")
    sub_cap.add_argument("--engine", type=str, choices=["gemini", "deepseek", "wd14", "joycaption"], required=True)
    sub_cap.add_argument("--trigger", type=str, default="", help="Trigger word to prepend")
    sub_cap.add_argument("--api-key", type=str, default=None, help="API Key for Gemini or DeepSeek")

    # 6. Train
    sub_tr = subparsers.add_parser("train", help="Run LoRA training")
    sub_tr.add_argument("--config", type=str, required=True, help="Path to YAML or TOML config")
    sub_tr.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint in Drive")

    args = parser.parse_args()

    if args.command == "init-drive":
        GDriveWorkspaceManager.mount_google_drive()
        GDriveWorkspaceManager.init_workspace(args.root)

    elif args.command == "profile":
        profile = HardwareProfiler.detect_and_profile(args.model)
        HardwareProfiler.display_profile(profile)

    elif args.command == "download-model":
        root = args.root or GDriveWorkspaceManager.DEFAULT_DRIVE_ROOT
        SmartModelDownloader.ensure_model_ready(args.model, root, hf_token=args.hf_token)

    elif args.command == "normalize-dataset":
        DatasetNormalizer.normalize_folder(args.input_dir, prefix=args.prefix, target_format=args.format)

    elif args.command == "caption":
        if args.engine == "gemini":
            captioner = GeminiVisionCaptioner(api_key=args.api_key)
        elif args.engine == "deepseek":
            captioner = DeepSeekVisionCaptioner(api_key=args.api_key)
        elif args.engine == "wd14":
            captioner = WD14Tagger()
        else:
            captioner = JoyCaptioner()
        captioner.caption_directory(args.dir, trigger_word=args.trigger)

    elif args.command == "train":
        config = ConfigManager.load_config(args.config)
        profile = HardwareProfiler.detect_and_profile(config.training.model_family)
        config = ConfigManager.apply_hardware_profile(config, profile)
        
        resume_from = None
        if args.resume:
            resume_info = ResumeManager.get_resume_status(config.training.checkpoint_dir)
            if resume_info["can_resume"]:
                resume_from = resume_info["checkpoint_path"]

        fam = config.training.model_family.lower()
        if "flux" in fam or "krea" in fam:
            trainer = AIToolkitTrainer(config)
        elif any(k in fam for k in ["sdxl", "pony", "sd15", "sd35"]):
            trainer = KohyaTrainer(config)
        else:
            trainer = DiffusersTrainer(config)

        trainer.train(resume_from=resume_from)

if __name__ == "__main__":
    main()
