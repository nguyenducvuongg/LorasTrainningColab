import os
import yaml

try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        tomllib = None

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, model_validator
from .logger import setup_logger
from .hardware import GPUProfile

logger = setup_logger(__name__)

class DatasetConfig(BaseModel):
    dataset_dir: str
    repeats: int = 10
    task_type: str = "character"  # character, face, body, style, enhancement, control
    resolution: int = 1024
    enable_bucketing: bool = True
    # Bucket resolution — hỗ trợ cả hai tên để tương thích với Kohya và codebase cũ
    min_bucket_res: int = 512
    max_bucket_res: int = 2048
    min_bucket_resolution: int = 512   # alias cho Kohya trainer
    max_bucket_resolution: int = 2048  # alias cho Kohya trainer
    caption_extension: str = ".txt"
    shuffle_caption: bool = True
    keep_tokens: int = 1

    @model_validator(mode="after")
    def sync_bucket_aliases(self) -> "DatasetConfig":
        """Đồng bộ hai cặp tên field bucket resolution với nhau."""
        # Nếu user set min_bucket_res, sync sang min_bucket_resolution
        if self.min_bucket_res != 512:
            self.min_bucket_resolution = self.min_bucket_res
        elif self.min_bucket_resolution != 512:
            self.min_bucket_res = self.min_bucket_resolution
        if self.max_bucket_res != 2048:
            self.max_bucket_resolution = self.max_bucket_res
        elif self.max_bucket_resolution != 2048:
            self.max_bucket_res = self.max_bucket_resolution
        return self

class NetworkConfig(BaseModel):
    network_module: str = "networks.lora"  # lora, lycoris, dora
    network_dim: int = 32                  # Rank
    network_alpha: float = 16.0
    network_dropout: float = 0.0
    network_args: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    base_model_path: str
    model_family: str = "flux"  # flux, flux-kontext, krea, sdxl, pony, sd35, sd15, qwen
    output_name: str = "my_custom_lora"
    output_dir: str = "/content/drive/MyDrive/Colab_LoRA_Studio/outputs/final_loras"
    checkpoint_dir: str = "/content/drive/MyDrive/Colab_LoRA_Studio/outputs/checkpoints"
    sample_dir: str = "/content/drive/MyDrive/Colab_LoRA_Studio/outputs/samples"
    logging_dir: str = "/content/drive/MyDrive/Colab_LoRA_Studio/outputs/logs"
    
    # Training Loop
    epochs: int = 10
    max_train_steps: Optional[int] = None
    save_every_n_epochs: int = 1
    save_every_n_steps: Optional[int] = 250
    save_state: bool = True  # For auto-resume
    
    # Hardware & Performance (overridden by GPUProfile if enabled)
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    text_encoder_lr: Optional[float] = 5e-5
    optimizer_type: str = "AdamW8bit"  # AdamW, AdamW8bit, Prodigy, Adafactor
    optimizer_args: Dict[str, Any] = Field(default_factory=dict)
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    mixed_precision: str = "bf16"  # fp16, bf16, fp8
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = False  # Cho Kohya — không dùng khi train TE

    # Advanced Training Params
    noise_offset: float = 0.0             # Noise offset cho SD1.5/SDXL
    min_snr_gamma: Optional[float] = None # SNR loss weighting (5.0 recommended)
    network_weights: Optional[str] = None  # Cho phép tiếp tục train từ LoRA cũ
    v_parameterization: bool = False       # Cho SD 2.x / v-prediction models
    
    # Monitoring & Notifications
    discord_webhook_url: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    wandb_token: Optional[str] = None
    wandb_project: Optional[str] = None
    sample_prompt: Optional[str] = None
    sample_every_n_steps: int = 100

class LoRAConfig(BaseModel):
    dataset: DatasetConfig
    network: NetworkConfig
    training: TrainingConfig

class ConfigManager:
    """Manages loading, merging, validating, and saving YAML/TOML configs."""
    
    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def load_toml(file_path: str) -> Dict[str, Any]:
        if tomllib is None:
            raise ImportError("Neither 'tomllib' (Python 3.11+) nor 'toml' package is available.")
        
        # Check if it has loads or load with binary mode
        try:
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            with open(file_path, "r", encoding="utf-8") as f:
                if hasattr(tomllib, "loads"):
                    return tomllib.loads(f.read())
                elif hasattr(tomllib, "load"):
                    return tomllib.load(f)
                return {}

    @classmethod
    def load_config(cls, file_path: str) -> LoRAConfig:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in [".yaml", ".yml"]:
            data = cls.load_yaml(file_path)
        elif ext in [".toml"]:
            data = cls.load_toml(file_path)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        return LoRAConfig.model_validate(data)

    @classmethod
    def apply_hardware_profile(cls, config: LoRAConfig, profile: GPUProfile) -> LoRAConfig:
        """Adapts configuration settings based on the auto-detected GPU profile."""
        config.training.batch_size = profile.recommended_batch_size
        config.training.gradient_accumulation_steps = profile.gradient_accumulation_steps
        config.training.gradient_checkpointing = profile.gradient_checkpointing
        config.training.mixed_precision = profile.precision
        config.training.cache_latents_to_disk = profile.cache_latents_to_disk
        
        if profile.optimizer.lower() == "adamw8bit":
            config.training.optimizer_type = "AdamW8bit"
        elif profile.optimizer.lower() == "prodigy":
            config.training.optimizer_type = "Prodigy"
            config.training.learning_rate = 1.0  # Prodigy default D-Adaptation LR
            config.training.optimizer_args["d_coef"] = 1.0
        elif profile.optimizer.lower() == "adamw":
            config.training.optimizer_type = "AdamW"
            
        logger.info(f"Applied dynamic optimizations for GPU Tier: [bold green]{profile.tier}[/bold green]")
        return config

    @staticmethod
    def save_yaml(config: LoRAConfig, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved merged config to: {output_path}")
