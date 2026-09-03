from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, model_validator

class ModelFamily(str, Enum):
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    SDXL = "sdxl"
    PONY_V6 = "pony-v6"
    ILLUSTRIOUS = "illustrious"
    SD35 = "sd3.5"
    SD15 = "sd1.5"
    KREA2 = "krea2"
    Z_IMAGE = "z-image"
    WAN21 = "wan2.1"

class TrainingObjective(str, Enum):
    FACE_IDENTITY_100 = "face_identity_100"
    CHARACTER_FULLBODY = "character_fullbody"
    PHOTOREAL_SKIN = "photoreal_skin"
    ART_STYLE = "art_style"
    OBJECT_PRODUCT = "object_product"

class PrecisionMode(str, Enum):
    AUTO = "auto"
    FP8 = "fp8"
    BF16 = "bf16"
    FP16 = "fp16"

class DatasetConfig(BaseModel):
    dataset_path: str = Field(..., description="Đường dẫn thư mục ảnh huấn luyện")
    trigger_word: str = Field("sks", description="Từ khóa kích hoạt chính của LoRA")
    class_word: str = Field("person", description="Lớp khái niệm (person, woman, style, etc.)")
    repeats: int = Field(10, description="Số lần lặp lại mỗi ảnh trong 1 epoch")
    resolution: int = Field(1024, description="Độ phân giải tối đa của ảnh")
    enable_bucketing: bool = Field(True, description="Aspect Ratio Bucketing tránh méo hình")
    auto_caption: bool = Field(True, description="Tự động gán nhãn bằng AI Vision")
    caption_backend: str = Field("joycaption", description="joycaption | florence2 | wd14 | gemini")
    
    # Kỹ thuật độc quyền cho 100% Likeness
    subject_decoupling: bool = Field(
        True, 
        description="Lọc nhãn cô lập chủ thể: triệt tiêu từ miêu tả khuôn mặt để 100% nhận diện dồn vào trigger"
    )
    face_crop_multiscale: bool = Field(
        True, 
        description="Tự động trích xuất crop cận mặt 1024px + nửa thân + toàn thân"
    )

class TrainingConfig(BaseModel):
    model_family: ModelFamily = Field(ModelFamily.FLUX_DEV, description="Dòng mô hình nền tảng")
    base_model_path: str = Field("black-forest-labs/FLUX.1-dev", description="Đường dẫn repo hoặc file model")
    objective: TrainingObjective = Field(TrainingObjective.FACE_IDENTITY_100, description="Mục tiêu chuyên biệt")
    
    output_dir: str = Field("./outputs", description="Thư mục xuất file LoRA .safetensors")
    output_name: str = Field("my_omni_lora", description="Tên file LoRA kết quả")
    
    # Kiến trúc LoRA & DoRA
    network_dim: int = Field(32, description="LoRA Rank (16, 32, 64, 128)")
    network_alpha: int = Field(16, description="LoRA Alpha (thường đặt Rank / 2 hoặc bằng Rank)")
    use_dora: bool = Field(True, description="Weight-Decomposed LoRA - Tái hiện 100% chất lượng Full Fine-Tune")
    
    # Optimizer & Hyperparameters
    optimizer_type: str = Field("Prodigy", description="Prodigy | AdamW8bit | AdamW | Adafactor")
    learning_rate: float = Field(1.0, description="1.0 cho Prodigy, 1e-4 cho AdamW")
    unet_lr: Optional[float] = Field(None, description="Tốc độ học riêng cho Unet / DiT")
    text_encoder_lr: Optional[float] = Field(None, description="Tốc độ học riêng cho Text Encoder")
    
    epochs: int = Field(12, description="Số chu kỳ huấn luyện")
    batch_size: int = Field(1, description="Kích thước batch huấn luyện")
    gradient_accumulation_steps: int = Field(2, description="Số bước tích lũy gradient")
    save_every_n_epochs: int = Field(2, description="Tần suất lưu checkpoint")
    seed: int = Field(42, description="Hạt giống ngẫu nhiên tạo tính tất định")
    
    # Colab Memory Optimizations
    cache_latents_to_disk: bool = Field(True, description="Pre-cache latents giảm 60% VRAM")
    gradient_checkpointing: bool = Field(True, description="Tiết kiệm VRAM bộ nhớ đệm")
    mixed_precision: PrecisionMode = Field(PrecisionMode.AUTO, description="Độ chính xác số học")

class ValidationConfig(BaseModel):
    enabled: bool = Field(True, description="Kích hoạt kiểm thử định kỳ trong lúc huấn luyện")
    sample_every_n_steps: int = Field(250, description="Sinh ảnh sau mỗi N bước")
    validation_prompts: List[str] = Field(
        default_factory=lambda: [
            "A high-quality studio portrait of {trigger} {class_name}, neutral background, 8k, sharp focus",
            "A cinematic photo of {trigger} {class_name} walking on a vibrant neon street at night, realistic",
            "A candid photo of {trigger} {class_name} laughing at a cozy sunny cafe, natural sunlight"
        ],
        description="Danh sách prompt kiểm thử nhận diện & độ biến thiên bối cảnh"
    )
    # Khảo sát Likeness 100%
    compute_likeness_score: bool = Field(True, description="Đo lường độ giống ảnh gốc bằng ArcFace/InsightFace")
    select_best_checkpoint: bool = Field(True, description="Tự động chọn checkpoint có điểm giống cao nhất")

class OmniConfig(BaseModel):
    dataset: DatasetConfig
    training: TrainingConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "OmniConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)
