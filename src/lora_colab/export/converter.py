import os
from typing import Dict, Any, Optional
from ..core.logger import setup_logger

try:
    from safetensors.torch import load_file, save_file
except ImportError:
    load_file, save_file = None, None

logger = setup_logger(__name__)

class LoRAConverter:
    """Utilities to inspect, attach metadata, and convert LoRA safetensors formats."""

    @staticmethod
    def attach_metadata(
        lora_path: str,
        output_path: Optional[str] = None,
        base_model: str = "FLUX.1-dev",
        trigger_words: str = "",
        author: str = "Colab LoRA Studio"
    ):
        """Attaches standardized metadata tags into safetensors header."""
        out_p = output_path or lora_path
        state_dict = load_file(lora_path)
        
        metadata = {
            "ss_base_model_version": base_model,
            "ss_tag_frequency": trigger_words,
            "ss_author": author,
            "format": "pt",
            "generator": "Colab-LoRA-Studio"
        }
        
        save_file(state_dict, out_p, metadata=metadata)
        logger.info(f"Attached metadata to: {out_p}")
