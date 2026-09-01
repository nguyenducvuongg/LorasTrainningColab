import os
from typing import Optional
from ..core.logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

class LoRAMerger:
    """Merges LoRA weights directly into base model weights."""

    @staticmethod
    def merge_lora_to_base(
        base_model_path: str,
        lora_weights_path: str,
        output_checkpoint_path: str,
        alpha_multiplier: float = 1.0
    ) -> bool:
        try:
            from diffusers import AutoPipelineForText2Image

            console.print(f"[bold cyan]🔄 Merging LoRA into base model...[/bold cyan]")
            pipe = AutoPipelineForText2Image.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            pipe.load_lora_weights(lora_weights_path)
            pipe.fuse_lora(lora_scale=alpha_multiplier)
            pipe.unload_lora()

            os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
            pipe.save_pretrained(output_checkpoint_path, safe_serialization=True)
            console.print(f"[bold green]✓ Merged model saved to: {output_checkpoint_path}[/bold green]")
            return True
        except Exception as e:
            logger.error(f"Failed to merge LoRA: {e}")
            return False
