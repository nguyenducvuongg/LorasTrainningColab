import os
from typing import Optional, List
from PIL import Image
from ..core.logger import setup_logger, console

try:
    import torch
except ImportError:
    torch = None

logger = setup_logger(__name__)

class SamplePreviewGenerator:
    """Generates preview images during or after training using Diffusers pipeline."""

    @staticmethod
    def generate_preview(
        base_model_path: str,
        lora_weights_path: Optional[str],
        prompt: str,
        negative_prompt: str = "ugly, blurry, low quality, deformed",
        output_path: str = "sample.png",
        model_family: str = "sdxl",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        seed: int = 42
    ) -> Optional[str]:
        try:
            from diffusers import AutoPipelineForText2Image

            console.print(f"[bold cyan]🎨 Generating preview for prompt:[/bold cyan] '{prompt}'...")
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

            is_single_file = base_model_path.endswith((".safetensors", ".ckpt"))
            if is_single_file:
                pipe = AutoPipelineForText2Image.from_single_file(
                    base_model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).to(device)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    base_model_path,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).to(device)

            if lora_weights_path and os.path.exists(lora_weights_path):
                pipe.load_lora_weights(lora_weights_path)

            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            image.save(output_path)
            console.print(f"[bold green]✓ Preview image saved to: {output_path}[/bold green]")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate sample preview: {e}")
            return None
