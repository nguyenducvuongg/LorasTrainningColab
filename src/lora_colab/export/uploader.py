import os
from typing import Optional

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class ModelUploader:
    """Uploads finished LoRA checkpoints to Hugging Face Hub or Civitai."""

    @staticmethod
    def upload_to_huggingface(
        lora_file_path: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload trained LoRA from Colab LoRA Studio"
    ) -> bool:
        try:
            hf_token = token or os.environ.get("HF_TOKEN")
            if not hf_token:
                raise ValueError("HF_TOKEN is required to upload to Hugging Face.")

            api = HfApi(token=hf_token)
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

            console.print(f"[bold cyan]📤 Uploading '{lora_file_path}' to HuggingFace repo '{repo_id}'...[/bold cyan]")
            api.upload_file(
                path_or_fileobj=lora_file_path,
                path_in_repo=os.path.basename(lora_file_path),
                repo_id=repo_id,
                commit_message=commit_message
            )
            console.print(f"[bold green]✓ Successfully uploaded to https://huggingface.co/{repo_id}[/bold green]")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")
            return False
