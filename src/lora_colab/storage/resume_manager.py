import os
import re
from typing import Optional, Tuple, List, Dict, Any
from ..core.logger import setup_logger, console

logger = setup_logger(__name__)

class ResumeManager:
    """Manages finding recent checkpoints in Google Drive for seamless auto-resume."""

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Tuple[str, int]]:
        """
        Scans checkpoint_dir and returns (path_to_latest_file, step_or_epoch_number).
        Returns None if no checkpoint found.
        """
        if not os.path.exists(checkpoint_dir):
            return None

        candidates: List[Tuple[str, int]] = []
        pattern = re.compile(r"[-_](?:step|epoch|state)?[-_]?(\d+)", re.IGNORECASE)

        for item in os.listdir(checkpoint_dir):
            full_path = os.path.join(checkpoint_dir, item)
            match = pattern.search(item)
            if match:
                try:
                    step_num = int(match.group(1))
                    candidates.append((full_path, step_num))
                except ValueError:
                    continue

        if not candidates:
            # Fallback: check modification time
            all_files = [
                os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
                if f.endswith((".safetensors", ".pt", ".bin")) or os.path.isdir(os.path.join(checkpoint_dir, f))
            ]
            if all_files:
                latest = max(all_files, key=os.path.getmtime)
                return (latest, 0)
            return None

        # Sort by step number descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    @classmethod
    def get_resume_status(cls, checkpoint_dir: str) -> Dict[str, Any]:
        """Returns detailed resume status information."""
        latest = cls.find_latest_checkpoint(checkpoint_dir)
        if latest:
            checkpoint_path, step = latest
            console.print(f"[bold yellow]⚡ Found existing checkpoint in Google Drive:[/bold yellow] {checkpoint_path} (Step/Epoch: {step})")
            return {
                "can_resume": True,
                "checkpoint_path": checkpoint_path,
                "step": step
            }
        else:
            logger.info("No prior checkpoint found. Starting fresh training session.")
            return {
                "can_resume": False,
                "checkpoint_path": None,
                "step": 0
            }
