from abc import ABC, abstractmethod
from typing import List, Optional, Dict

class BaseCaptioner(ABC):
    """Abstract base class for all dataset captioning engines."""

    @abstractmethod
    def caption_image(self, image_path: str, trigger_word: Optional[str] = None) -> str:
        """Generates caption text for a single image."""
        pass

    @abstractmethod
    def caption_directory(
        self,
        directory: str,
        trigger_word: Optional[str] = None,
        overwrite: bool = False
    ) -> Dict[str, int]:
        """Generates caption text for all images in a directory."""
        pass
