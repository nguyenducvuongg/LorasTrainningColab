from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..core.config import LoRAConfig

class BaseTrainer(ABC):
    """Abstract interface for all LoRA training engines."""

    def __init__(self, config: LoRAConfig):
        self.config = config

    @abstractmethod
    def build_command_or_config(self) -> Any:
        """Constructs the configuration file or execution command for the engine."""
        pass

    @abstractmethod
    def train(self, resume_from: Optional[str] = None) -> bool:
        """Launches the training process and handles logging/monitoring."""
        pass
