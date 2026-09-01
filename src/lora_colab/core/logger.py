import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

def setup_logger(name: str = "lora_colab", level: int = logging.INFO) -> logging.Logger:
    """Setup a rich colorized logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        
    return logger
