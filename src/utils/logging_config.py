from loguru import logger
import sys
from pathlib import Path


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/trading_system.log",
    rotation: str = "100 MB",
    retention: str = "30 days"
) -> None:
    """
    Configure loguru logger with file and console output.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    logger.info(f"Logger initialized with level={log_level}")


setup_logger()
