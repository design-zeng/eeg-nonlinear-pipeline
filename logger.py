import logging
from pathlib import Path
from config import LOGGER_DIR

def get_logger(name: str = "eeg_pipeline"):
    LOGGER_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        file_handler = logging.FileHandler(LOGGER_DIR / f"{name}.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
