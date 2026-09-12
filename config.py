import os
from pathlib import Path

TASK_MAP: dict[str, list[str]] = {
    # "RST1": ["RST1"],
    # "RST2": ["RST2"],
    "IDG": ["1_IDG", "2_IDG", "3_IDG"],
    "IDE": ["1_IDE", "2_IDE", "3_IDE"],
    "IDR": ["1_IDR", "2_IDR", "3_IDR"],
}
# Project root = .../eeg_pipeline_project
ROOT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = ROOT_DIR / "storage"

INPUT_DIR = STORAGE_DIR / "input_data"
OUTPUT_DIR = STORAGE_DIR / "output_data"
IMAGE_DIR = STORAGE_DIR / "images"
LOGGER_DIR = STORAGE_DIR / "logs"

# Custom temporary cache directory (can be set to local fast NVMe on Colab)
CUSTOM_CACHE_DIR = os.getenv("EEG_CACHE_DIR", "")
CACHE_DIR = Path(CUSTOM_CACHE_DIR) if CUSTOM_CACHE_DIR else None

FS = 500
CHANNELS = 64
TAU = 10  # for f7 and f8 tau parameter
LAG = 1  # for f1 and f3 lag(tau) parameter
EMB_DIM = 2

# Default number of parallel tasks per feature channel (can override via EEG_PARALLEL_TASKS)
PARALLEL_TASK_COUNT = int(os.getenv("EEG_PARALLEL_TASKS", "1"))

# Memory limit threshold (can override via EEG_MEM_LIMIT)
MAX_WORKER_MEMORY_LIMIT = float(os.getenv("EEG_MEM_LIMIT", "0.00005"))

# CPU utilization ratio (can override via EEG_CPU_RATIO)
CPU_UTILIZATION_RATIO = float(os.getenv("EEG_CPU_RATIO", "0.00005"))

