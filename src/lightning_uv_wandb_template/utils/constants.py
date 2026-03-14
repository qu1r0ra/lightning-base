from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]

DATA_DIR = ROOT_DIR / "data"
BY_CLASS_DIR = DATA_DIR / "by_class"
ML_SPLIT_DIR = DATA_DIR / "ml_split"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

ASSETS_DIR = ROOT_DIR / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"

# Data configurations
IMAGE_SIZE = 256
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

# Model & Execution defaults
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
SEEDS = [42, 1337, 7, 1234, 99]
DEFAULT_SEED = SEEDS[0]

DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_DROPOUT = 0.0
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1

# Task configurations
NUM_CLASSES = 5

# Weights & Biases
WANDB_ENTITY = "golshi-glazer"
WANDB_PROJECT = "lightning-uv-wandb-template"

# Visualization defaults
DPI = 300
