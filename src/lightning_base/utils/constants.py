from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[3]
DATA_DIR = ROOT_DIR / "data"

# Data configurations
VAL_SPLIT = 0.2
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
SEEDS = [42, 1337, 7, 1234, 99]
DEFAULT_SEED = SEEDS[0]

# Model defaults
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_DROPOUT = 0.0
DEFAULT_NUM_CLASSES = 6
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1

# Weights & Biases
WANDB_ENTITY = "golshi-glazer"
WANDB_PROJECT = "lightning-base"
