import random
import shutil
import zipfile
from pathlib import Path

import typer
from tqdm import tqdm

from lightning_uv_wandb_template.data.download import (
    download_plant_doc,
    download_plant_village,
)
from lightning_uv_wandb_template.utils.constants import (
    BY_CLASS_DIR,
    DATA_DIR,
    DEFAULT_SEED,
    IMAGE_EXTENSIONS,
    ML_SPLIT_DIR,
    SPLITS,
)
from lightning_uv_wandb_template.utils.logger import get_logger
from lightning_uv_wandb_template.utils.seed import seed_everything

logger = get_logger(__name__)
app = typer.Typer(help="Data Management Utility")


@app.command()
def unzip(zip_path: Path = BY_CLASS_DIR / "data.zip") -> None:
    """Unzip the data.zip file in by_class directory."""
    if not zip_path.exists():
        logger.warning(f"Zip file not found at {zip_path}. Skipping unzip.")
        return

    logger.info(f"Unzipping {zip_path} to {BY_CLASS_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(BY_CLASS_DIR)
    logger.info("Unzip complete.")


@app.command()
def setup() -> None:
    """Create the directory structure for the dataset."""
    classes_file = DATA_DIR / "classes.txt"

    if not classes_file.exists():
        logger.error(f"Class list not found at {classes_file}")
        raise FileNotFoundError(f"Class list not found at {classes_file}")

    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]

    logger.info("Initializing data directories...")

    for cls in classes:
        (BY_CLASS_DIR / cls).mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (ML_SPLIT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    logger.info(f"Successfully created folders for {len(classes)} classes.")


@app.command()
def split(
    force: bool = typer.Option(False, help="Force split even if it already exists"),
) -> None:
    """Split the dataset into train/val/test sets."""
    if ML_SPLIT_DIR.exists() and any(ML_SPLIT_DIR.iterdir()) and not force:
        logger.info(f"Split directory {ML_SPLIT_DIR} already exists. Skipping split.")
        return

    if force and ML_SPLIT_DIR.exists():
        logger.warning(f"Force flag set. Removing existing {ML_SPLIT_DIR}...")
        shutil.rmtree(ML_SPLIT_DIR)

    if not BY_CLASS_DIR.exists():
        logger.error(f"Source directory {BY_CLASS_DIR} does not exist.")
        raise FileNotFoundError(f"Source directory {BY_CLASS_DIR} does not exist.")

    class_folders = [
        f
        for f in BY_CLASS_DIR.iterdir()
        if f.is_dir() and f.name != ".ipynb_checkpoints"
    ]

    if not class_folders:
        logger.error(f"No class folders found in {BY_CLASS_DIR}")
        raise RuntimeError(f"No class folders found in {BY_CLASS_DIR}")

    logger.info(f"Found {len(class_folders)} classes. Starting split...")

    for class_folder in class_folders:
        class_name = class_folder.name

        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(list(class_folder.glob(f"*{ext}")))

        if not images:
            logger.warning(f"No images found for class '{class_name}'. Skipping.")
            continue

        random.shuffle(images)

        num_images = len(images)
        train_idx = int(num_images * SPLITS["train"])
        val_idx = train_idx + int(num_images * SPLITS["val"])

        assignments = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:],
        }

        logger.info(f"Processing '{class_name}': {num_images} images")

        for split_name, split_images in assignments.items():
            split_path = ML_SPLIT_DIR / split_name / class_name
            split_path.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(split_images, desc=f"  -> {split_name}", leave=False):
                shutil.copy2(img_path, split_path / img_path.name)

    logger.info("Data splitting complete!")


@app.command()
def download() -> None:
    """Download external datasets via Kaggle."""
    logger.info("Downloading external datasets...")
    download_plant_village()
    download_plant_doc()


@app.command()
def init() -> None:
    """Run all data initialization tasks (unzip -> setup -> split)."""
    seed_everything(DEFAULT_SEED)
    unzip()
    setup()
    split()


if __name__ == "__main__":
    app()
