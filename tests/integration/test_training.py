from pathlib import Path

import pytest
import torch.nn as nn
from lightning.pytorch import Trainer
from PIL import Image

from lightning_uv_wandb_template.data.datamodule import TemplateDataModule
from lightning_uv_wandb_template.models.classifier import TemplateClassifier


def create_mock_dataset(path: Path, num_classes: int = 2, num_images: int = 4) -> None:
    """Creates a minimal image folder dataset structure."""
    for split in ["train", "val", "test"]:
        split_path = path / split
        split_path.mkdir(parents=True, exist_ok=True)
        for i in range(num_classes):
            class_path = split_path / f"class_{i}"
            class_path.mkdir(exist_ok=True)
            for j in range(num_images):
                # Create small dummy images
                img = Image.new("RGB", (32, 32), color=(i * 10, j * 10, 0))
                img.save(class_path / f"img_{j}.jpg")


@pytest.mark.slow
def test_training_loop_fast_dev_run(tmp_path: Path) -> None:
    # 1. Setup mock data
    num_classes = 2
    create_mock_dataset(tmp_path, num_classes=num_classes, num_images=2)

    # 2. Setup DataModule
    dm = TemplateDataModule(
        data_dir=tmp_path,
        batch_size=2,
        num_workers=0,  # Avoid multiprocessing overhead in tests
        image_size=32,
    )

    # 3. Setup Model (Tiny backbone for speed)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )

    model = TemplateClassifier(
        model=backbone,
        num_classes=num_classes,
    )

    # 4. Run Trainer
    trainer = Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=dm)

    # 5. Assertions
    assert trainer.state.finished, "Training loop failed to finish"
