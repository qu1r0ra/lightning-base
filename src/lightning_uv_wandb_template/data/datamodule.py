from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from lightning_uv_wandb_template.utils.constants import (
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_SEED,
    NUM_WORKERS,
    PIN_MEMORY,
    VAL_SPLIT,
)


class TemplateDataModule(LightningDataModule):
    """
    A unified, simple DataModule template for image classification.
    Supports pre-split directories, automatic validation splits, and K-Fold CV.
    """

    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        val_split: float = VAL_SPLIT,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        seed: int = DEFAULT_SEED,
        k_fold: int = 1,
        fold_index: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None
        self._classes: list[str] | None = None

        # K-Fold state
        self._pool: Dataset | None = None
        self._splits: list[tuple[list[int], list[int]]] | None = None

        self.transform = Compose([ToTensor()])

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def classes(self) -> list[str] | None:
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self._classes) if self._classes else 0

    def prepare_data(self) -> None:
        """Download or unpack data here if necessary."""
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.train_dir.exists() and self.val_dir.exists():
                train_ds = ImageFolder(root=self.train_dir, transform=self.transform)
                val_ds = ImageFolder(root=self.val_dir, transform=self.transform)
                self._classes = train_ds.classes
                full_ds = ConcatDataset([train_ds, val_ds])
            elif self.data_dir.exists():
                full_ds = ImageFolder(root=self.data_dir, transform=self.transform)
                self._classes = full_ds.classes
            else:
                raise FileNotFoundError(f"Data directory {self.data_dir} not found.")

            total_size = len(full_ds)

            if self.hparams.k_fold > 1:
                # Generate simple generic K-Folds using PyTorch RNG
                indices = torch.randperm(
                    total_size,
                    generator=torch.Generator().manual_seed(self.hparams.seed),
                ).tolist()
                fold_size = total_size // self.hparams.k_fold
                self._splits = []
                for i in range(self.hparams.k_fold):
                    val_start = i * fold_size
                    val_end = (
                        (i + 1) * fold_size
                        if i < self.hparams.k_fold - 1
                        else total_size
                    )
                    val_idx = indices[val_start:val_end]
                    train_idx = indices[:val_start] + indices[val_end:]
                    self._splits.append((train_idx, val_idx))

                self._pool = full_ds
                self.set_fold(self.hparams.fold_index)
            else:
                # Standard train/val split
                if self.train_dir.exists() and self.val_dir.exists():
                    self.train_dataset = ImageFolder(
                        root=self.train_dir, transform=self.transform
                    )
                    self.val_dataset = ImageFolder(
                        root=self.val_dir, transform=self.transform
                    )
                else:
                    val_size = int(total_size * self.hparams.val_split)
                    train_size = total_size - val_size
                    self.train_dataset, self.val_dataset = random_split(
                        full_ds,
                        [train_size, val_size],
                        generator=torch.Generator().manual_seed(self.hparams.seed),
                    )

        if stage == "test" or stage is None:
            if self.test_dir.exists():
                self.test_dataset = ImageFolder(
                    root=self.test_dir, transform=self.transform
                )
                self._classes = self.test_dataset.classes

        if stage == "predict":
            predict_root = self.test_dir if self.test_dir.exists() else self.data_dir
            if predict_root.exists():
                self.predict_dataset = ImageFolder(
                    root=predict_root, transform=self.transform
                )

    def set_fold(self, fold_index: int) -> None:
        """Sets the active subsets for a specific fold in K-Fold Cross-Validation."""
        if self._splits is None or self._pool is None:
            raise RuntimeError("Splits not initialized. Call setup() first.")

        if not (0 <= fold_index < self.hparams.k_fold):
            raise ValueError(
                f"Fold index {fold_index} out of range (0-{self.hparams.k_fold - 1})"
            )

        self.hparams.fold_index = fold_index
        train_idx, val_idx = self._splits[fold_index]

        self.train_dataset = Subset(self._pool, train_idx)
        self.val_dataset = Subset(self._pool, val_idx)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataloader called before setup()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataloader called before setup()")
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.predict_dataset is None:
            raise RuntimeError("predict_dataloader called before setup()")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
