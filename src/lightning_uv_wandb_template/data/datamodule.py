from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)
from torchvision.datasets import ImageFolder

from lightning_uv_wandb_template.data.transforms import create_pipeline
from lightning_uv_wandb_template.utils.constants import (
    BATCH_SIZE,
    DEFAULT_SEED,
    IMAGE_SIZE,
    ML_SPLIT_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)


class TemplateDataModule(LightningDataModule):
    """
    An advanced, unified DataModule template for image classification.
    Supports pre-split directories, automatic validation splits, Stratified K-Fold CV,
    and WeightedRandomSampler for class imbalance.
    """

    def __init__(
        self,
        data_dir: str | Path = ML_SPLIT_DIR,
        val_split: float | None = None,
        batch_size: int = BATCH_SIZE,
        use_weighted_sampler: bool = False,
        seed: int = DEFAULT_SEED,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = PIN_MEMORY,
        k_fold: int = 1,
        fold_index: int = 0,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir).resolve()

        self.train_transform = create_pipeline(image_size, is_train=True, is_dl=True)
        self.val_transform = create_pipeline(image_size, is_train=False, is_dl=True)

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None
        self.sampler: WeightedRandomSampler | None = None
        self._classes: list[str] | None = None

        self._splits: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._pool_labels: list[int] | None = None
        self._train_pool: ConcatDataset | None = None
        self._val_pool: ConcatDataset | None = None

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
        """
        Download or initialize data here.
        Note: This is called only once on a single CPU/GPU.
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.train_dir.exists() and self.val_dir.exists():
                if self.hparams.k_fold > 1:
                    train_ds_no_aug = ImageFolder(root=self.train_dir)
                    val_ds_no_aug = ImageFolder(root=self.val_dir)

                    self._classes = train_ds_no_aug.classes
                    self._pool_labels = train_ds_no_aug.targets + val_ds_no_aug.targets

                    skf = StratifiedKFold(
                        n_splits=self.hparams.k_fold,
                        shuffle=True,
                        random_state=self.hparams.seed,
                    )
                    self._splits = list(
                        skf.split(np.zeros(len(self._pool_labels)), self._pool_labels)
                    )

                    self._train_pool = ConcatDataset(
                        [
                            ImageFolder(
                                root=self.train_dir, transform=self.train_transform
                            ),
                            ImageFolder(
                                root=self.val_dir, transform=self.train_transform
                            ),
                        ]
                    )
                    self._val_pool = ConcatDataset(
                        [
                            ImageFolder(
                                root=self.train_dir, transform=self.val_transform
                            ),
                            ImageFolder(
                                root=self.val_dir, transform=self.val_transform
                            ),
                        ]
                    )

                    self.set_fold(self.hparams.fold_index)
                else:
                    self.train_dataset = ImageFolder(
                        root=self.train_dir, transform=self.train_transform
                    )
                    self.val_dataset = ImageFolder(
                        root=self.val_dir, transform=self.val_transform
                    )
                    self._classes = self.train_dataset.classes

                    if self.hparams.use_weighted_sampler:
                        self.sampler = self._create_weighted_sampler(
                            self.train_dataset.targets
                        )

            elif self.data_dir.exists():
                full_ds = ImageFolder(root=self.data_dir)
                self._classes = full_ds.classes

                total_size = len(full_ds)
                actual_val_split = (
                    self.hparams.val_split
                    if self.hparams.val_split is not None
                    else 0.2
                )
                val_size = int(total_size * actual_val_split)
                train_size = total_size - val_size

                train_indices, val_indices = random_split(
                    range(total_size),
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.hparams.seed),
                )

                self.train_dataset = Subset(
                    ImageFolder(root=self.data_dir, transform=self.train_transform),
                    train_indices,
                )
                self.val_dataset = Subset(
                    ImageFolder(root=self.data_dir, transform=self.val_transform),
                    val_indices,
                )
            else:
                raise FileNotFoundError(f"Data directory {self.data_dir} not found.")

        if stage == "test" or stage is None:
            if self.test_dir.exists():
                self.test_dataset = ImageFolder(
                    root=self.test_dir, transform=self.val_transform
                )
                self._classes = self.test_dataset.classes

        if stage == "predict":
            predict_root = self.test_dir if self.test_dir.exists() else self.data_dir
            if predict_root.exists():
                self.predict_dataset = ImageFolder(
                    root=predict_root, transform=self.val_transform
                )

    def set_fold(self, fold_index: int) -> None:
        """Swaps active subsets for K-Fold CV and updates sampler if needed."""
        if self._splits is None or self._train_pool is None or self._val_pool is None:
            raise RuntimeError("Base splits not initialized. Call setup() first.")

        if not (0 <= fold_index < self.hparams.k_fold):
            raise ValueError(
                f"Fold index {fold_index} out of range (0-{self.hparams.k_fold - 1})"
            )

        self.hparams.fold_index = fold_index
        train_idx, val_idx = self._splits[fold_index]

        self.train_dataset = Subset(self._train_pool, train_idx)
        self.val_dataset = Subset(self._val_pool, val_idx)

        if self.hparams.use_weighted_sampler and self._pool_labels is not None:
            train_labels = [self._pool_labels[i] for i in train_idx]
            self.sampler = self._create_weighted_sampler(train_labels)

    def _create_weighted_sampler(
        self, labels: list[int] | np.ndarray
    ) -> WeightedRandomSampler:
        """Creates a WeightedRandomSampler to address class imbalance."""
        targets = np.array(labels)
        unique_targets, counts = np.unique(targets, return_counts=True)
        weight = 1.0 / counts
        weight_map = {t: weight[i] for i, t in enumerate(unique_targets)}
        samples_weight = np.array([weight_map[t] for t in targets])
        return WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True if self.sampler is None else False,
            sampler=self.sampler,
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
