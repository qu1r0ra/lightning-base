import torch
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightning_uv_wandb_template.utils.constants import (
    DEFAULT_DROPOUT,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    NUM_CLASSES,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
)
from lightning_uv_wandb_template.utils.metrics import EVAL_METRICS, TRAIN_METRICS


class TemplateClassifier(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = NUM_CLASSES,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        dropout: float = DEFAULT_DROPOUT,
        scheduler_patience: int = SCHEDULER_PATIENCE,
        scheduler_factor: float = SCHEDULER_FACTOR,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.save_hyperparameters(ignore=["model"])

        self.train_metrics = TRAIN_METRICS.clone(prefix="train_")
        self.val_metrics = EVAL_METRICS.clone(prefix="val_")
        self.test_metrics = EVAL_METRICS.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        output = self.train_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        output = self.val_metrics(y_hat, y)
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss)
        output = self.test_metrics(y_hat, y)
        self.log_dict(output)

        return loss

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.scheduler_factor,
                    patience=self.scheduler_patience,
                ),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
