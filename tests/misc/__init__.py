import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TemplateClassifier(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 6,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(y_hat, y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss)

        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc)

        self.test_f1(y_hat, y)
        self.log("test_f1", self.test_f1)

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
                "scheduler": ReduceLROnPlateau(optimizer, mode="min"),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
