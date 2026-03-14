import torch
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_train_metrics(num_classes: int) -> MetricCollection:
    """Factory for training metrics."""
    return MetricCollection(
        {"acc": Accuracy(task="multiclass", num_classes=num_classes)}
    )


def get_eval_metrics(num_classes: int) -> MetricCollection:
    """Factory for evaluation metrics (val/test)."""
    return MetricCollection(
        {
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "precision": Precision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "recall": Recall(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
        }
    )


def format_metrics(
    metrics_dict: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, float]:
    """Format MetricCollection output into a dict of floats with an optional prefix."""
    return {f"{prefix}{k}": float(v) for k, v in metrics_dict.items()}
