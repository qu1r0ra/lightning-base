import torch
import torch.nn as nn
from torchvision.models import resnet18

from lightning_uv_wandb_template.models.classifier import TemplateClassifier


def test_template_classifier_init() -> None:
    model = resnet18(num_classes=6)
    classifier = TemplateClassifier(
        model=model,
        num_classes=6,
        lr=1e-3,
        weight_decay=0.01,
        dropout=0.1,
    )

    assert classifier.hparams.num_classes == 6
    assert classifier.lr == 1e-3
    assert classifier.weight_decay == 0.01
    assert classifier.dropout == 0.1
    assert isinstance(classifier.model, nn.Module)
    assert isinstance(classifier.loss, nn.CrossEntropyLoss)


def test_template_classifier_forward() -> None:
    model = resnet18(num_classes=6)
    classifier = TemplateClassifier(model=model, num_classes=6)

    batch_size = 4
    channels, height, width = 3, 224, 224
    dummy_input = torch.randn(batch_size, channels, height, width)

    with torch.no_grad():
        output = classifier(dummy_input)

    assert output.shape == (
        batch_size,
        6,
    ), "Output shape should match (batch_size, num_classes)"
