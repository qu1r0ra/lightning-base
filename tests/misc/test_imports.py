def test_imports() -> None:
    import lightning
    import torch
    import torchmetrics

    assert torch is not None
    assert torchmetrics is not None
    assert lightning is not None
