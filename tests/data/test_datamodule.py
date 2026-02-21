from pathlib import Path

from lightning_uv_wandb_template.data.datamodule import TemplateDataModule


def test_datamodule_defaults() -> None:
    dm = TemplateDataModule()

    assert dm.hparams.batch_size == 32
    assert dm.hparams.seed == 42
    assert dm.hparams.val_split == 0.2
    assert dm.hparams.num_workers == 4
    assert dm.hparams.k_fold == 1
    assert dm.hparams.fold_index == 0


def test_datamodule_paths(tmp_path: Path) -> None:
    dm = TemplateDataModule(data_dir=tmp_path)

    assert dm.data_dir == tmp_path
    assert dm.train_dir == tmp_path / "train"
    assert dm.val_dir == tmp_path / "val"
    assert dm.test_dir == tmp_path / "test"
