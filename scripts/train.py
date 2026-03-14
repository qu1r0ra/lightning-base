from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from lightning_uv_wandb_template.data.datamodule import TemplateDataModule
from lightning_uv_wandb_template.models.classifier import TemplateClassifier


def main() -> None:
    load_dotenv()
    LightningCLI(
        model_class=TemplateClassifier,
        datamodule_class=TemplateDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
