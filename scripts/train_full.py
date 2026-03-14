import os
import sys

import typer

from lightning_uv_wandb_template.utils.logger import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="Full Training Pipeline")


@app.command()
def train(config: str = "configs/baselines/template_baseline.yaml") -> None:
    """Run full training pipeline with PyTorch Lightning CLI."""
    cmd = (
        f"uv run python src/lightning_uv_wandb_template/engines/cli.py fit "
        f"--config {config}"
    )
    logger.info(f"Executing: {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logger.error(f"Command failed with exit code {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    app()
