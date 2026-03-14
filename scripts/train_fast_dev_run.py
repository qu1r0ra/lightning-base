import os
import sys

import typer

from lightning_uv_wandb_template.utils.logger import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="Fast Development Run")


@app.command()
def run(config: str = "configs/baselines/template_baseline.yaml") -> None:
    """Run a fast development run (one batch) to check for errors."""
    cmd = (
        f"{sys.executable} scripts/train.py fit "
        f"--config {config} --trainer.fast_dev_run true"
    )
    logger.info(f"Executing: {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        logger.error(f"Command failed with exit code {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    app()
