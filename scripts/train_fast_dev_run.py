import os
import sys

from lightning_uv_wandb_template.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config_path = "configs/baselines/template.yaml"
    cmd = (
        f"uv run python src/lightning_uv_wandb_template/engines/cli.py fit "
        f"--config {config_path} "
        f"--trainer.fast_dev_run=True "
        f"--trainer.logger=False "
        f"--data.num_workers=0"
    )
    logger.info(f"Executing Fast Dev Run: {cmd}")
    sys.exit(os.system(cmd))


if __name__ == "__main__":
    main()
