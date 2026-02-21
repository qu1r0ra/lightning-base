import os
import sys

from lightning_uv_wandb_template.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config_path = "configs/baselines/template.yaml"
    cmd = (
        f"uv run python src/lightning_uv_wandb_template/engines/cli.py fit "
        f"--config {config_path}"
    )
    logger.info(f"Executing: {cmd}")
    sys.exit(os.system(cmd))


if __name__ == "__main__":
    main()
