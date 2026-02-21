import itertools
import os
import sys

import yaml

from lightning_uv_wandb_template.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    A rudimentary grid search runner that sweeps over the search_space defined in
    configs/grids/template_grid.yaml and invokes the CLI engine using permutations.
    """
    grid_path = "configs/grids/template_grid.yaml"
    baseline_path = "configs/baselines/template.yaml"

    if not os.path.exists(grid_path):
        logger.error(f"Grid configuration not found at {grid_path}")
        sys.exit(1)

    with open(grid_path) as f:
        grid_config = yaml.safe_load(f)

    search_space = grid_config.get("search_space", {})
    fixed_params = grid_config.get("fixed_params", {})

    logger.info(f"Loaded Grid Configuration from {grid_path}")
    logger.info(f"Fixed parameters: {fixed_params}")
    logger.info(f"Search space: {search_space}")

    keys, values = zip(*search_space.items(), strict=True)
    permutations = [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]

    logger.info(f"Starting {len(permutations)} experiments...")

    for i, combination in enumerate(permutations):
        logger.info(f"\\n--- Running Experiment {i + 1}/{len(permutations)} ---")
        merged_params = {**fixed_params, **combination}

        args = []
        for key, value in merged_params.items():
            if key in ["max_epochs"]:
                args.append(f"--trainer.{key}={value}")
            elif key in ["batch_size", "seed"]:
                args.append(f"--data.init_args.{key}={value}")
            else:
                args.append(f"--model.init_args.{key}={value}")

        cli_arg_string = " ".join(args)
        cmd = (
            f"uv run python src/lightning_uv_wandb_template/engines/cli.py fit "
            f"--config {baseline_path} {cli_arg_string}"
        )

        logger.info(f"Grid Command {i + 1}: {cmd}")
        res = os.system(cmd)
        if res != 0:
            logger.error(
                f"Experiment {i + 1} failed with code {res}. "
                "Terminating grid search early."
            )
            sys.exit(res)

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
