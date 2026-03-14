.PHONY: help train fast-dev grid-search test test-all lint format clean clean-artifacts

UV := $(shell command -v uv || echo $(HOME)/.local/bin/uv)
PYTHON = $(UV) run python
PIP = $(UV) pip

help:
	@echo "Available commands:"
	@echo "  make train        		- Run full training pipeline"
	@echo "  make fast-dev     		- Run fast dev run to check for errors"
	@echo "  make grid-search  		- Run grid search experiment"
	@echo "  make test         		- Run fast tests (slow tests skipped by default)"
	@echo "  make test-all     		- Run all tests including slow ones"
	@echo "  make lint         		- Run linting (ruff check)"
	@echo "  make format       		- Run formatting (ruff format)"
	@echo "  make clean        		- Remove temporary files and logs"
	@echo "  make clean-artifacts  	- Remove all generated artifacts (models, checkpoints)"
	@echo "  make data-init    		- Unzip, setup, and split data"
	@echo "  make sync-nb      		- Synchronize notebooks with Jupytext scripts"

train:
	$(PYTHON) scripts/train_full.py

fast-dev:
	$(PYTHON) scripts/train_fast_dev_run.py

grid-search:
	$(PYTHON) scripts/train_grid_search.py

test:
	$(UV) run pytest -v -s

test-all:
	$(UV) run pytest -v -s -m ''

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff check --fix . && $(UV) run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov lightning_logs wandb

clean-artifacts:
	rm -rf artifacts/checkpoints artifacts/logs artifacts/predictions

data-init:
	$(PYTHON) src/lightning_uv_wandb_template/data/utils.py init

sync-nb:
	$(UV) run jupytext --sync notebooks/**/*.ipynb
