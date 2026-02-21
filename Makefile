.PHONY: help train fast-dev grid-search test test-all lint format clean clean-artifacts

ifeq (, $(shell which uv))
    PYTHON = python3
    PIP = pip3
else
    PYTHON = uv run python
    PIP = uv pip
endif

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

train:
	$(PYTHON) scripts/train_full.py

fast-dev:
	$(PYTHON) scripts/train_fast_dev_run.py

grid-search:
	$(PYTHON) scripts/train_grid_search.py

test:
	uv run pytest -v -s

test-all:
	uv run pytest -v -s -m ''

lint:
	uv run ruff check .

format:
	uv run ruff check --fix . && uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov lightning_logs wandb

clean-artifacts:
	rm -rf artifacts/checkpoints artifacts/logs artifacts/predictions
