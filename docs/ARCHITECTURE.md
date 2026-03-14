# Project Architecture

This document describes the architectural design and directory structure of the `lightning-uv-wandb-template` template. It serves as a guide for developers and contributors to understand the core PyTorch Lightning components and their interactions in this boilerplate.

## Directory Structure Overview

```text
.
├── assets/                 # Static assets (diagrams, readme images)
│   ├── figures/            # Result figures and plots
│   └── readme/             # Images for documentation
├── configs/                # Lightning CLI configuration files (.yaml)
│   ├── baselines/          # Default configurations for baseline models
│   └── grids/              # Sweep configurations for hyperparameter tuning
├── data/                   # Local dataset storage (e.g., train/, val/, test/)
├── docs/                   # Project documentation and specifications
│   ├── agents/             # AI agent-specific directives and context
│   └── ARCHITECTURE.md     # Technical design and architecture overview
├── notebooks/              # Jupyter notebooks for EDA, prototyping, and analysis
│   ├── dev/                # Experimental and scratchpad notebooks
│   └── reproducibility/    # Numbered notebooks to replicate project results
├── scripts/                # Entry points for training (train.py), grid search, and validation
├── src/                    # Source code package
│   └── lightning_uv_wandb_template/     # Main library package
│       ├── data/           # LightningDataModules, transforms, and data utilities
│       ├── models/         # LightningModule wrapping definitions
│       └── utils/          # Shared utilities (logging, constants)
├── tests/                  # Hierarchical test suite mirroring src/ structure
│   ├── data/               # Data structure tests
│   ├── integration/        # End-to-end integration tests
│   ├── misc/               # Miscellaneous health and sanity checks
│   ├── models/             # Unit tests for models and architectures
│   └── utils/              # Utility logic tests
└── AGENTS.md               # Root entry point for AI assistants
```

## Core Design Principles

### 1. Unified Public API

Each subpackage in `src/lightning_uv_wandb_template/` (e.g., `data`, `models`, `utils`) uses `__init__.py` to expose a clean, flattened public API where applicable.

- **Internal developers** use relative imports where appropriate or `from lightning_uv_wandb_template.x import y` to avoid circular dependencies.
- **External consumers** (scripts, tests, notebooks) use the clean package-level imports.

### 2. Deep Learning Service (Lightning Centric)

The pipeline is built using **PyTorch Lightning** for state-of-the-art reproducibility and boilerplate reduction.

- **LightningModule (`TemplateClassifier`)**: The core class handling the training loop, optimization, logging, and metrics. It wraps any standard `nn.Module` and utilizes centralized `MetricCollection` for streamlined multi-metric tracking (Accuracy, F1, etc.).
- **Lightning CLI**: The main entry point is `scripts/train.py`, which uses configuration files in `configs/` to promote "Configuration as Code". The CLI supports overrides via command-line arguments.

### 3. Data Management & Reproducibility

- **DataModule (`TemplateDataModule`)**:
  - Handles dataset splitting (Train/Val/Test).
  - Automatically identifies pre-split folders or executes dynamic `random_split` based on configuration.
- **Seed Everything**: A centralized `seed_everything` utility inside PyTorch Lightning ensures deterministic behavior across Python, Numpy, specific libraries, and PyTorch.

### 5. Code Quality & Standards

- **Type Safety**: The codebase utilizes comprehensive type hinting using modern Python 3.10+ syntax (e.g., `list[str] | None`) to improve readability and developer experience.
- **Formatting**: Code is formatted and linted using `ruff` to ensure PEP 8 compliance.
- **Testing**: A comprehensive test suite (`tests/`) covers unit tests (logic verification) and slower integration tests (end-to-end pipeline).

## Tools & Dependencies

- **uv**: Package management and environment isolation.
- **PyTorch Lightning**: Deep learning framework.
- **Albumentations**: Fast and flexible image augmentation library.
- **WandB**: Experiment tracking and visualization.
