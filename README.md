# lightning-uv-wandb-template <!-- omit from toc -->

![title](./assets/readme/title.jpg)

<!-- Refer to https://shields.io/badges for usage -->

![Year, Term, Course](https://img.shields.io/badge/AY2526--T2-GORUSHI-blue)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white) ![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-f37626?logo=jupyter&logoColor=white)

A deep learning project template integrating PyTorch Lightning, `uv`, and Weights & Biases (WandB). This structure is designed to be modular and easy to adapt for any image classification task.

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Running the Project](#3-running-the-project)
  - [3.1. Prerequisites](#31-prerequisites)
  - [3.2. Reproducing the Results](#32-reproducing-the-results)
- [4. References](#4-references)

## 1. Introduction

`lightning-uv-wandb-template` is a boilerplate for deep learning projects. It integrates three core tools out-of-the-box:

1. **PyTorch Lightning**: For the training loop and CLI-driven configuration.
2. **uv**: For fast dependency management, formatting (`ruff`), and testing (`pytest`).
3. **Weights & Biases (WandB)**: For experiment tracking and logging.

This foundation is modular; you can seamlessly adapt or swap these tools (e.g., to `pip` package manager or `MLflow` experiment tracker) as your project evolves.

## 2. Project Structure

A high-level overview of the repository organization:

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
├── scripts/                # Utility scripts for batch training, grid search, and validation
├── src/                    # Source code package
│   └── lightning_uv_wandb_template/     # Main library package
│       ├── data/           # LightningDataModules, transforms, and data utilities
│       ├── models/         # LightningModule wrapping definitions
│       ├── schemas/        # Pydantic validation schemas
│       └── utils/          # Shared utilities (logging, constants)
├── tests/                  # Hierarchical test suite mirroring src/ structure
│   ├── data/               # Data structure tests
│   ├── integration/        # End-to-end integration tests
│   ├── misc/               # Miscellaneous health and sanity checks
│   ├── models/             # Unit tests for models and architectures
│   └── utils/              # Utility logic tests
└── AGENTS.md               # Root entry point for AI assistants
```

For a detailed look at the internal design, public APIs, and architectural decisions, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 3. Running the Project

### 3.1. Prerequisites

To reproduce our results, you will need the following installed:

1. **Git:** Used to clone this repository.

2. **Python:** We require Python `3.11` for this project. You do not need to install the specific version as it will be installed by `uv`.

3. **uv:** The package manager we used. Installation instructions can be found at <https://docs.astral.sh/uv/getting-started/installation/>.

### 3.2. Reproducing the Results

1. Clone this repository:

   ```bash
   git clone https://github.com/qu1r0ra/lightning-uv-wandb-template
   ```

2. Navigate to the project root and install all dependencies:

   ```bash
   cd lightning-uv-wandb-template
   uv sync
   ```

3. Prepare the dataset:
   Ensure your raw data is in `data/by_class` (or use `make data-init` if you have a `data.zip` there).

   ```bash
   make data-init
   ```

4. Run through the Jupyter notebooks in `notebooks/reproducibility/` in numerical order:
   1. `01_Exploratory_Data_Analysis.ipynb`
   2. `02_Model_Selection_Training.ipynb`
   3. ...

   _Notes_
   - When running a notebook, select `.venv` in root as the kernel.
   - More instructions can be found in each notebook.

## 4. References

[1] Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., . . . Chintala, S. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation [Conference paper]. _29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24)_. <https://doi.org/10.1145/3620665.3640366>

[2] Falcon, W., & The PyTorch Lightning team. (2019). _PyTorch Lightning_ (Version 1.4) [Computer software]. <https://doi.org/10.5281/zenodo.3828935>
