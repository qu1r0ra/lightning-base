# Metacontext

This document captures high-level insights, architectural decisions, and documented bottlenecks encountered during the development of this project. It serves as a persistent memory to avoid re-treading documented dead-ends and to understand the non-obvious "why" behind specific design choices.

## Key Discoveries and Bottlenecks

### 1. Data-Level Constraints

- **Insight**: Identify if performance is capped by inherent dataset properties (e.g., label ambiguity, limited resolution, or class overlap).
- **Documentation**: Record empirical proof of performance plateaus to avoid redundant tuning beyond mathematical limits.
- **Future Direction**: Document architectural shifts required to break the ceiling (e.g., transitioning to different learning frameworks).

### 2. Resolution and Model Capacity

- **Insight**: Document the relationship between input resolution and model performance for specific architectures.
- **Verdict**: Specify the optimal resolution and the reasoning (e.g., kernel size alignment or feature dilation).

### 3. Data Augmentation Strategy

- **Insight**: Use `albumentations` via an adapter for more robust and varied augmentations (Rotation, Symmetry, Dropout, Jitter) compared to standard `torchvision` transforms.
- **Benefit**: Significantly improves generalization on small or noisy datasets.

### 4. Evaluation and Imbalance Handling

- **Insight**: Use `StratifiedKFold` to maintain class distribution across folds. Use `WeightedRandomSampler` during training if classes are imbalanced.
- **Convention**: Set `k_fold > 1` in the `TemplateDataModule` to enable CV mode.

## Implementation Standards

### 1. Scripting Pattern: "Raise in Logic, Exit in Main"

- **Standard**: Functions in `src/` and `scripts/` should raise descriptive exceptions instead of calling `sys.exit()`.
- **Reasoning**: Ensures logic is unit-testable and reusable without killing the parent process.
- **Wrapper**: Handle exit codes exclusively in the `if __name__ == "__main__":` block or via task runners.

### 2. Environment and Infrastructure

- **Quirks**: Document any hardware-specific or OS-specific behavior (e.g., MPS vs CUDA vs CPU quirks).
- **Reproducibility**: Note any deviations from the standard environment that were required for specific experiments.
- **UV Path**: `Makefile` now uses a shortened, OS-agnostic path resolution for `uv`.

- **Standard**: Define metrics in `src/lightning_uv_wandb_template/utils/metrics.py` using factory functions.
- **Reasoning**: Ensures metrics are initialized with the correct `num_classes` at runtime, preventing mismatches in polymorphic tasks.
- **Integration**: `LightningModule` subclasses call these factories during `__init__`.

### 4. Integration Testing (Smoke Tests)

- **Insight**: Unit tests verify isolated logic, but don't catch loop failures.
- **Standard**: Use `fast_dev_run=True` in integration tests (`tests/integration/`) with mock data to smoke-test the entire training loop efficiently.

### 5. Pydantic Schemas

- **Insight**: Moved validation logic to a dedicated `src/lightning_uv_wandb_template/schemas/` package.
- **Benefit**: Separates configuration validation from general utilities, providing a structured way to handle hyperparameters outside of YAML.

### 6. Entry Point Reorganization

- **Insight**: Decoupled the core `LightningCLI` runner from top-level project scripts.
- **Verdict**: The main entry point is `scripts/train.py`, which is invoked by higher-level runners (e.g., `train_full.py`, `train_grid_search.py`) via subprocesses. This ensures that experiment logic remains isolated and cleanly configurable.

## Architectural Debt and Design Choices

### 1. [Constraint/Decision Name]

- **Context**: Why was this choice made? what were the alternatives?
- **Implications**: How does this affect future development or maintenance?

## Current Status

- **Deep Learning**: `TemplateClassifier` implemented with centralized metrics and dynamic factory patterns.
- **Classical ML**: Scikit-learn integrated for advanced splitting (Stratified K-Fold).
- **Data Pipeline**: Automated with `make data-init`. Using Albumentations for augmentation.
- **Validation**: Strict Pydantic schemas implemented in `schemas/config.py`.
- **Infrastructure**: Entry points reorganized under `scripts/` with decoupled CLI logic.
- **Testing**: Passing 6 tests, including end-to-end training loop smoke tests.
- **Quality**: Verified clean state (formatting, linting, notebook sync) via `make full-check`.
