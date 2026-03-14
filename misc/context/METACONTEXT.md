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

### 3. Centralized Metric Management

- **Standard**: Define all training and evaluation metrics in `src/lightning_uv_wandb_template/utils/metrics.py` using `MetricCollection`.
- **Integration**: `LightningModule` subclasses should clone these collections to ensure consistent metric computation across different stages (train/val/test).

## Architectural Debt and Design Choices

### 1. [Constraint/Decision Name]

- **Context**: Why was this choice made? what were the alternatives?
- **Implications**: How does this affect future development or maintenance?

## Current Status

- **Deep Learning**: Advanced `TemplateClassifier` implemented with centralized metrics and AdamW/ReduceLROnPlateau.
- **Classical ML**: Scikit-learn integrated for advanced splitting (Stratified K-Fold).
- **Data Pipeline**: Automated with `make data-init` (unzip, setup, split). Using Albumentations for augmentation.
- **Next Task**: Model training and evaluation on specific datasets.
