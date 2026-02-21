# Gemini System Instructions

Welcome to the `lightning-uv-wandb-template` template. When assisting within this repository, do not rely solely on your baseline training assumptions.

## 1. Mandatory Reading

You must read and adhere to [`CONTEXT.md`](CONTEXT.md). It establishes critical project boundaries:

- **Tooling**: We use `uv` exclusively. No exceptions for `pip` or `conda`.
- **Data Safety**: Do not script custom dataset splits manually. Datasets are dynamically managed by the `TemplateDataModule`, which can leverage existing PyTorch built-ins like `random_split`.

## 2. Architecture Awareness

Read [`ARCHITECTURE.md`](../ARCHITECTURE.md) to comprehend how the PyTorch Lightning boilerplate is constructed and how configuration-as-code ties into the main module components.

## 3. Execution Protocol

- Lean on the included `Makefile` (e.g., `make train`, `make test`) as your primary task runner interface.
- Complete operations and workflows cleanly by invoking `make format` and verifying logic through `make test`.
